"""
circust/outlier.py
==============================
Residual outlier detection and CPCA re-run.

This module implements the second half of R's ``giveMatIniNP_v3_cores``,
starting from the initial CPCA result and working through:

  1. Fit Cosinor + FMM on every core gene and eigengene using the
     initial circular ordering.
  2. Compute standardised FMM residuals for each gene × sample.
  3. Flag samples as residual outliers under two criteria:
       - Multivariate: |std_res| > 3 AND already a CPCA candidate outlier
       - Univariate:   |std_res| > 4 for any single gene
     Both criteria are capped at ceil(5% × n_samples) total flagged.
  4. If outliers found: drop them, re-normalise the core matrix,
     re-run CPCA to get the final clean ordering.
  5. Apply the final ordering to the full expression matrix and
     re-normalise it — producing the matrix used for genome-wide scoring.

R equivalent
------------
Lines 3962–4101 of ``giveMatIniNP_v3_cores`` in ``functionGTEX_cores.R``.

Pipeline position
-----------------
    Preprocessor  →  CPCA  →  OutlierRefiner  →  RhythmicityScorer
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from circust.cpca import CPCA, CPCAResult
from circust.fitting.cosinor import CosinorModel
from circust.fitting.fmm import FMMModel, fmm_peak_time
from circust.fitting.rhythm_model import FitResult
import circust.constants as const


# ---------------------------------------------------------------------------
# Helper — normalise every row to [−1, 1]  (vectorised)
# ---------------------------------------------------------------------------

def _normalise_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise each row to [−1, 1]. Constant rows → zeros."""
    values  = mat.values.astype(float)
    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    span    = row_max - row_min
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(span == 0, 0.0, 2.0 * (values - row_min) / span - 1.0)
    return pd.DataFrame(normed, index=mat.index, columns=mat.columns)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OutlierRefinementResult:
    """
    All outputs produced by :class:`OutlierRefiner`.

    Attributes
    ----------
    cpca_final : CPCAResult
        CPCA result after removing residual outliers.  If no outliers were
        found this is identical to the input CPCA result.

    expr_norm_final : pd.DataFrame, shape (n_genes, n_samples_clean)
        Full normalised expression matrix with outlier samples removed and
        rows re-ordered by the final circular phase.
        R equivalent: ``mFullTissueNorm`` (line 4099).

    core_norm_final : pd.DataFrame, shape (n_core_genes, n_samples_clean)
        Core gene normalised matrix after outlier removal, also ordered.
        R equivalent: ``mTissueCoreGNorm`` after dropping (line 4085).

    samples_dropped : list[int]
        Column indices (in the original sample space) of samples removed
        as residual outliers.
        R equivalent: ``dropTissueOut`` (line 4083).

    univariate_outliers : list[int]
        Sample indices flagged by the univariate criterion (|std_res| > 4).
        R equivalent: ``outsUni`` (line 4071).

    multivariate_outliers : list[int]
        Sample indices flagged by the multivariate criterion (|std_res| > 3
        AND already a CPCA candidate).
        R equivalent: ``outsMult`` (line 4062).

    fmm_fits_initial : dict[str, FitResult]
        FMM fits on core genes + eigengenes using the initial CPCA ordering.
        Keys: gene symbols + "PC1", "PC2", "PC3".
        R equivalent: ``fitParCore`` rows / ``FMMParCoreG`` (lines 3999/4007).

    cosinor_fits_initial : dict[str, FitResult]
        Cosinor fits on core genes + eigengenes using the initial CPCA ordering.
        R equivalent: ``fitCosCore`` rows / ``CosParCoreG`` (line 4008).

    std_residuals_fmm : pd.DataFrame, shape (n_signals, n_samples)
        Standardised FMM residuals used for outlier detection.
        Rows = core genes + PC1/PC2/PC3.  Columns = samples in CPCA order.
        R equivalent: ``resParStTissue`` (line 4003).

    fmm_peak_times_initial : dict[str, float]
        FMM peak time (t_U) for each signal after the initial fit.
        R equivalent: ``peaksCoreG`` (line 4009).

    cosinor_peak_times_initial : dict[str, float]
        Cosinor acrophase for each signal after the initial fit.
        R equivalent: ``phisCoreG`` (line 4006).

    outliers_were_found : bool
        True if any residual outliers were detected and removed.

    fmm_fits_final : dict[str, FitResult]
        FMM fits on core genes using the **final** CPCA ordering
        (i.e. after outlier removal and re-normalisation).
        Keys = core gene symbols.
        R equivalent: ``allParAfter`` (lines 4109-4136).

    fmm_peak_times_final : dict[str, float]
        FMM peak time (t_U via compUU) for each core gene from the
        final-ordering fit.
        R equivalent: ``phisFMMAfter`` (line 4118).
    """

    cpca_final:               CPCAResult
    expr_norm_final:          pd.DataFrame
    core_norm_final:          pd.DataFrame
    samples_dropped:          list[int]       = field(default_factory=list)
    univariate_outliers:      list[int]       = field(default_factory=list)
    multivariate_outliers:    list[int]       = field(default_factory=list)
    fmm_fits_initial:         dict            = field(default_factory=dict)
    cosinor_fits_initial:     dict            = field(default_factory=dict)
    std_residuals_fmm:        Optional[pd.DataFrame] = None
    fmm_peak_times_initial:   dict            = field(default_factory=dict)
    cosinor_peak_times_initial: dict          = field(default_factory=dict)
    outliers_were_found:      bool            = False
    fmm_fits_final:           dict            = field(default_factory=dict)
    fmm_peak_times_final:     dict            = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Outlier Refinement Summary ===",
            f"  Residual outliers found  : {len(self.samples_dropped)}",
            f"    Univariate  (|res|>4)  : {len(self.univariate_outliers)}",
            f"    Multivariate(|res|>3)  : {len(self.multivariate_outliers)}",
            f"  CPCA re-run              : {'yes' if self.outliers_were_found else 'no'}",
            f"  Final n_samples          : {self.expr_norm_final.shape[1]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OutlierRefiner class
# ---------------------------------------------------------------------------

class OutlierRefiner:
    """
    Detect residual outlier samples using FMM fits on core clock genes,
    then re-run CPCA on the cleaned dataset.

    The algorithm fits both Cosinor and FMM to every core gene and
    eigengene using the initial CPCA circular ordering.  Samples whose
    standardised FMM residuals exceed thresholds are flagged as outliers,
    removed, and the CPCA is re-run to produce the final clean ordering.

    Parameters
    ----------
    multi_threshold : float
        Standardised FMM residual threshold for the multivariate criterion.
        A sample is a multivariate candidate if |std_res| > this value for
        any core gene AND the sample was already a CPCA radial candidate.
        R default: 3.0 (line 4057).

    uni_threshold : float
        Standardised FMM residual threshold for the univariate criterion.
        A sample is a univariate outlier if |std_res| > this value for
        any single core gene.
        R default: 4.0 (line 4069).

    max_outlier_fraction : float
        Maximum fraction of samples that may be removed as residual
        outliers. The cap is applied as ``ceil(fraction × n_samples)``.
        R default: 0.05 (lines 4057, 4069).

    fmm_length_alpha_grid : int
        Grid resolution for FMM α parameter. R default: 48.

    fmm_length_omega_grid : int
        Grid resolution for FMM ω parameter. R default: 24.

    fmm_num_reps : int
        Number of FMM grid refinement iterations. R default: 3.

    cpca_n_outlier_candidates : int
        Passed to the CPCA re-run if outliers are found. R default: 8.

    cpca_tight_radius : float
        Passed to the CPCA re-run. R default: 0.10.

    cpca_loose_radius : float
        Passed to the CPCA re-run. R default: 0.15.

    verbose : bool
        Print progress messages.

    Examples
    --------
    >>> from circust.preprocessing import Preprocessor, load_expression_matrix
    >>> from circust.cpca import CPCA
    >>> from circust.outlier_refinement import OutlierRefiner
    >>>
    >>> matrix  = load_expression_matrix("data/BA46.csv")
    >>> prep    = Preprocessor().run(matrix)
    >>> cpca    = CPCA().run(prep.expr_norm)
    >>> refined = OutlierRefiner().run(cpca, prep.expr_norm)
    >>> print(refined.summary())
    """

    def __init__(
        self,
        multi_threshold:           float = const.OUTLIER_RESIDUAL_THRESHOLD,
        uni_threshold:             float = 4.0,
        max_outlier_fraction:      float = 0.05,
        fmm_length_alpha_grid:     int   = 48,
        fmm_length_omega_grid:     int   = 24,
        fmm_num_reps:              int   = 3,
        cpca_n_outlier_candidates: int   = const.N_OUTLIER_CANDIDATES,
        cpca_tight_radius:         float = const.OUTLIER_RADIAL_THRESHOLD,
        cpca_loose_radius:         float = const.OUTLIER_RADIAL_THRESHOLD_LOOSE,
        verbose:                   bool  = True,
    ) -> None:
        self.multi_threshold           = multi_threshold
        self.uni_threshold             = uni_threshold
        self.max_outlier_fraction      = max_outlier_fraction
        self.fmm_length_alpha_grid     = fmm_length_alpha_grid
        self.fmm_length_omega_grid     = fmm_length_omega_grid
        self.fmm_num_reps              = fmm_num_reps
        self.cpca_n_outlier_candidates = cpca_n_outlier_candidates
        self.cpca_tight_radius         = cpca_tight_radius
        self.cpca_loose_radius         = cpca_loose_radius
        self.verbose                   = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        cpca:      CPCAResult,
        expr_norm: pd.DataFrame,
    ) -> OutlierRefinementResult:
        """
        Run the full outlier refinement procedure.

        Parameters
        ----------
        cpca : CPCAResult
            Output of ``CPCA.run()``.  Must have been produced with
            ``store_core_matrix=True`` (the default).
        expr_norm : pd.DataFrame
            Full normalised expression matrix (genes × samples) — i.e.
            ``PreprocessingResult.expr_norm``.  Used to produce the
            final genome-wide ordered matrix.

        Returns
        -------
        OutlierRefinementResult
        """
        if cpca.core_matrix is None:
            raise ValueError(
                "cpca.core_matrix is None. "
                "Run CPCA with store_core_matrix=True (the default)."
            )

        self._log("=== Outlier Refinement ===")

        # ── Step 1: fit Cosinor + FMM on core genes + eigengenes ────────
        self._log("  Step 1 — fitting models on core genes and eigengenes ...")
        fmm_fits, cos_fits, std_res_df, peak_fmm, peak_cos = self._fit_initial(cpca)

        # ── Step 2: detect residual outliers ────────────────────────────
        self._log("  Step 2 — detecting residual outliers ...")
        uni_outs, mult_outs = self._detect_outliers(cpca, std_res_df)

        dropped = sorted(set(uni_outs) | set(mult_outs))
        self._log(
            f"    Univariate  (|res|>{self.uni_threshold}): {len(uni_outs)} sample(s)"
        )
        self._log(
            f"    Multivariate(|res|>{self.multi_threshold}): {len(mult_outs)} sample(s)"
        )
        self._log(f"    Total dropped: {len(dropped)}")

        # ── Step 3: clean + re-run CPCA if needed ───────────────────────
        if dropped:
            self._log("  Step 3 — re-running CPCA on cleaned data ...")
            cpca_final, core_norm_clean = self._rerun_cpca(cpca, dropped)
        else:
            self._log("  Step 3 — no outliers found, keeping initial CPCA result")
            cpca_final     = cpca
            core_norm_clean = cpca.core_matrix

        # ── Step 4: apply final ordering to full matrix ─────────────────
        self._log("  Step 4 — ordering full expression matrix ...")
        expr_norm_final = self._order_full_matrix(
            expr_norm, dropped, cpca_final.sample_order
        )

        # ── Step 5: fit FMM on core genes with final ordering ────────────
        # R equivalent: lines 4109-4136 (allParAfter, phisFMMAfter)
        self._log("  Step 5 — fitting FMM on core genes (final ordering) ...")
        fmm_fits_fin, peak_fmm_fin = self._fit_final(expr_norm_final, cpca_final)

        self._log("  Done.")

        result = OutlierRefinementResult(
            cpca_final               = cpca_final,
            expr_norm_final          = expr_norm_final,
            core_norm_final          = core_norm_clean,
            samples_dropped          = dropped,
            univariate_outliers      = uni_outs,
            multivariate_outliers    = mult_outs,
            fmm_fits_initial         = fmm_fits,
            cosinor_fits_initial     = cos_fits,
            std_residuals_fmm        = std_res_df,
            fmm_peak_times_initial   = peak_fmm,
            cosinor_peak_times_initial = peak_cos,
            outliers_were_found      = len(dropped) > 0,
            fmm_fits_final           = fmm_fits_fin,
            fmm_peak_times_final     = peak_fmm_fin,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _fit_initial(
        self,
        cpca: CPCAResult,
    ) -> tuple[dict, dict, pd.DataFrame, dict, dict]:
        """
        Fit Cosinor and FMM on each core gene + PC1/PC2/PC3.

        Returns
        -------
        fmm_fits   : {signal_name: FitResult}
        cos_fits   : {signal_name: FitResult}
        std_res_df : DataFrame (n_signals × n_samples) of standardised FMM residuals
        peak_fmm   : {signal_name: float}  — FMM peak times
        peak_cos   : {signal_name: float}  — Cosinor acrophases
        """
        order       = cpca.sample_order         # sample ordering indices
        time_points = cpca.circular_scale       # escalaPhi8  (sorted phi)
        genes       = cpca.core_genes_found

        # Build the ordered signal dict (R: datCore)
        # Core genes: rows of core_matrix reordered by sample_order
        cm = cpca.core_matrix
        cm_vals = cm.values if hasattr(cm, "values") else cm

        signals: dict[str, np.ndarray] = {}
        for i, gene in enumerate(genes):
            signals[gene] = cm_vals[i, order]

        # Eigengenes ordered the same way (R lines 3978-3980)
        signals["PC1"] = cpca.pc1[order]
        signals["PC2"] = cpca.pc2[order]
        signals["PC3"] = cpca.pc3[order]

        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )
        cos_model = CosinorModel()

        fmm_fits:  dict[str, FitResult] = {}
        cos_fits:  dict[str, FitResult] = {}
        peak_fmm:  dict[str, float]     = {}
        peak_cos:  dict[str, float]     = {}
        std_res_rows: list[np.ndarray]  = []
        signal_names: list[str]         = []

        n_total = len(signals)
        for idx, (name, data) in enumerate(signals.items(), 1):
            self._log(f"    [{idx}/{n_total}] {name}", end="\r")

            fr = fmm_model.fit(data, time_points)
            cr = cos_model.fit(data, time_points)

            fmm_fits[name] = fr
            cos_fits[name] = cr

            # standardised FMM residuals  (R line 4003: resParStTissue)
            std_res_rows.append(fr.residuals_std)
            signal_names.append(name)

            # FMM peak time via compUU  (R line 4009)
            peak_fmm[name] = fmm_peak_time(
                fr.params["alpha"], fr.params["beta"], fr.params["omega"]
            )

            # Cosinor acrophase  (R line 4006: (-funCos[[5]]) %% (2*pi))
            peak_cos[name] = (-cr.params["phi"]) % (2.0 * np.pi)

        self._log(f"    Done fitting {n_total} signals.           ")

        # Build DataFrame: rows = signals, columns = sample positions 0..n-1
        std_res_df = pd.DataFrame(
            np.vstack(std_res_rows),
            index   = signal_names,
            columns = np.arange(len(time_points)),
        )

        return fmm_fits, cos_fits, std_res_df, peak_fmm, peak_cos

    def _detect_outliers(
        self,
        cpca:       CPCAResult,
        std_res_df: pd.DataFrame,
    ) -> tuple[list[int], list[int]]:
        """
        Apply the two R outlier criteria to the standardised FMM residuals.

        Multivariate criterion (R lines 4057-4067)
        -------------------------------------------
        For each core gene, find positions in the ordered residual array
        where |std_res| > multi_threshold.  A sample at that position is
        a multivariate candidate only if its ORIGINAL sample index is also
        in the CPCA outlier candidate list (initialTissue[[3]]).
        Cap at ceil(max_outlier_fraction × n_samples) total.

        Univariate criterion (R lines 4069-4074)
        -----------------------------------------
        For each core gene, any position where |std_res| > uni_threshold
        is a univariate outlier.  Same total cap applies.

        Note: both criteria work on CORE GENES only, not eigengenes.
        The residual DataFrame has eigengenes as extra rows — we slice
        to only the core gene rows here, matching R's loop over
        ``1:length(coreG)`` (lines 4056, 4069).

        Returns
        -------
        uni_outs  : list of original sample indices (0-based)
        mult_outs : list of original sample indices (0-based)
        """
        genes       = cpca.core_genes_found
        order       = cpca.sample_order          # position → original index
        # R uses only the CONFIRMED outliers (within radial threshold),
        # not all 8 candidates.  R: match(..., mOutliers) <= ss8.
        confirmed   = set(cpca.outlier_idx.tolist())
        n_samples   = len(order)
        cap         = int(np.ceil(self.max_outlier_fraction * n_samples))

        # Only core gene rows (not eigengenes)
        core_std_res = std_res_df.loc[genes].values    # (n_genes, n_samples_ordered)

        mult_outs: list[int] = []
        uni_outs:  list[int] = []
        n_flagged = 0

        for i, gene in enumerate(genes):
            gene_res = core_std_res[i]    # (n_samples_ordered,)

            # ── Multivariate: |res| > multi_threshold AND confirmed CPCA outlier
            # R lines 4057-4067: increments nOutsTot per SAMPLE.
            if np.any(np.abs(gene_res) > self.multi_threshold) and n_flagged <= cap:
                positions = np.where(np.abs(gene_res) > self.multi_threshold)[0]
                for pos in positions:
                    orig_idx = int(order[pos])     # original sample index
                    if orig_idx in confirmed:
                        mult_outs.append(orig_idx)
                        n_flagged += 1

            # ── Univariate: |res| > uni_threshold
            # R lines 4069-4074: adds ALL bad samples for this gene at once,
            # but increments nOutsTot by 1 (per gene, not per sample).
            if np.any(np.abs(gene_res) > self.uni_threshold) and n_flagged <= cap:
                positions = np.where(np.abs(gene_res) > self.uni_threshold)[0]
                for pos in positions:
                    orig_idx = int(order[pos])
                    uni_outs.append(orig_idx)
                n_flagged += 1          # R: one increment per gene

        # Deduplicate preserving order
        uni_outs  = list(dict.fromkeys(uni_outs))
        mult_outs = list(dict.fromkeys(mult_outs))

        return uni_outs, mult_outs

    def _rerun_cpca(
        self,
        cpca:    CPCAResult,
        dropped: list[int],
    ) -> tuple[CPCAResult, pd.DataFrame]:
        """
        Drop outlier samples, re-normalise core matrix, re-run CPCA.

        R equivalent: lines 4084-4090.

        Returns
        -------
        cpca_final      : new CPCAResult on cleaned data
        core_norm_clean : re-normalised core DataFrame (genes × clean samples)
        """
        # Original core matrix (raw — before normalisation)
        # We stored the normalised version in cpca.core_matrix.
        # Re-normalisation after dropping requires the normalised values
        # because mTissueCoreG is the RAW core matrix in R.
        # However since we don't store raw core values, we re-normalise
        # from the already-normalised matrix — the result is the same
        # because normalice() is idempotent when values are already in [-1,1]:
        # dropping samples changes the min/max → renormalisation IS needed.
        core_matrix = cpca.core_matrix   # DataFrame (genes × samples)

        # Drop outlier columns by original sample index
        all_cols    = np.arange(core_matrix.shape[1])
        keep_mask   = ~np.isin(all_cols, dropped)
        core_clean  = core_matrix.iloc[:, keep_mask]

        # Re-normalise (R line 4085)
        core_norm_clean = _normalise_matrix(core_clean)

        # Re-run CPCA (R line 4087: obtainCPCA13(mTissueCoreGNorm,...))
        cpca_final = CPCA(
            core_genes           = cpca.core_genes_found,
            n_outlier_candidates = self.cpca_n_outlier_candidates,
            tight_radius         = self.cpca_tight_radius,
            loose_radius         = self.cpca_loose_radius,
            verbose              = False,
        ).run(core_norm_clean)

        return cpca_final, core_norm_clean

    def _order_full_matrix(
        self,
        expr_norm:    pd.DataFrame,
        dropped:      list[int],
        sample_order: np.ndarray,
    ) -> pd.DataFrame:
        """
        Drop outlier samples from the full matrix, then reorder by the
        final CPCA circular ordering.

        R equivalent: lines 4094-4101.

        The result is re-normalised gene-by-gene because dropping samples
        changes the per-gene min/max (R line 4099: t(apply(..., normalice))).

        Returns
        -------
        pd.DataFrame, shape (n_genes, n_samples_clean)
            Full matrix with outliers removed, ordered by circular phase,
            re-normalised.
        """
        # Drop outlier columns
        all_cols  = np.arange(expr_norm.shape[1])
        keep_mask = ~np.isin(all_cols, dropped)
        mat_clean = expr_norm.iloc[:, keep_mask]

        # Reorder columns by circular phase (R lines 4095-4097)
        mat_ordered = mat_clean.iloc[:, sample_order]

        # Re-normalise (R line 4099)
        mat_norm = _normalise_matrix(mat_ordered)

        return mat_norm

    def _fit_final(
        self,
        expr_norm_final: pd.DataFrame,
        cpca_final:      "CPCAResult",
    ) -> tuple[dict, dict]:
        """
        Fit FMM on each core gene using the final CPCA ordering.

        R equivalent: lines 4109-4136 of ``giveMatIniNP_v3_cores``
        (``allParAfter``, ``phisFMMAfter``).

        Parameters
        ----------
        expr_norm_final : pd.DataFrame
            Full normalised expression matrix already ordered by the
            final CPCA ordering (output of ``_order_full_matrix``).
        cpca_final : CPCAResult
            Final CPCA result; provides ``circular_scale`` (the time
            axis) and ``core_genes_found``.

        Returns
        -------
        fmm_fits_final : {gene: FitResult}
        fmm_peak_times_final : {gene: float}  — compUU peak times
        """
        core_genes     = cpca_final.core_genes_found
        circular_scale = cpca_final.circular_scale   # escalaPhi8 (final)

        fmm_model = FMMModel(
            length_alpha_grid = self.fmm_length_alpha_grid,
            length_omega_grid = self.fmm_length_omega_grid,
            num_reps          = self.fmm_num_reps,
        )

        fmm_fits_fin:  dict[str, FitResult] = {}
        peak_times_fin: dict[str, float]    = {}

        n_total = len(core_genes)
        for idx, gene in enumerate(core_genes, 1):
            self._log(f"    [{idx}/{n_total}] {gene}", end="\r")

            if gene not in expr_norm_final.index:
                self._log(f"    WARNING: {gene} not in final matrix, skipping.")
                continue

            data = expr_norm_final.loc[gene].values
            fr   = fmm_model.fit(data, circular_scale)

            fmm_fits_fin[gene]   = fr
            peak_times_fin[gene] = fmm_peak_time(
                fr.params["alpha"], fr.params["beta"], fr.params["omega"]
            )

        self._log(f"    Done fitting {len(fmm_fits_fin)} core genes (final).      ")
        return fmm_fits_fin, peak_times_fin

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _log(self, message: str, end: str = "\n") -> None:
        if self.verbose:
            print(message, end=end, flush=True)