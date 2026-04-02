"""
circust/candidate_selection.py
==============================
Stage 3.2: Candidate gene set selection via circular-sector filtering.

Selects genes whose expression profiles show strong rhythmic behaviour
according to both non-parametric (NP R²) and parametric (FMM) criteria.
Genes are partitioned into 8 sectors of 45° each on the circular phase
axis, and within each sector the top candidates are retained.

Algorithm
---------
1. Filter genes with NP R² above the median.
2. Compute the Cosinor acrophase for every surviving gene.
3. Define 8 equispaced 45° sectors centred on the circular median of
   all acrophases.
4. Within each sector, iterate through genes sorted by NP R² (descending):
   - Fit FMM and check ω > 0.1 (not too spiky) and R²_par > threshold.
   - Accept up to *tam* genes per sector, continuing beyond *tam* only
     while the gene's NP R² remains below the 75th percentile of the
     full distribution.
5. Collect all accepted genes across sectors.

R equivalent
------------
``minSetCand3_v2_cores()`` (lines 1242–1692 of ``functionGTEX_cores.R``).

Pipeline position
-----------------
    NonParametricScorer  →  **CandidateSelector**  →  ReferenceSetBuilder
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.fitting.cosinor import CosinorModel
from circust.fitting.fmm import FMMModel, fmm_peak_time


# ═══════════════════════════════════════════════════════════════════════════
# Circular statistics helpers
# ═══════════════════════════════════════════════════════════════════════════

def circular_median(angles: np.ndarray) -> float:
    """
    Circular median: the angle θ that minimises  Σ d(θ, θᵢ)  where
    *d* is the geodesic (arc-length) distance on the unit circle.

    Matches the definition used by R's ``circular::median.circular()``.

    Parameters
    ----------
    angles : (n,) array in [0, 2π).

    Returns
    -------
    float — circular median in [0, 2π).
    """
    angles = np.asarray(angles, dtype=np.float64) % (2.0 * pi)
    n = len(angles)
    if n == 0:
        return 0.0
    if n == 1:
        return float(angles[0])

    # Evaluate cost at every observed angle (O(n²) — adequate for ≤ 20 000
    # genes; the R circular package does the same).
    best_cost  = np.inf
    best_angle = angles[0]
    for candidate in angles:
        diffs = np.abs(angles - candidate)
        diffs = np.minimum(diffs, 2.0 * pi - diffs)
        cost  = diffs.sum()
        if cost < best_cost:
            best_cost  = cost
            best_angle = candidate

    return float(best_angle % (2.0 * pi))


def assign_sectors(
    peaks: np.ndarray,
    reference: float,
    n_sectors: int = 8,
) -> np.ndarray:
    """
    Assign each peak angle to one of *n_sectors* equi-spaced sectors
    centred on *reference*.

    The sectors are numbered 1 … n_sectors.  Sector 1 contains peaks
    closest to *reference* (in the counter-clockwise direction); sector
    *n_sectors* is the furthest.

    This vectorises the 8-cascading-if structure in R (lines 1300–1334).

    Parameters
    ----------
    peaks     : (n,) array of angles in [0, 2π).
    reference : float — circular median used as the origin.
    n_sectors : int — default 8 (45° each).

    Returns
    -------
    (n,) int array — sector labels in {1, 2, …, n_sectors}.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    # Rotate so that reference ≡ 0, then map to sector.
    rotated = (peaks - reference) % (2.0 * pi)
    # R assigns sector 1 to the arc CLOSEST to reference (highest rotated
    # values, i.e. those just below 2π), sector 8 to the arc centred at
    # reference − 7π/4.  The R code checks sectors from the top down:
    #   if rotated > boundary_2  → sector 1
    #   if boundary_3 < rotated < boundary_2 → sector 2
    #   …
    #   if rotated < boundary_8  → sector 8
    # The boundaries are at k·π/4 for k = 1…7.
    # Mapping: sector = n_sectors − floor(rotated / (2π / n_sectors)).
    sector_width = 2.0 * pi / n_sectors
    sectors = n_sectors - np.floor(rotated / sector_width).astype(int)
    # Clamp to [1, n_sectors] for floating-point edge cases.
    sectors = np.clip(sectors, 1, n_sectors)
    return sectors


def sector_boundaries(reference: float, n_sectors: int = 8) -> np.ndarray:
    """
    Return the *n_sectors* sector centre-points (m1…m8 in the R code).

    R equivalent: ``m1 = peakRef;  m2 = (peakRef − π/4) mod 2π; …``
    (lines 1280–1287).
    """
    return np.array([
        (reference - k * pi / (n_sectors / 2)) % (2.0 * pi)
        for k in range(n_sectors)
    ])


def reduce_sector(
    peaks: np.ndarray,
    r2: np.ndarray,
    names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iteratively remove the lowest-R² gene from a sector until all
    remaining genes exceed the 75th-percentile R² threshold **or** the
    sector already has ≤ 25 genes.

    R equivalent: ``reduceSectors_v2()`` (lines 140–161).

    Parameters
    ----------
    peaks : (k,) array of peak angles.
    r2    : (k,) array of R² values.
    names : (k,) array of gene name strings.

    Returns
    -------
    (peaks, r2, names) — filtered arrays.
    """
    peaks = np.array(peaks, dtype=np.float64)
    r2    = np.array(r2, dtype=np.float64)
    names = np.array(names, dtype=object)

    if len(r2) <= 1:
        return peaks, r2, names

    q75 = float(np.quantile(r2, 0.75))

    # R condition: keep pruning while (n > 25 OR count_above_Q75 > 25).
    while len(r2) > 0:
        n_above = int(np.sum(r2 > q75))
        if not (len(peaks) > 25 or n_above > 25):
            break
        if r2.min() >= q75:
            break
        drop = int(np.argmin(r2))
        peaks = np.delete(peaks, drop)
        r2    = np.delete(r2, drop)
        names = np.delete(names, drop)

    return peaks, r2, names


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateResult:
    """
    Output of :class:`CandidateSelector`.

    Attributes
    ----------
    gene_names : list[str]
        Accepted candidate gene symbols.
        R: ``namesIn``  → ``outSetCandRefG[[1]]``.

    expr_data : pd.DataFrame, shape (n_accepted, n_samples)
        Expression rows for accepted genes.
        R: ``datosParTissue``  → ``outSetCandRefG[[2]]``.

    fmm_fitted : pd.DataFrame
        FMM fitted values.
        R: ``adjParamTissue``  → ``outSetCandRefG[[3]]``.

    fmm_params : np.ndarray, shape (n_accepted, 5)
        FMM parameters [M, A, α, β, ω] per gene.
        R: ``paramParamTissue``  → ``outSetCandRefG[[4]]``.

    cos_fitted : pd.DataFrame
        Cosinor fitted values.
        R: ``adjCosTissue``  → ``outSetCandRefG[[5]]``.

    cos_params : np.ndarray, shape (n_accepted, 3)
        Cosinor parameters [M, A, φ] per gene.
        R: ``paramCosTissue``  → ``outSetCandRefG[[6]]``.

    fmm_peaks : np.ndarray, shape (n_accepted,)
        FMM peak times (via compUU) per gene.
        R: ``peaksIn``  → ``outSetCandRefG[[7]]``.

    sector_labels : np.ndarray of int, shape (n_accepted,)
        Sector assignment (1–8) per gene.
        R: ``regIn``  → ``outSetCandRefG[[8]]``.

    r2_par : np.ndarray, shape (n_accepted,)
        Parametric R² per gene.
        R: ``r2ParIn``  → ``outSetCandRefG[[9]]``.

    ranked_r2_np : np.ndarray
        All genes' NP R² sorted descending (distribution reference).
        R: ``mDistR2Tissue``  → ``outSetCandRefG[[11]]``.

    sector_centres : np.ndarray, shape (8,)
        The 8 sector centre-points m1…m8.
        R: ``outSetCandRefG[[20]]`` … ``outSetCandRefG[[27]]``.

    discarded_genes : list[str]
        Genes examined but rejected (failed ω or R²_par check).
        R: ``nameDiscard``  → ``outSetCandRefG[[28]]``.
    """

    gene_names:      list        = field(default_factory=list)
    expr_data:       pd.DataFrame = field(default_factory=pd.DataFrame)
    fmm_fitted:      pd.DataFrame = field(default_factory=pd.DataFrame)
    fmm_params:      np.ndarray  = field(default_factory=lambda: np.zeros((0, 5)))
    cos_fitted:      pd.DataFrame = field(default_factory=pd.DataFrame)
    cos_params:      np.ndarray  = field(default_factory=lambda: np.zeros((0, 3)))
    fmm_peaks:       np.ndarray  = field(default_factory=lambda: np.array([]))
    sector_labels:   np.ndarray  = field(default_factory=lambda: np.array([], dtype=int))
    r2_par:          np.ndarray  = field(default_factory=lambda: np.array([]))
    ranked_r2_np:    np.ndarray  = field(default_factory=lambda: np.array([]))
    sector_centres:  np.ndarray  = field(default_factory=lambda: np.zeros(8))
    discarded_genes: list        = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.gene_names)
        n_sectors_used = len(np.unique(self.sector_labels)) if n > 0 else 0
        lines = [
            "=== Candidate Selection Summary ===",
            f"  Genes accepted      : {n}",
            f"  Sectors populated   : {n_sectors_used} / 8",
            f"  Genes discarded     : {len(self.discarded_genes)}",
        ]
        if n > 0:
            for s in range(1, 9):
                cnt = int(np.sum(self.sector_labels == s))
                if cnt > 0:
                    lines.append(f"    Sector {s}: {cnt} gene(s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CandidateSelector
# ═══════════════════════════════════════════════════════════════════════════

class CandidateSelector:
    """
    Select candidate rhythmic genes via 8-sector FMM-validated filtering.

    R equivalent: ``minSetCand3_v2_cores()`` (lines 1242–1692).

    Parameters
    ----------
    tam : int
        Target number of genes per sector before the "continue beyond
        target" rule kicks in.  R default: 50.

    omega_min : float
        Minimum FMM ω to accept a gene (rejects spiky fits).
        R default: 0.1.

    r2_par_floor : float
        Absolute floor for parametric R²; the effective threshold is
        ``min(Q1_core_R2, r2_par_floor)``.  R default: 0.4.

    fmm_length_alpha_grid : int
        Grid size for FMM α.  R default: 48.

    fmm_length_omega_grid : int
        Grid size for FMM ω.  R default: 24.

    fmm_num_reps : int
        FMM refinement iterations.  R default: 3.

    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        tam:                   int   = 50,
        omega_min:             float = 0.1,
        r2_par_floor:          float = 0.4,
        fmm_length_alpha_grid: int   = 48,
        fmm_length_omega_grid: int   = 24,
        fmm_num_reps:          int   = 3,
        verbose:               bool  = True,
    ) -> None:
        self.tam                   = tam
        self.omega_min             = omega_min
        self.r2_par_floor          = r2_par_floor
        self.fmm_length_alpha_grid = fmm_length_alpha_grid
        self.fmm_length_omega_grid = fmm_length_omega_grid
        self.fmm_num_reps          = fmm_num_reps
        self.verbose               = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        r2_np:            np.ndarray,
        expr_norm:        pd.DataFrame,
        circular_scale:   np.ndarray,
        r2_par_core:      np.ndarray,
    ) -> CandidateResult:
        """
        Select candidate genes.

        Parameters
        ----------
        r2_np : (n_genes,) array
            Non-parametric R² for every gene in the expression matrix.
            R: ``obtainNPRefG[[5]]``.

        expr_norm : pd.DataFrame, shape (n_genes, n_samples)
            Full normalised expression matrix ordered by the preliminary
            circular time.
            R: ``preOrdRefG2[[3]]``  (= ``mFullTissueNorm``).

        circular_scale : (n_samples,) array
            Circular time axis from the preliminary ordering.
            R: ``preOrdRefG2[[2]]``  (= ``escParTissue``).

        r2_par_core : (n_core,) array
            Parametric R² of the core clock genes (from the preliminary
            ordering FMM fits).  Used to set the acceptance threshold.
            R: ``basicOrdRefG2[[5]]``  (= ``R2ParcoreG``).

        Returns
        -------
        CandidateResult
        """
        self._log("=== Stage 3.2: Candidate Gene Selection ===")

        all_genes   = expr_norm.index.values
        n_genes     = len(all_genes)
        r2_np       = np.asarray(r2_np, dtype=np.float64)
        circ        = np.asarray(circular_scale, dtype=np.float64)

        # ── 1. Ranked NP R² distribution ──────────────────────────────────
        rank_order    = np.argsort(r2_np)[::-1]
        ranked_r2_np  = r2_np[rank_order]
        q3_all        = float(np.quantile(ranked_r2_np, 0.75))

        # ── 2. Filter: NP R² > median ────────────────────────────────────
        r2_median = float(np.median(r2_np))
        self._log(f"  Median NP R² = {r2_median:.3f}")

        mask_above_median = r2_np > r2_median
        genes_above       = all_genes[mask_above_median]
        r2_above          = r2_np[mask_above_median]

        self._log(f"  Genes above median: {len(genes_above)}")

        # ── 3. Cosinor acrophase for every surviving gene ─────────────────
        cos_model  = CosinorModel()
        cos_peaks  = np.empty(len(genes_above), dtype=np.float64)
        for i, gene in enumerate(genes_above):
            row = expr_norm.loc[gene].values.astype(np.float64)
            cr  = cos_model.fit(row, circ)
            cos_peaks[i] = (-cr.params["phi"]) % (2.0 * pi)

        # ── 4. Sector assignment ──────────────────────────────────────────
        peak_ref = circular_median(cos_peaks)
        centres  = sector_boundaries(peak_ref)
        sectors  = assign_sectors(cos_peaks, peak_ref)

        self._log(f"  Circular median of Cosinor peaks: {peak_ref:.3f} rad")
        for s in range(1, 9):
            cnt = int(np.sum(sectors == s))
            self._log(f"    Sector {s}: {cnt} gene(s)")

        # ── 5. Per-sector FMM filtering ───────────────────────────────────
        r2_par_threshold = min(
            float(np.quantile(r2_par_core, 0.25)), self.r2_par_floor
        )
        self._log(f"  R²_par acceptance threshold: {r2_par_threshold:.3f}")

        fmm_model = FMMModel(
            length_alpha_grid=self.fmm_length_alpha_grid,
            length_omega_grid=self.fmm_length_omega_grid,
            num_reps=self.fmm_num_reps,
        )

        # Accumulators (matching R's rbind/c patterns).
        acc_names:      list[str]         = []
        acc_expr:       list[np.ndarray]  = []
        acc_fmm_fit:    list[np.ndarray]  = []
        acc_fmm_par:    list[np.ndarray]  = []  # [M, A, α, β, ω]
        acc_cos_fit:    list[np.ndarray]  = []
        acc_cos_par:    list[np.ndarray]  = []  # [M, A, φ]
        acc_peaks:      list[float]       = []
        acc_sectors:    list[int]         = []
        acc_r2par:      list[float]       = []
        acc_discarded:  list[str]         = []

        total_accepted = 0

        for sector_id in range(1, 9):
            sec_mask    = sectors == sector_id
            sec_genes   = genes_above[sec_mask]
            sec_r2_np   = r2_above[sec_mask]

            if len(sec_genes) == 0:
                continue

            # Sort by NP R² descending.
            order       = np.argsort(sec_r2_np)[::-1]
            sec_genes   = sec_genes[order]
            sec_r2_np   = sec_r2_np[order]

            q1_sector   = float(np.quantile(sec_r2_np, 0.25))
            r2_floor_s  = max(q1_sector, self.r2_par_floor)

            in_sector   = 0
            rec         = 0

            # R while-loop condition (lines 1345, 1387, etc.):
            # while (sector_has_genes &
            #        (in_sector < tam | (in_sector >= tam & R2_NP < Q3_all)) &
            #        rec <= len(genes) &
            #        R2_NP[rec] > max(Q1_sector, 0.4))
            while rec < len(sec_genes) and sec_r2_np[rec] > r2_floor_s:
                # "Continue beyond target" gate.
                if in_sector >= self.tam and sec_r2_np[rec] >= q3_all:
                    break

                gene = sec_genes[rec]
                rec += 1

                # Fit FMM.
                row = expr_norm.loc[gene].values.astype(np.float64)
                fr  = fmm_model.fit(row, circ)

                omega  = fr.params["omega"]
                r2_par = fr.r2

                self._log(
                    f"    [{total_accepted+1}] {gene}  "
                    f"R2_NP={sec_r2_np[rec-1]:.3f}  "
                    f"R2_par={r2_par:.3f}  ω={omega:.3f}",
                    end="\r",
                )

                if omega > self.omega_min and r2_par > r2_par_threshold:
                    in_sector      += 1
                    total_accepted += 1

                    # Cosinor fit for this gene.
                    cr = cos_model.fit(row, circ)

                    peak_fmm = fmm_peak_time(
                        fr.params["alpha"], fr.params["beta"], fr.params["omega"]
                    )

                    acc_names.append(gene)
                    acc_expr.append(row)
                    acc_fmm_fit.append(fr.fitted)
                    acc_fmm_par.append([
                        fr.params["M"], fr.params["A"],
                        fr.params["alpha"], fr.params["beta"], fr.params["omega"],
                    ])
                    acc_cos_fit.append(cr.fitted)
                    acc_cos_par.append([
                        cr.params["M"], cr.params["A"], cr.params["phi"],
                    ])
                    acc_peaks.append(peak_fmm)
                    acc_sectors.append(sector_id)
                    acc_r2par.append(r2_par)
                else:
                    acc_discarded.append(gene)

            self._log(
                f"    Sector {sector_id}: accepted {in_sector}, "
                f"examined {rec}           "
            )

        self._log(f"  Total accepted: {total_accepted}")

        # ── 6. Build result ───────────────────────────────────────────────
        n_acc = len(acc_names)
        cols  = expr_norm.columns

        result = CandidateResult(
            gene_names      = acc_names,
            expr_data       = pd.DataFrame(
                np.vstack(acc_expr) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            fmm_fitted      = pd.DataFrame(
                np.vstack(acc_fmm_fit) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            fmm_params      = np.array(acc_fmm_par).reshape(n_acc, 5) if n_acc else np.zeros((0, 5)),
            cos_fitted      = pd.DataFrame(
                np.vstack(acc_cos_fit) if n_acc else np.empty((0, len(circ))),
                index=acc_names, columns=cols,
            ),
            cos_params      = np.array(acc_cos_par).reshape(n_acc, 3) if n_acc else np.zeros((0, 3)),
            fmm_peaks       = np.array(acc_peaks),
            sector_labels   = np.array(acc_sectors, dtype=int),
            r2_par          = np.array(acc_r2par),
            ranked_r2_np    = ranked_r2_np,
            sector_centres  = centres,
            discarded_genes = acc_discarded,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _log(self, message: str, end: str = "\n") -> None:
        if self.verbose:
            print(message, end=end, flush=True)
