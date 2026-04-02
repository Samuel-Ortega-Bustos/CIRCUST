"""
circust/preliminary_order.py
============================
Stage 2: Preliminary Circular Ordering.

Implements R's ``basicPreOder_cores`` (Step 2.1) and ``basicOder_cores``
(Steps 2.2 & 2.3) from ``functionGTEX_cores.R``.

Goal
----
The CPCA-derived circular ordering is rotation-invariant and has an
arbitrary direction (clockwise or counter-clockwise).  This stage fixes
both issues using biological priors about the mammalian circadian clock:

  * **ARNTL** peaks near Zeitgeber time 0 (subjective midnight/dawn).
    After the rotation it sits at π on the circular scale.

  * **DBP** peaks in the active phase, which should fall in [0, π) when
    ARNTL is placed at π.  If DBP ends up in [π, 2π) the direction of
    the ordering is wrong and is reversed.

Step 2.1 — ``basicPreOder_cores``
-----------------------------------
1. Rotate the circular scale so that ARNTL's FMM peak time sits at π.
2. Determine direction: if (DBP_peak − ARNTL_peak + π) mod 2π < π then
   DBP is in the first half → forward direction; otherwise reverse.
3. Re-parameterise each core gene's FMM fit in the new coordinate frame.
4. Classify the remaining core genes as *day* (peak ∈ [0, π)) or
   *night* (peak ∈ [π, 2π)).

Step 2.2 & 2.3 — ``basicOder_cores``
--------------------------------------
Apply a refined biological consistency check.  If any of the following
conditions hold (``aviso = True``):
  - DBP is too close to 0 (1 − cos(DBP_peak) ≤ 0.1)
  - DBP is too close to π (1 − cos(DBP_peak − π) ≤ 0.1)
  - Fewer than half of the non-ARNTL/DBP genes peak in [0, π)
AND the order DBP < CRY1 < ARNTL does NOT hold, then flip the direction.

R source lines
--------------
* Step 2.1: ``basicPreOder_cores``  (lines 4192-4263)
* Step 2.2: ``basicOder_cores``     (lines 4527-4594)

Pipeline position
-----------------
    OutlierRefiner  →  PreliminaryOrderEstimator  →  (Stage 3)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from math import pi

from circust.outlier import OutlierRefinementResult
from circust.fitting.fmm import fmm_peak_time
from circust.fitting.rhythm_model import FitResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreliminaryOrderResult:
    """
    Combined output of Stage 2.1 and Stage 2.2 & 2.3.

    Attributes
    ----------
    sample_order : np.ndarray of int, shape (n_samples,)
        Final sample indices into the cleaned expression matrix.
        R equivalent: ``basicOrdRefG2[[1]]``  (= ``indSin``).

    circular_scale : np.ndarray of float, shape (n_samples,)
        Circular time axis in [0, 2π) after direction correction.
        R equivalent: ``basicOrdRefG2[[2]]``  (= ``escSincroRefG``).

    expr_ordered : pd.DataFrame, shape (n_genes, n_samples)
        Full normalised expression matrix in the final circular order.
        R equivalent: ``basicOrdRefG2[[3]]``  (= ``matNewNew``).

    peak_times : np.ndarray of float, shape (n_core_genes,)
        FMM peak time for each core gene in the final coordinate frame.
        R equivalent: ``basicOrdRefG2[[4]]``  (= ``peaksPreNew``).

    r2_fmm : np.ndarray of float, shape (n_core_genes,)
        R² of the FMM fit for each core gene (from final-ordering fits).
        R equivalent: ``basicOrdRefG2[[5]]``.

    fmm_params : np.ndarray of float, shape (n_core_genes, 5)
        FMM parameters [M, A, α, β, ω] per core gene in the new frame.
        R equivalent: ``basicOrdRefG2[[6]]``  (= ``pars``).

    direction_flipped : bool
        True if the direction was reversed by either Step 2.1 or 2.2.
        R equivalent: ``basicOrdRefG2[[7]]``  (= ``cambiaOri``).

    within_group_indices : np.ndarray of int, shape (n_samples,)
        0-based positional indices within the group after potential flip.
        R equivalent: ``basicOrdRefG2[[8]]``  (= ``indNewNew``), but
        converted to 0-based (R is 1-based).

    core_genes : list[str]
        Core gene symbols in the order used.

    day_genes : list[str]
        Core genes (exc. ARNTL, DBP) with peak in [0, π).

    night_genes : list[str]
        Core genes (exc. ARNTL, DBP) with peak in [π, 2π).

    pre_sample_order : np.ndarray of int
        Sample order from Step 2.1 (before the Step 2.2 check).
        R equivalent: ``preOrdRefG2[[1]]``.

    pre_circular_scale : np.ndarray of float
        Circular scale from Step 2.1.
        R equivalent: ``preOrdRefG2[[2]]``.

    pre_expr_ordered : pd.DataFrame
        Expression matrix ordered by Step 2.1.
        R equivalent: ``preOrdRefG2[[3]]``.

    pre_peak_times : np.ndarray of float
        FMM peak times from Step 2.1.
        R equivalent: ``preOrdRefG2[[4]]``.

    pre_fmm_params : np.ndarray of float, shape (n_core_genes, 5)
        FMM parameters from Step 2.1.
        R equivalent: ``preOrdRefG2[[6]]``  (= ``parCore``).

    pre_direction_reversed : bool
        True if Step 2.1 reversed the direction (DBP in wrong half).
        R equivalent: ``preOrdRefG2[[13]]``  (= ``reverse``).
    """

    sample_order:           np.ndarray
    circular_scale:         np.ndarray
    expr_ordered:           pd.DataFrame
    peak_times:             np.ndarray
    r2_fmm:                 np.ndarray
    fmm_params:             np.ndarray
    direction_flipped:      bool
    within_group_indices:   np.ndarray
    core_genes:             list

    day_genes:              list       = field(default_factory=list)
    night_genes:            list       = field(default_factory=list)

    pre_sample_order:       np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    pre_circular_scale:     np.ndarray = field(default_factory=lambda: np.array([]))
    pre_expr_ordered:       pd.DataFrame = field(default_factory=pd.DataFrame)
    pre_peak_times:         np.ndarray = field(default_factory=lambda: np.array([]))
    pre_fmm_params:         np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    pre_direction_reversed: bool       = False

    def summary(self) -> str:
        lines = [
            "=== Preliminary Order Summary ===",
            f"  Direction flipped    : {self.direction_flipped}",
            f"  Step-2.1 reversed   : {self.pre_direction_reversed}",
            f"  Day genes  (0..π)   : {self.day_genes}",
            f"  Night genes (π..2π) : {self.night_genes}",
            f"  Core gene peaks     : {np.round(self.peak_times, 3).tolist()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PreliminaryOrderEstimator
# ---------------------------------------------------------------------------

class PreliminaryOrderEstimator:
    """
    Implement Stage 2 of CIRCUST: set the biological reference frame.

    Runs two sub-steps mirroring the R pipeline:
      1. ``basicPreOder_cores`` — rotate so ARNTL sits at π, detect
         direction from DBP's position.
      2. ``basicOder_cores``   — refined consistency check; flip if
         the biology looks inconsistent.

    Parameters
    ----------
    arntl_gene : str
        Anchor gene expected to peak near subjective midnight.
        Default: ``"ARNTL"``.

    dbp_gene : str
        Direction-determining gene expected to peak in the active phase
        (first half, [0, π)).  Default: ``"DBP"``.

    cry1_gene : str
        Gene used in the refined consistency check.  Default: ``"CRY1"``.

    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        arntl_gene: str  = "ARNTL",
        dbp_gene:   str  = "DBP",
        cry1_gene:  str  = "CRY1",
        verbose:    bool = True,
    ) -> None:
        self.arntl_gene = arntl_gene
        self.dbp_gene   = dbp_gene
        self.cry1_gene  = cry1_gene
        self.verbose    = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        refined:    OutlierRefinementResult,
        core_genes: list[str],
    ) -> PreliminaryOrderResult:
        """
        Run Stage 2 (Steps 2.1 and 2.2).

        Parameters
        ----------
        refined : OutlierRefinementResult
            Output of ``OutlierRefiner.run()``.  Provides
            ``fmm_fits_final``, ``fmm_peak_times_final``,
            ``expr_norm_final``, and ``cpca_final``.

        core_genes : list of str
            Ordered list of core clock gene symbols (same order used in
            ``CPCA`` and ``OutlierRefiner``).

        Returns
        -------
        PreliminaryOrderResult
        """
        self._log("=== Stage 2: Preliminary Ordering ===")

        # ── Step 2.1: basicPreOder_cores ─────────────────────────────────
        self._log("  Step 2.1 — rotating to ARNTL reference frame ...")
        (o_pre, esc_pre, mat_pre, peaks_pre, r2_fmm, par_pre,
         names_day, names_night, reversed_21) = self._pre_order(
            refined, core_genes
        )

        # ── Step 2.2: basicOder_cores ─────────────────────────────────────
        self._log("  Step 2.2 — checking biological consistency ...")
        (o_fin, esc_fin, mat_fin, peaks_fin,
         pars_fin, flipped_22, ind_new) = self._basic_order(
            o_pre, esc_pre, mat_pre, peaks_pre, par_pre, core_genes
        )

        # Overall direction flag (either step could have reversed)
        direction_flipped = reversed_21 ^ flipped_22

        result = PreliminaryOrderResult(
            sample_order           = o_fin,
            circular_scale         = esc_fin,
            expr_ordered           = mat_fin,
            peak_times             = peaks_fin,
            r2_fmm                 = r2_fmm,
            fmm_params             = pars_fin,
            direction_flipped      = direction_flipped,
            within_group_indices   = ind_new,
            core_genes             = core_genes,
            day_genes              = names_day,
            night_genes            = names_night,
            pre_sample_order       = o_pre,
            pre_circular_scale     = esc_pre,
            pre_expr_ordered       = mat_pre,
            pre_peak_times         = peaks_pre,
            pre_fmm_params         = par_pre,
            pre_direction_reversed = reversed_21,
        )

        self._log(result.summary())
        return result

    # ------------------------------------------------------------------
    # Step 2.1 — basicPreOder_cores
    # ------------------------------------------------------------------

    def _pre_order(
        self,
        refined:    OutlierRefinementResult,
        core_genes: list[str],
    ) -> tuple:
        """
        R equivalent: ``basicPreOder_cores`` (lines 4192-4263).

        Returns
        -------
        o_new       : np.ndarray  — sample ordering indices (columns of expr)
        esc_new     : np.ndarray  — circular scale in [0, 2π)
        mat_new     : pd.DataFrame — full matrix in new order
        peaks       : np.ndarray  — FMM peak times per core gene
        r2_fmm      : np.ndarray  — R² per core gene (from final fits)
        par_core    : np.ndarray  shape (n_core, 5) — [M, A, α, β, ω]
        names_day   : list[str]   — core genes with peak in [0, π)
        names_night : list[str]   — core genes with peak in [π, 2π)
        reversed    : bool        — True if direction was flipped here
        """
        circular_scale = refined.cpca_final.circular_scale   # escalaPhi8
        expr_ordered   = refined.expr_norm_final              # mFullTissueNorm
        fmm_fits       = refined.fmm_fits_final               # allParAfter
        peak_times_fin = refined.fmm_peak_times_final         # phisFMMAfter

        n_core = len(core_genes)

        # ── R² values from final-ordering FMM fits (m3r2[:,0]) ───────────
        r2_fmm = np.array([
            fmm_fits[g].r2 if g in fmm_fits else float("nan")
            for g in core_genes
        ])

        # ── ARNTL and DBP peak times in the original CPCA frame ──────────
        arntl_peak = peak_times_fin[self.arntl_gene]  # peakRefEpidermis2
        dbp_peak   = peak_times_fin[self.dbp_gene]    # peakDBPEpidermis2

        # ── Rotate circular scale: ARNTL → π ─────────────────────────────
        # R: escTEpidermis <- order((listaStep1[[2]] - peakRef + pi) %% (2π))
        rotated = (circular_scale - arntl_peak + pi) % (2.0 * pi)
        esc_T   = np.argsort(rotated)          # escTEpidermis
        esc2    = rotated[esc_T]               # esc2

        # ── Direction: forward or reverse ─────────────────────────────────
        # R: if((peakDBP - peakRef + π) %% (2π) < π) → forward
        dbp_rotated = (dbp_peak - arntl_peak + pi) % (2.0 * pi)
        forward     = dbp_rotated < pi          # True → keep direction

        # ── Re-parameterise each core gene's FMM in the new frame ─────────
        # R (per gene i):
        #   newAlpha = (alpha_i - peakRef + π) %% 2π
        #   if forward: peak_i = compUU(newAlpha, beta_i, omega_i)
        #               par_i  = [M, A, newAlpha, beta, omega]
        #   else:       peak_i = compUU(2π-newAlpha, 2π-beta, omega)
        #               par_i  = [M, A, 2π-newAlpha, 2π-beta, omega]
        peaks    = np.zeros(n_core)
        par_core = np.zeros((n_core, 5))

        for i, gene in enumerate(core_genes):
            if gene not in fmm_fits:
                continue
            fr = fmm_fits[gene]
            alpha = fr.params["alpha"]
            beta  = fr.params["beta"]
            omega = fr.params["omega"]
            M     = fr.params["M"]
            A     = fr.params["A"]

            new_alpha = (alpha - arntl_peak + pi) % (2.0 * pi)

            if forward:
                peaks[i]    = fmm_peak_time(new_alpha, beta, omega)
                par_core[i] = [M, A, new_alpha, beta, omega]
            else:
                na2 = (2.0 * pi - new_alpha) % (2.0 * pi)
                nb2 = (2.0 * pi - beta)      % (2.0 * pi)
                peaks[i]    = fmm_peak_time(na2, nb2, omega)
                par_core[i] = [M, A, na2, nb2, omega]

        # ── Reorder matrix columns ─────────────────────────────────────────
        if forward:
            o_new   = esc_T
            esc_new = esc2
            mat_new = expr_ordered.iloc[:, esc_T].copy()
        else:
            o_new   = esc_T[::-1].copy()
            esc_new = (2.0 * pi - esc2[::-1]) % (2.0 * pi)
            mat_new = expr_ordered.iloc[:, esc_T[::-1]].copy()

        mat_new.columns = range(mat_new.shape[1])

        # ── Classify as day / night (excluding ARNTL and DBP) ─────────────
        names_day   = []
        names_night = []
        for i, gene in enumerate(core_genes):
            if gene == self.arntl_gene or gene == self.dbp_gene:
                continue
            if 0 <= peaks[i] < pi:
                names_day.append(gene)
            else:
                names_night.append(gene)

        self._log(
            f"    ARNTL peak: {arntl_peak:.3f} rad  |  "
            f"DBP rotated: {dbp_rotated:.3f} rad  |  "
            f"Direction: {'forward' if forward else 'REVERSED'}"
        )

        return (o_new, esc_new, mat_new, peaks, r2_fmm, par_core,
                names_day, names_night, not forward)

    # ------------------------------------------------------------------
    # Step 2.2 — basicOder_cores
    # ------------------------------------------------------------------

    def _basic_order(
        self,
        o:          np.ndarray,
        esc:        np.ndarray,
        mat:        pd.DataFrame,
        peaks:      np.ndarray,
        par:        np.ndarray,
        core_genes: list[str],
    ) -> tuple:
        """
        R equivalent: ``basicOder_cores`` (lines 4527-4594).

        Checks biological consistency and flips if needed.

        Returns
        -------
        o_new      : np.ndarray
        esc_new    : np.ndarray
        mat_new    : pd.DataFrame
        peaks_new  : np.ndarray
        pars_new   : np.ndarray  shape (n_core, 5)
        flipped    : bool
        ind_new    : np.ndarray  (0-based positional indices)
        """
        arntl_i = core_genes.index(self.arntl_gene) if self.arntl_gene in core_genes else None
        dbp_i   = core_genes.index(self.dbp_gene)   if self.dbp_gene   in core_genes else None
        cry1_i  = core_genes.index(self.cry1_gene)  if self.cry1_gene  in core_genes else None

        if dbp_i is None:
            self._log(
                f"  WARNING: {self.dbp_gene} not found in core_genes; "
                "skipping direction check."
            )
            n = len(o)
            return o, esc, mat, peaks, par, False, np.arange(n)

        dbp_peak  = peaks[dbp_i]
        n_core    = len(core_genes)

        # Count genes in [0, π) excluding ARNTL and DBP
        peak_d = sum(
            1 for i, g in enumerate(core_genes)
            if g != self.arntl_gene and g != self.dbp_gene
            and 0 < peaks[i] < pi
        )
        mitad = int(np.floor((n_core - 2) / 2))

        # aviso conditions (R lines 4531-4537)
        p6am  = 1.0 - np.cos(dbp_peak)          # 1-cos(DBP) — distance from 0
        p6pm  = 1.0 - np.cos(dbp_peak - pi)     # 1-cos(DBP-π) — distance from π
        aviso = (p6am <= 0.1) or (p6pm <= 0.1) or (peak_d < mitad)

        # Exclusion: don't flip if DBP < CRY1 < ARNTL (all ascending)
        if aviso and cry1_i is not None and arntl_i is not None:
            excl  = (peaks[dbp_i] < peaks[cry1_i] < peaks[arntl_i])
            flip  = not excl
        elif aviso:
            flip  = True
        else:
            flip  = False

        n = len(o)

        if flip:
            self._log(
                f"    Flipping direction  "
                f"(aviso=True, DBP_peak={dbp_peak:.3f}, "
                f"peak_d={peak_d}/{mitad})"
            )
            peaks_new = (2.0 * pi - peaks) % (2.0 * pi)
            o_new     = o[::-1].copy()
            esc_new   = (2.0 * pi - esc[::-1]) % (2.0 * pi)
            mat_new   = mat.iloc[:, ::-1].copy()
            mat_new.columns = range(mat_new.shape[1])
            pars_new        = par.copy()
            pars_new[:, 2]  = (2.0 * pi - par[:, 2]) % (2.0 * pi)  # α
            pars_new[:, 3]  = (2.0 * pi - par[:, 3]) % (2.0 * pi)  # β
            ind_new   = np.arange(n - 1, -1, -1)
        else:
            self._log("    Direction is consistent — no flip needed.")
            peaks_new = peaks.copy()
            o_new     = o.copy()
            esc_new   = esc.copy()
            mat_new   = mat.copy()
            pars_new  = par.copy()
            ind_new   = np.arange(n)

        return o_new, esc_new, mat_new, peaks_new, pars_new, flip, ind_new

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
