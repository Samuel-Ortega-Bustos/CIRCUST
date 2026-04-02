"""
circust/nonparametric.py
========================
Non-parametric rhythmicity scoring via circular unimodal isotonic regression.

Stage 3.1 of CIRCUST: for each gene, fit an *up-down* (one peak, one trough)
isotonic regression to the expression profile ordered by the preliminary
circular time.  Compare against a flat (mean) model to obtain R².

The method uses the Pool-Adjacent-Violators Algorithm (PAVA) to enforce
monotonicity constraints on circular segments, searching over all candidate
trough–peak pairs to find the best-fitting unimodal shape.

R equivalent
------------
``computeNP()``  (lines 13–30 of ``functionGTEX_cores.R``) which calls
``function1Local_modif()`` → ``busquedaMejor()`` → ``pavaC()`` from
``NucleoComun.R`` and ``upDownUp_NP_Code_*.R``.

Pipeline position
-----------------
    PreliminaryOrderEstimator  →  **NonParametricScorer**  →  CandidateSelector
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# Pool-Adjacent-Violators Algorithm  (R: pavaC / Iso::pava)
# ═══════════════════════════════════════════════════════════════════════════

def pava_increasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Increasing isotonic regression via the Pool-Adjacent-Violators Algorithm.

    Given observations *y* with optional positive weights *w*, return the
    vector ŷ that minimises  Σ wᵢ (yᵢ − ŷᵢ)²  subject to  ŷ₁ ≤ ŷ₂ ≤ … ≤ ŷₙ.

    This is a direct port of the C function ``pavaCAdri`` called by the R
    wrapper ``pavaC()`` in ``NucleoComun.R`` (lines 3–25).

    Complexity: O(n) time, O(n) space.

    Parameters
    ----------
    y : (n,) array
        Observed values.
    w : (n,) array or None
        Positive weights.  If *None*, uniform weights are used.

    Returns
    -------
    (n,) array — monotonically non-decreasing fitted values.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n <= 1:
        return y.copy()

    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)

    # Stack-based block-merge algorithm.
    # Each block stores (cumulative weighted sum, cumulative weight, right index).
    block_sum = np.empty(n, dtype=np.float64)
    block_w   = np.empty(n, dtype=np.float64)
    block_end = np.empty(n, dtype=np.intp)
    nb = 0                                          # number of active blocks

    for i in range(n):
        block_sum[nb] = y[i] * w[i]
        block_w[nb]   = w[i]
        block_end[nb] = i
        nb += 1

        # Merge while the monotonicity constraint is violated.
        while nb >= 2:
            avg_prev = block_sum[nb - 2] / block_w[nb - 2]
            avg_curr = block_sum[nb - 1] / block_w[nb - 1]
            if avg_prev <= avg_curr:
                break
            block_sum[nb - 2] += block_sum[nb - 1]
            block_w[nb - 2]   += block_w[nb - 1]
            block_end[nb - 2]  = block_end[nb - 1]
            nb -= 1

    # Write the block averages back into the result array.
    result = np.empty(n, dtype=np.float64)
    start = 0
    for b in range(nb):
        end = block_end[b] + 1
        result[start:end] = block_sum[b] / block_w[b]
        start = end

    return result


def pava_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Decreasing isotonic regression.

    Equivalent to negating *y*, running ``pava_increasing``, and negating
    back.  R equivalent: ``pavaCAdriDecreasing``.
    """
    return -pava_increasing(-np.asarray(y, dtype=np.float64), w)


# ═══════════════════════════════════════════════════════════════════════════
# Circular local-extrema detection  (R: extremosLocales in NucleoComun.R)
# ═══════════════════════════════════════════════════════════════════════════

def find_local_extrema(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local minima (troughs) and local maxima (peaks) using circular
    neighbours.

    A point *i* is a local maximum if  v[i−1] ≤ v[i]  AND  v[i] ≥ v[i+1],
    with indices wrapping circularly.  Local minima are defined symmetrically.

    R equivalent: ``extremosLocales(v)`` in ``NucleoComun.R`` (lines 36–60).

    Note on the R source: the code comments label the maxima as "mínimos
    locales" and vice-versa (a copy-paste error in the Spanish comments),
    but the variable names ``candU`` (up = max) and ``candL`` (low = min)
    are correct.

    Parameters
    ----------
    v : (n,) array

    Returns
    -------
    candL : 1-D int array — indices of local minima (troughs).
    candU : 1-D int array — indices of local maxima (peaks).
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev  = np.roll(v, 1)
    next_ = np.roll(v, -1)

    candU = np.where((prev <= v) & (v >= next_))[0]    # local maxima
    candL = np.where((prev >= v) & (v <= next_))[0]    # local minima

    return candL, candU


# ═══════════════════════════════════════════════════════════════════════════
# Circular unimodal isotonic fit  (R: busquedaMejor in NucleoComun.R)
# ═══════════════════════════════════════════════════════════════════════════

def circular_unimodal_fit(
    v: np.ndarray,
) -> tuple[np.ndarray, float, int, int] | None:
    """
    Fit a circular unimodal (up-then-down) isotonic regression.

    Searches over all candidate (trough, peak) pairs — local minima and
    maxima of *v* — to find the pair (L, U) that yields the lowest MSE
    when fitting an increasing PAVA from L→U and a decreasing PAVA from
    U→L (circularly).

    R equivalent
    ------------
    ``busquedaMejor(v, candL, candU)`` in ``NucleoComun.R`` (lines 83–478).

    Parameters
    ----------
    v : (n,) array
        Gene expression values ordered by circular time.

    Returns
    -------
    fitted : (n,) array — fitted values.
    mse    : float      — mean squared error.
    L_opt  : int        — index of optimal trough (0-based).
    U_opt  : int        — index of optimal peak (0-based).
    None if no valid fit is found.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)

    if n == 0:
        return None

    candL, candU = find_local_extrema(v)

    if len(candL) == 0 or len(candU) == 0:
        # No extrema — constant fit is the best we can do.
        fitted = np.full(n, v.mean())
        mse = float(np.mean((v - fitted) ** 2))
        return fitted, mse, 0, 0

    # Doubled vector for circular-wrap indexing.
    v2 = np.concatenate([v, v])

    best_mse    = np.inf
    best_fitted = None
    best_L      = 0
    best_U      = 0

    for indL in candL:
        # Order U candidates: first those ≥ indL, then those < indL.
        # This mirrors R's  c(candU[candU>=indL], candU[candU<indL]).
        mask      = candU >= indL
        ordered_U = np.concatenate([candU[mask], candU[~mask]])

        # ── Forward pass: identify valid U's with their increasing PAVAs ──
        valid_Us:   list[int]        = []
        valid_incs: list[np.ndarray] = []

        for indU in ordered_U:
            inc_result = _increasing_segment(v, v2, indL, indU, n)

            if inc_result is None:
                # "break" signal: PAVA doesn't rise from trough.
                break

            full_inc, is_valid_U = inc_result
            if is_valid_U:
                valid_Us.append(indU)
                valid_incs.append(full_inc)

        # ── Backward pass: compute decreasing PAVAs for valid U's ─────────
        for idx_rev, indU in enumerate(reversed(valid_Us)):
            j = len(valid_Us) - 1 - idx_rev
            full_inc = valid_incs[j]

            if indL == indU:
                # Constant fit.
                fitted = full_inc
                mse = float(np.mean((v - fitted) ** 2))
                if mse < best_mse:
                    best_mse, best_fitted = mse, fitted
                    best_L, best_U = indL, indU
                continue

            dec_result = _decreasing_segment(v, v2, indL, indU, n)

            if dec_result is None:
                # "break" signal: decreasing fit can't reach trough.
                break

            dec_fit, is_valid = dec_result
            if not is_valid:
                continue

            # Assemble full circular fit.
            fitted = _assemble(v, indL, indU, full_inc, dec_fit, n)
            mse = float(np.mean((v - fitted) ** 2))

            if mse < best_mse:
                best_mse, best_fitted = mse, fitted
                best_L, best_U = indL, indU

        if best_mse == 0.0:
            break

    if best_fitted is None:
        return None

    return best_fitted, best_mse, best_L, best_U


# ── Helpers for circular_unimodal_fit ─────────────────────────────────────

def _increasing_segment(
    v: np.ndarray, v2: np.ndarray,
    indL: int, indU: int, n: int,
) -> tuple[np.ndarray, bool] | None:
    """
    Compute the increasing PAVA from *indL* to *indU* (circular),
    with ``v[indL]`` and ``v[indU]`` as fixed boundary values.

    Returns
    -------
    (full_inc, is_valid_U) — fitted values from L to U and validity flag.
    None — "break" signal: no more U's can be valid for this L.
    """
    if indL == indU:
        return np.full(n, v.mean()), True

    # Interior points between L and U (exclusive of both endpoints).
    if indL < indU:
        k_interior = indU - indL - 1
        if k_interior > 0:
            interior_vals = v[indL + 1 : indU]
        else:
            interior_vals = np.array([], dtype=np.float64)
    else:
        # Wrap around: L+1, L+2, …, n-1, 0, 1, …, U-1
        k_interior = (n - indL - 1) + indU
        if k_interior > 0:
            interior_vals = v2[indL + 1 : indL + 1 + k_interior]
        else:
            interior_vals = np.array([], dtype=np.float64)

    # Build full increasing fit: [v[L], pava(interior), v[U]]
    if len(interior_vals) > 0:
        inc_fit  = pava_increasing(interior_vals)
        full_inc = np.empty(len(interior_vals) + 2, dtype=np.float64)
        full_inc[0]    = v[indL]
        full_inc[1:-1] = inc_fit
        full_inc[-1]   = v[indU]
    else:
        full_inc = np.array([v[indL], v[indU]], dtype=np.float64)

    # Validity checks (matching R's busquedaMejor).
    if len(full_inc) > 2:
        # First interior point must exceed trough.
        if full_inc[1] <= v[indL]:
            return None   # "break" — no further U's valid for this L.
        # Last interior point must be below peak.
        is_valid = full_inc[-2] < v[indU]
    else:
        # Only 2 points (L and U adjacent): always valid.
        is_valid = True

    return full_inc, is_valid


def _decreasing_segment(
    v: np.ndarray, v2: np.ndarray,
    indL: int, indU: int, n: int,
) -> tuple[np.ndarray, bool] | None:
    """
    Compute the decreasing PAVA on the interior arc from *indU* to *indL*
    (the complement of the increasing arc).

    Unlike the increasing segment, the endpoints ``v[indU]`` and ``v[indL]``
    are **not** included in the PAVA input — only the strict interior is
    fitted.

    Returns
    -------
    (dec_fit, is_valid) — fitted interior values and validity flag.
    None — "break" signal: no more U's can be valid.
    """
    # Interior points: indU+1, …, indL-1  (circularly).
    if indU < indL:
        k_interior = indL - indU - 1
        if k_interior > 0:
            interior_vals = v[indU + 1 : indL]
        else:
            interior_vals = np.array([], dtype=np.float64)
    else:
        # Wrap: indU+1, …, n-1, 0, …, indL-1
        k_interior = (n - indU - 1) + indL
        if k_interior > 0:
            interior_vals = v2[indU + 1 : indU + 1 + k_interior]
        else:
            interior_vals = np.array([], dtype=np.float64)

    if len(interior_vals) == 0:
        return np.array([], dtype=np.float64), True

    dec_fit = pava_decreasing(interior_vals)

    # Validity: last fitted value must be ≥ trough (can connect smoothly).
    if dec_fit[-1] < v[indL]:
        return None  # "break" signal.

    # First fitted value must be ≤ peak (starts below the maximum).
    is_valid = dec_fit[0] <= v[indU]

    return dec_fit, is_valid


def _assemble(
    v: np.ndarray,
    indL: int, indU: int,
    full_inc: np.ndarray,
    dec_fit: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Place the increasing and decreasing fitted values back into a
    length-*n* array aligned with the original positions of *v*.

    ``full_inc`` covers positions L, L+1, …, U (including endpoints).
    ``dec_fit`` covers positions U+1, U+2, …, L-1 (interior only).
    """
    fitted = np.empty(n, dtype=np.float64)

    # ── Increasing arc: L → U ──
    if indL <= indU:
        inc_pos = np.arange(indL, indU + 1)
    else:
        inc_pos = np.concatenate([np.arange(indL, n), np.arange(0, indU + 1)])

    fitted[inc_pos] = full_inc

    # ── Decreasing arc: U+1 → L-1 ──
    if len(dec_fit) > 0:
        if indU < indL:
            dec_pos = np.arange(indU + 1, indL)
        else:
            dec_pos = np.concatenate([
                np.arange(indU + 1, n), np.arange(0, indL)
            ])
        fitted[dec_pos] = dec_fit

    return fitted


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NonParametricResult:
    """
    Output of :class:`NonParametricScorer`.

    Attributes
    ----------
    fitted : pd.DataFrame, shape (n_genes, n_samples)
        Unimodal isotonic regression fitted values per gene.
        R equivalent: ``computeNP()[[1]]``  (= ``fitNP``).

    mse_np : np.ndarray, shape (n_genes,)
        MSE of the non-parametric fit for each gene.
        R equivalent: ``computeNP()[[3]]``  (= ``msseNP``).

    mse_flat : np.ndarray, shape (n_genes,)
        MSE of the flat (mean) model for each gene.
        R equivalent: ``computeNP()[[4]]``  (= ``msseFlat``).

    r2 : np.ndarray, shape (n_genes,)
        Non-parametric R² = 1 − MSE_NP / MSE_flat.
        R equivalent: ``computeNP()[[5]]``  (= ``R2``).

    gene_names : list[str]
        Gene symbols in the same row order as the arrays above.
    """

    fitted:     pd.DataFrame
    mse_np:     np.ndarray
    mse_flat:   np.ndarray
    r2:         np.ndarray
    gene_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.r2)
        above50 = int(np.sum(self.r2 > 0.5))
        above70 = int(np.sum(self.r2 > 0.7))
        lines = [
            "=== Non-Parametric Rhythmicity Summary ===",
            f"  Genes scored        : {n}",
            f"  Median R2_NP        : {float(np.median(self.r2)):.3f}",
            f"  R2_NP > 0.5         : {above50} ({100*above50/max(n,1):.1f}%)",
            f"  R2_NP > 0.7         : {above70} ({100*above70/max(n,1):.1f}%)",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# NonParametricScorer
# ═══════════════════════════════════════════════════════════════════════════

class NonParametricScorer:
    """
    Score every gene in an expression matrix for non-parametric rhythmicity
    using circular unimodal isotonic regression.

    For each gene the algorithm fits the best up-down (one peak, one trough)
    shape using PAVA over all candidate extrema pairs.  The R² against a
    flat (mean) model quantifies how well the ordered expression profile
    follows a unimodal oscillation — without assuming any parametric
    waveform.

    R equivalent: ``computeNP(datos)`` (lines 13–30).

    Parameters
    ----------
    verbose : bool
        Print progress messages.

    Examples
    --------
    >>> from circust.nonparametric import NonParametricScorer
    >>> result = NonParametricScorer().run(expr_matrix)
    >>> print(result.summary())
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(self, expr_ordered: pd.DataFrame) -> NonParametricResult:
        """
        Score all genes.

        Parameters
        ----------
        expr_ordered : pd.DataFrame, shape (n_genes, n_samples)
            Full normalised expression matrix already ordered by the
            preliminary circular time.
            R equivalent: ``preOrdRefG2[[3]]``  (= ``matNewNew``).

        Returns
        -------
        NonParametricResult
        """
        genes  = expr_ordered.index.tolist()
        n_genes, n_samples = expr_ordered.shape
        values = expr_ordered.values.astype(np.float64)

        self._log("=== Stage 3.1: Non-Parametric Rhythmicity Scoring ===")

        fitted_mat = np.zeros((n_genes, n_samples), dtype=np.float64)
        mse_np     = np.zeros(n_genes, dtype=np.float64)
        mse_flat   = np.zeros(n_genes, dtype=np.float64)
        r2         = np.zeros(n_genes, dtype=np.float64)

        for i in range(n_genes):
            if self.verbose and (i + 1) % 500 == 0:
                self._log(f"  [{i+1}/{n_genes}]")

            row = values[i]

            # Flat model: predict the mean.
            row_mean = row.mean()
            mse_flat[i] = float(np.mean((row - row_mean) ** 2))

            # Unimodal isotonic fit.
            result = circular_unimodal_fit(row)
            if result is not None:
                fitted_mat[i] = result[0]
                mse_np[i]     = result[1]
            else:
                # Fallback: constant fit.
                fitted_mat[i] = row_mean
                mse_np[i]     = mse_flat[i]

            # R² = 1 − MSE_NP / MSE_flat.
            if mse_flat[i] > 0:
                r2[i] = 1.0 - mse_np[i] / mse_flat[i]
            else:
                r2[i] = 0.0

        self._log(f"  Done — {n_genes} genes scored.")

        result = NonParametricResult(
            fitted     = pd.DataFrame(fitted_mat, index=genes,
                                      columns=expr_ordered.columns),
            mse_np     = mse_np,
            mse_flat   = mse_flat,
            r2         = r2,
            gene_names = genes,
        )

        self._log(result.summary())
        return result

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
