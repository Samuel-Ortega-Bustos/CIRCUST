"""
circust/fitting/fmm.py
=======================
Flexible Multivariate Model (FMM) — single-component rhythm fitting.

Mathematical model
------------------
    y(t) = M + A · cos(β + 2·atan(ω·tan((t − α)/2)))

The inner term  Φ(t) = β + 2·atan(ω·tan((t − α)/2))  is a Möbius
transformation of the circle.  When ω = 1 it reduces to a simple
phase shift; as ω → 0 the waveform approaches a cosine (Cosinor).
The key advantage over Cosinor is that ω controls *waveform skewness*:
the model can represent asymmetric peaks found in real circadian data.

Parameters
----------
M     mesor       — mean level
A     amplitude   — always > 0
α     alpha       — peak-region location in [0, 2π)
β     beta        — waveform shape parameter in [0, 2π)
ω     omega       — skewness in (0, omegaMax]

Estimation algorithm
--------------------
This is an exact Python port of ``fitFMM_Par()`` from ``FMM.R``.

Step 1 — Grid search
    For each (α, ω) combination on a grid, fix those two parameters
    and fit M, A, β via Cosinor OLS on the Möbius-transformed time
    axis.  Select the candidate with the lowest RSS that also satisfies
    amplitude stability constraints.

Step 2 — Nelder-Mead refinement
    Polish the Step-1 estimate by minimising the RSS over all five
    parameters simultaneously, subject to the same stability constraints
    (infeasible points return ∞).

Steps 1+2 are repeated ``num_reps`` times, each time narrowing the
grid around the previous best estimate.

R equivalent
------------
``fitFMM_Par(vData, timePoints, lengthAlphaGrid=48, lengthOmegaGrid=24,
             omegaMax=1, numReps=3)``

Returns
-------
FitResult
    ``params`` keys: ``M``, ``A``, ``alpha``, ``beta``, ``omega``
"""
import numpy as np
from scipy.optimize import minimize

from src.circust.fitting.rhythm_model import FitResult, RhythmModel


# ---------------------------------------------------------------------------
# Internal helpers  (mirror R's hidden functions)
# ---------------------------------------------------------------------------

def _mobius(t: np.ndarray, alpha: float, omega: float) -> np.ndarray:
    """Möbius transformation of the time axis.

    Φ(t) = 2·atan(ω·tan((t − α)/2))
    R equivalent: ``parteMobius`` in ``step1FMM``.
    """
    return 2.0 * np.arctan(omega * np.tan((t - alpha) / 2.0))


def _fmm_predict(
    t: np.ndarray,
    M: float, A: float, alpha: float, beta: float, omega: float,
) -> np.ndarray:
    """Evaluate the FMM waveform at time points ``t``."""
    phi = beta + _mobius(t, alpha, omega)
    return M + A * np.cos(phi)


def _step1(
    alpha: float,
    omega: float,
    data: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Step 1: fix α and ω, estimate M, A, β via Cosinor OLS.

    Returns [M, A, alpha, beta, omega, RSS] — same layout as R's step1FMM.
    """
    mob      = _mobius(t, alpha, omega)
    t_star   = alpha + mob
    xx       = np.cos(t_star)
    zz       = np.sin(t_star)
    X        = np.column_stack([np.ones(len(data)), xx, zz])
    coeffs, *_ = np.linalg.lstsq(X, data, rcond=None)
    M_est    = coeffs[0]
    b        = coeffs[1]
    g        = coeffs[2]
    phi_est  = np.arctan2(-g, b)
    A_est    = np.sqrt(b**2 + g**2)
    beta_est = (phi_est + alpha) % (2 * np.pi)

    fitted   = M_est + A_est * np.cos(beta_est + mob)
    rss      = np.sum((data - fitted)**2) / len(t)
    return np.array([M_est, A_est, alpha, beta_est, omega, rss])


def _best_step1(
    data: np.ndarray,
    grid_results: np.ndarray,
) -> np.ndarray | None:
    """
    Select the grid candidate with the lowest RSS that satisfies
    amplitude stability constraints.

    R equivalent: ``bestStep1()``.
    Stability: M+A ≤ max(data) + 10% range  AND  M−A ≥ min(data) − 10% range.
    """
    order    = np.argsort(grid_results[:, 5])   # sort by RSS
    data_min = data.min()
    data_max = data.max()
    slack    = 0.1 * (data_max - data_min)

    for idx in order:
        row   = grid_results[idx]
        M, A  = row[0], row[1]
        if np.isnan(M) or np.isnan(A):
            continue
        if (M + A) <= (data_max + slack) and (M - A) >= (data_min - slack):
            return row
    return None


def _step2_objective(
    params: np.ndarray,
    data: np.ndarray,
    t: np.ndarray,
    omega_max: float,
) -> float:
    """
    Nelder-Mead objective: mean RSS with constraint penalties.

    R equivalent: ``step2FMM()``.
    Returns ∞ when any stability constraint is violated.
    """
    M, A, alpha, beta, omega = params
    data_min = data.min()
    data_max = data.max()
    slack    = 0.1 * (data_max - data_min)

    # Stability constraints (same as R)
    if (M + A) > (data_max + slack):  return np.inf
    if (M - A) < (data_min - slack):  return np.inf
    if A <= 0:                         return np.inf
    if omega <= 0:                     return np.inf
    if omega > omega_max:              return np.inf

    fitted = _fmm_predict(t, M, A, alpha, beta, omega)
    return float(np.sum((fitted - data)**2) / len(t))


# ---------------------------------------------------------------------------
# FMMModel class
# ---------------------------------------------------------------------------

class FMMModel(RhythmModel):
    """
    Single-component FMM (Flexible Multivariate Model) rhythm fitter.

    Fits  y(t) = M + A·cos(β + 2·atan(ω·tan((t − α)/2)))  using a
    grid-search + Nelder-Mead two-step algorithm.

    Parameters
    ----------
    length_alpha_grid : int
        Number of α values in the grid.  R default: 48.
    length_omega_grid : int
        Number of ω values in the grid.  R default: 24.
    omega_max : float
        Upper bound on ω.  R default: 1.0.
    num_reps : int
        Number of grid-refinement iterations.  R default: 3.

    Examples
    --------
    >>> import numpy as np
    >>> from circust.fitting.fmm import FMMModel
    >>>
    >>> t    = np.linspace(0, 2*np.pi, 50, endpoint=False)
    >>> data = 0.3 + 0.6 * np.cos(t - 1.2) + np.random.normal(0, 0.05, 50)
    >>> result = FMMModel().fit(data, t)
    >>> print(result.summary())
    """

    def __init__(
        self,
        length_alpha_grid: int   = 48,
        length_omega_grid: int   = 24,
        omega_max:         float = 1.0,
        num_reps:          int   = 3,
    ) -> None:
        self.length_alpha_grid = length_alpha_grid
        self.length_omega_grid = length_omega_grid
        self.omega_max         = omega_max
        self.num_reps          = num_reps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Fit the FMM model to ``data``.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples,)
            Observed expression values (normalised to [-1, 1]).
        time_points : np.ndarray, shape (n_samples,)
            Circular time axis in [0, 2π), typically ``CPCAResult.circular_scale``.

        Returns
        -------
        FitResult
            ``params`` keys: ``M``, ``A``, ``alpha``, ``beta``, ``omega``
        """
        data        = np.asarray(data,        dtype=float)
        time_points = np.asarray(time_points, dtype=float)

        # Initial grids (same as R defaults)
        alpha_grid = np.linspace(0, 2 * np.pi, self.length_alpha_grid, endpoint=False)
        omega_grid = np.exp(
            np.linspace(
                np.log(1e-4),
                np.log(self.omega_max),
                self.length_omega_grid,
            )
        )

        best_par = None

        for rep in range(self.num_reps):

            # ── Step 1: grid search ──────────────────────────────────────
            grid = np.array(
                [(a, w) for a in alpha_grid for w in omega_grid]
            )
            grid_results = np.array([
                _step1(row[0], row[1], data, time_points)
                for row in grid
            ])

            prev_best = best_par
            best_par  = _best_step1(data, grid_results)

            if best_par is None:
                # No stable candidate — revert to previous if available
                best_par = prev_best
                break

            # ── Step 2: Nelder-Mead refinement ───────────────────────────
            result = minimize(
                _step2_objective,
                x0     = best_par[:5],
                args   = (data, time_points, self.omega_max),
                method = "Nelder-Mead",
                options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
            )
            par_final = result.x.copy()

            # Wrap α and β into [0, 2π)
            par_final[2] = par_final[2] % (2 * np.pi)
            par_final[3] = par_final[3] % (2 * np.pi)

            # Narrow grids around current best for next iteration
            if rep < self.num_reps - 1:
                alpha_grid = self._refine_alpha_grid(
                    par_final[2], alpha_grid
                )
                omega_grid = self._refine_omega_grid(
                    par_final[4], omega_grid
                )

        if best_par is None:
            # Fallback: flat model
            par_final = np.array([data.mean(), 0.0, 0.0, 0.0, 0.5])
        
        M, A, alpha, beta, omega = par_final

        # ── Final fitted values & statistics ────────────────────────────
        fitted        = _fmm_predict(time_points, M, A, alpha, beta, omega)
        residuals     = data - fitted
        residuals_std = self._standardise_residuals(residuals)
        r2            = self._r2(data, fitted)
        sse           = float(np.sum(residuals**2))

        return FitResult(
            fitted        = fitted,
            params        = {
                "M":     float(M),
                "A":     float(A),
                "alpha": float(alpha),
                "beta":  float(beta),
                "omega": float(omega),
            },
            r2            = r2,
            residuals     = residuals,
            residuals_std = residuals_std,
            sse           = sse,
            model_name    = "fmm",
        )

    # ------------------------------------------------------------------
    # Grid refinement helpers
    # ------------------------------------------------------------------

    def _refine_alpha_grid(
        self,
        centre: float,
        prev_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Narrow the α grid around ``centre``.
        R equivalent: the alpha-grid narrowing in the numReps loop.
        """
        amplitude = 1.5 * np.mean(np.diff(np.sort(prev_grid)))
        new_grid  = np.linspace(
            centre - amplitude,
            centre + amplitude,
            len(prev_grid),
        )
        return new_grid % (2 * np.pi)

    def _refine_omega_grid(
        self,
        centre: float,
        prev_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Narrow the ω grid around ``centre``.
        R equivalent: the omega-grid narrowing in the numReps loop.
        """
        amplitude = 1.5 * np.mean(np.diff(np.sort(prev_grid)))
        new_grid  = np.linspace(
            max(centre - amplitude, 1e-6),
            min(centre + amplitude, self.omega_max),
            len(prev_grid),
        )
        return new_grid