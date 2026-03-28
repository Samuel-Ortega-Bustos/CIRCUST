"""
========================
Shared interface for all rhythm-fitting models in the CIRCUST pipeline.

Design note
-----------
``FitResult`` is intentionally a plain dataclass rather than a class
hierarchy.  The models differ in their *parameters* (Cosinor has 3,
FMM has 5) so those are stored as a plain dict rather than typed fields.
Everything else — fitted values, residuals, R² — is the same across
all models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Output of any rhythm-fitting model.

    Attributes
    ----------
    fitted : np.ndarray, shape (n_samples,)
        Values predicted by the model at each time point.

    params : dict
        Model-specific parameter estimates.

        Cosinor keys : ``M``, ``A``, ``phi``
            - M   : mesor (mean level)
            - A   : amplitude
            - phi : acrophase in [0, 2π)  — time of peak

        FMM keys : ``M``, ``A``, ``alpha``, ``beta``, ``omega``
            - M     : mesor
            - A     : amplitude
            - alpha : location parameter (peak region centre) in [0, 2π)
            - beta  : shape parameter in [0, 2π)
            - omega : skewness parameter in (0, 1]

    r2 : float
        Coefficient of determination  R² = 1 - SS_res / SS_tot.
        Ranges from -∞ to 1; a flat model (predicting the mean) gives 0.

    residuals : np.ndarray, shape (n_samples,)
        Raw residuals: data − fitted.

    residuals_std : np.ndarray, shape (n_samples,)
        Standardised residuals: (residuals − mean) / std.
        Used by the outlier-refinement stage.

    sse : float
        Sum of squared errors.

    model_name : str
        ``"cosinor"`` or ``"fmm"``.
    """

    fitted:        np.ndarray
    params:        dict
    r2:            float
    residuals:     np.ndarray
    residuals_std: np.ndarray
    sse:           float
    model_name:    str

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def peak_time(self) -> float:
        """
        Phase of the peak in [0, 2π).

        For Cosinor this is ``params["phi"]``.
        For FMM this is computed from alpha, beta, omega via the
        Möbius peak formula (see ``FMMModel.peak_time``).
        """
        if self.model_name == "cosinor":
            return float(self.params["phi"])
        elif self.model_name == "fmm":
            alpha = self.params["alpha"]
            beta  = self.params["beta"]
            omega = self.params["omega"]
            # peak of cos(beta + 2*atan(omega*tan((t-alpha)/2))) is at
            # t* = alpha + 2*atan(cos(beta/2) / (omega*sin(beta/2)+eps))
            # Simplified: t* where the argument of cos is 0 → beta + Mobius = 0
            # Use numerical search on a fine grid
            t_grid = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
            mobius = beta + 2 * np.arctan(
                omega * np.tan((t_grid - alpha) / 2)
            )
            return float(t_grid[np.argmin(np.abs(mobius % (2 * np.pi)))])
        else:
            raise NotImplementedError(f"peak_time not implemented for {self.model_name}")

    def summary(self) -> str:
        lines = [
            f"=== {self.model_name.upper()} Fit ===",
            f"  R²       : {self.r2:.4f}",
            f"  SSE      : {self.sse:.6f}",
            f"  params   : { {k: round(float(v), 5) for k, v in self.params.items()} }",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base model
# ---------------------------------------------------------------------------

class RhythmModel(ABC):
    """
    Common interface for all circadian rhythm fitting models.

    Subclasses must implement :meth:`fit`.

    Parameters passed to the constructor are hyperparameters that remain
    constant across calls to ``fit()`` (e.g. grid sizes for FMM).
    """

    @abstractmethod
    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Fit the model to ``data`` observed at ``time_points``.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples,)
            Normalised expression values (typically in [-1, 1]).
        time_points : np.ndarray, shape (n_samples,)
            Circular time axis in [0, 2π), as produced by CPCA
            (``CPCAResult.circular_scale``).

        Returns
        -------
        FitResult
        """

    # ------------------------------------------------------------------
    # Shared utility used by all models
    # ------------------------------------------------------------------

    @staticmethod
    def _r2(data: np.ndarray, fitted: np.ndarray) -> float:
        """R² = 1 − SS_res / SS_tot."""
        ss_res = np.sum((data - fitted) ** 2)
        ss_tot = np.sum((data - data.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _standardise_residuals(residuals: np.ndarray,ddof:int) -> np.ndarray:
        """(res − mean) / std, returns zeros if std == 0."""
        std = residuals.std(ddof=ddof)
        if std == 0:
            return np.zeros_like(residuals)
        return (residuals - residuals.mean()) / std