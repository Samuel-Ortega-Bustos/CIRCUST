"""
circust/fitting/cosinor.py
===========================
Cosinor rhythm model — a least-squares sinusoidal fit.

Mathematical model
------------------
    y(t) = M + A · cos(t + φ)

where

    M   mesor        — mean level of the signal
    A   amplitude    — half the peak-to-trough range
    φ   acrophase    — time of the peak (phase lag, in [0, 2π))

The model is fit via OLS by rewriting it as a linear regression:

    y = M + b·cos(t) + g·sin(t)
      = M + A·cos(t + φ)

with  b = A·cos(φ),  g = −A·sin(φ),  so that

    A   = √(b² + g²)
    φ   = atan2(−g, b) mod 2π

This is an exact Python port of R's ``funcionCosinor()`` from
``functionGTEX_cores.R`` (lines 83-96).

R signature::

    funcionCosinor(datos, time, periodo)
    → list(fitted, M, A, phase_vector, phi)

Python equivalent::

    CosinorModel().fit(data, time_points)
    → FitResult(fitted, params={M, A, phi}, r2, residuals, …)
"""

from __future__ import annotations

import numpy as np

from .rhythm_model import FitResult, RhythmModel


class CosinorModel(RhythmModel):
    """
    Single-component Cosinor model.

    Fits  y(t) = M + A·cos(t + φ)  via ordinary least squares.

    This is the fastest of the rhythm models — O(n) — and is used both
    as a standalone rhythmicity test and as the initial-parameter
    estimator inside FMM.

    Examples
    --------
    >>> import numpy as np
    >>> from circust.fitting.cosinor import CosinorModel
    >>>
    >>> t    = np.linspace(0, 2*np.pi, 50, endpoint=False)
    >>> data = 0.3 + 0.6 * np.cos(t - 1.2) + np.random.normal(0, 0.05, 50)
    >>> result = CosinorModel().fit(data, t)
    >>> print(result.summary())
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Fit the Cosinor model to ``data``.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples,)
            Observed expression values (normalised to [-1, 1]).
        time_points : np.ndarray, shape (n_samples,)
            Circular time axis in [0, 2π), typically ``CPCAResult.circular_scale``.

        Returns
        -------
        FitResult
            ``params`` keys: ``M``, ``A``, ``phi``
            ``fitted`` reproduces R's ``Mest + Aest*cos(time + phiEst)``
        """
        data        = np.asarray(data,        dtype=float)
        time_points = np.asarray(time_points, dtype=float)

        # ── OLS design matrix: [1, cos(t), sin(t)] ─────────────────────
        xx = np.cos(time_points)
        zz = np.sin(time_points)
        X  = np.column_stack([np.ones(len(data)), xx, zz])

        # least-squares solve  (equivalent to R's lm(data ~ xx + zz))
        coeffs, *_ = np.linalg.lstsq(X, data, rcond=None)
        M_est =  coeffs[0]   # intercept  (mesor)
        b     =  coeffs[1]   # cos coefficient
        g     =  coeffs[2]   # sin coefficient

        # ── Parameter recovery ─────────────────────────────────────────
        A_est   = np.sqrt(b**2 + g**2)                    # amplitude
        phi_est = np.arctan2(-g, b) % (2 * np.pi)         # acrophase

        # ── Fitted values ───────────────────────────────────────────────
        # R: Mest + Aest*cos(time + phiEst)
        fitted = M_est + A_est * np.cos(time_points + phi_est)

        # ── Residuals & R² ──────────────────────────────────────────────
        residuals     = data - fitted
        residuals_std = self._standardise_residuals(residuals)
        r2            = self._r2(data, fitted)
        sse           = float(np.sum(residuals**2))

        return FitResult(
            fitted        = fitted,
            params        = {"M": float(M_est), "A": float(A_est), "phi": float(phi_est)},
            r2            = r2,
            residuals     = residuals,
            residuals_std = residuals_std,
            sse           = sse,
            model_name    = "cosinor",
        )