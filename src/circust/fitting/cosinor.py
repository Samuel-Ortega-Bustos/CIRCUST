"""
circust/fitting/cosinor.py
===========================
Modelo ritmico Cosinor — ajuste sinusoidal por minimos cuadrados.

Modelo matematico
-----------------
    y(t) = M + A * cos(t + phi)

donde

    M   mesor        — nivel medio de la senal
    A   amplitud     — mitad del rango pico-a-valle
    phi acrofase     — momento del pico (desfase, en [0, 2*pi))

El modelo se ajusta por OLS reescribiendolo como regresion lineal:

    y = M + b*cos(t) + g*sin(t)
      = M + A*cos(t + phi)

con  b = A*cos(phi),  g = -A*sin(phi),  de modo que

    A   = sqrt(b**2 + g**2)
    phi = atan2(-g, b) mod 2*pi

Port exacto de ``funcionCosinor()`` de R en
``functionGTEX_cores.R`` (lineas 83-96).

Firma en R::

    funcionCosinor(datos, time, periodo)
    -> list(fitted, M, A, phase_vector, phi)

Equivalente en Python::

    CosinorModel().fit(data, time_points)
    -> FitResult(fitted, params={M, A, phi}, r2, residuals, ...)
"""
import numpy as np

from circust.fitting.rhythm_model import FitResult, RhythmModel


class CosinorModel(RhythmModel):
    """
    Modelo Cosinor de un solo componente.

    Ajusta  y(t) = M + A*cos(t + phi)  por minimos cuadrados ordinarios.

    Es el mas rapido de los modelos ritmicos — O(n) — y se usa tanto
    como test de ritmicidad independiente como estimador de parametros
    iniciales dentro de FMM.

    Ejemplo
    -------
    >>> import numpy as np
    >>> from circust.fitting.cosinor import CosinorModel
    >>>
    >>> t    = np.linspace(0, 2*np.pi, 50, endpoint=False)
    >>> data = 0.3 + 0.6 * np.cos(t - 1.2) + np.random.normal(0, 0.05, 50)
    >>> result = CosinorModel().fit(data, t)
    >>> print(result.summary())
    """

    @staticmethod
    def peak_time(phi: float) -> float:
        """
        Tiempo de pico del modelo Cosinor.

        El maximo de  M + A*cos(t + phi)  ocurre cuando t + phi = 0:
            t_peak = -phi  mod 2*pi
        """
        return float((-phi) % (2.0 * np.pi))

    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Ajusta el modelo Cosinor a ``data``.

        Parametros
        ----------
        data : np.ndarray, forma (n_muestras,)
            Valores de expresion observados (normalizados a [-1, 1]).
        time_points : np.ndarray, forma (n_muestras,)
            Eje temporal circular en [0, 2*pi).

        Devuelve
        --------
        FitResult
            Claves de ``params``: ``M``, ``A``, ``phi``
            ``fitted`` reproduce ``Mest + Aest*cos(time + phiEst)`` de R.
        """
        data        = np.asarray(data,        dtype=float)
        time_points = np.asarray(time_points, dtype=float)

        # ── Matriz de diseno OLS: [1, cos(t), sin(t)] ─────────────────
        xx = np.cos(time_points)
        zz = np.sin(time_points)
        X  = np.column_stack([np.ones(len(data)), xx, zz])

        # Resolucion por minimos cuadrados (equivalente a lm(data ~ xx + zz) de R)
        coeffs, *_ = np.linalg.lstsq(X, data, rcond=None)
        M_est  = coeffs[0]   # intercepto (mesor)
        bcos   = coeffs[1]   # coeficiente coseno
        bsin   = coeffs[2]   # coeficiente seno

        # ── Recuperacion de parametros ─────────────────────────────────
        A_est   = np.sqrt(bcos**2 + bsin**2)                    # amplitud
        phi_est = np.arctan2(-bsin, bcos) % (2 * np.pi)         # acrofase

        # ── Valores ajustados ──────────────────────────────────────────
        # R: Mest + Aest*cos(time + phiEst)
        fitted = M_est + A_est * np.cos(time_points + phi_est)

        # ── Residuos y R-cuadrado ──────────────────────────────────────
        residuals     = data - fitted
        residuals_std = self._standardise_residuals(residuals, 3)
        r2            = self._r2(data, fitted)
        sse           = float(np.sum(residuals**2))

        return FitResult(
            fitted        = fitted,
            params        = {"M": float(M_est), "A": float(A_est), "phi": float(phi_est)},
            peak_time     = CosinorModel.peak_time(float(phi_est)),
            r2            = r2,
            residuals     = residuals,
            residuals_std = residuals_std,
            sse           = sse,
            model_name    = "cosinor",
        )
