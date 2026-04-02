"""
Interfaz compartida para todos los modelos de ajuste ritmico del pipeline CIRCUST.

Nota de diseno
--------------
``FitResult`` es intencionadamente un dataclass plano en lugar de una jerarquia
de clases.  Los modelos difieren en sus parametros (Cosinor tiene 3, FMM tiene 5),
por lo que se almacenan como un dict generico.  Todo lo demas valores ajustados,
residuos, R-cuadrado es comun a todos los modelos.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Contenedor de resultados
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Resultado de cualquier modelo de ajuste ritmico.

    Atributos
    ---------
    fitted : np.ndarray, dim (n_muestras,)
        Valores predichos por el modelo en cada punto temporal.

    params : dict
        Parametros estimados del modelo.

        Claves Cosinor : ``M``, ``A``, ``phi``
            - M   : mesor (nivel medio)
            - A   : amplitud
            - phi : acrofase en [0, 2*pi) — momento del pico

        Claves FMM : ``M``, ``A``, ``alpha``, ``beta``, ``omega``
            - M     : mesor
            - A     : amplitud
            - alpha : parametro de localizacion (centro de la region del pico) en [0, 2*pi)
            - beta  : parametro de forma en [0, 2*pi)
            - omega : parametro de asimetria en (0, 1]

    r2 : float
        Coeficiente de determinacion  R-cuadrado = 1 - SS_res / SS_tot.
        Rango de -inf a 1; un modelo plano (prediciendo la media) da 0.

    residuals : np.ndarray, forma (n_muestras,)
        Residuos brutos: datos - ajustado.

    residuals_std : np.ndarray, forma (n_muestras,)
        Residuos estandarizados: (residuos - media) / desv_std.
        Utilizados en la etapa de refinamiento de outliers.

    sse : float
        Suma de errores al cuadrado.

    model_name : str
        ``"cosinor"`` o ``"fmm"``.
    """

    fitted:        np.ndarray
    params:        dict
    r2:            float
    residuals:     np.ndarray
    residuals_std: np.ndarray
    sse:           float
    model_name:    str

    # ------------------------------------------------------------------
    # Metodos auxiliares
    # ------------------------------------------------------------------

    @property
    def peak_time(self) -> float:
        """
        Fase del pico en [0, 2*pi).

        Para Cosinor es ``params["phi"]``.
        Para FMM usa la formula cerrada ``compUU`` de R:
            t_U = alpha + 2*atan2((1/omega)*sin(-beta/2), cos(-beta/2))  mod 2*pi
        """
        if self.model_name == "cosinor":
            return float(self.params["phi"])
        elif self.model_name == "fmm":
            from circust.fitting.fmm import fmm_peak_time
            return fmm_peak_time(
                self.params["alpha"], self.params["beta"], self.params["omega"]
            )
        else:
            raise NotImplementedError(f"peak_time no implementado para {self.model_name}")

    def summary(self) -> str:
        lines = [
            f"=== Ajuste {self.model_name.upper()} ===",
            f"  R-cuadrado : {self.r2:.4f}",
            f"  SSE        : {self.sse:.6f}",
            f"  parametros : { {k: round(float(v), 5) for k, v in self.params.items()} }",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modelo base abstracto
# ---------------------------------------------------------------------------

class RhythmModel(ABC):
    """
    Interfaz comun para todos los modelos de ajuste de ritmos circadianos.

    Las subclases deben implementar :meth:`fit`.

    Los parametros pasados al constructor son hiperparametros que permanecen
    constantes entre llamadas a ``fit()`` (p.ej. tamanos de rejilla para FMM).
    """

    @abstractmethod
    def fit(
        self,
        data:        np.ndarray,
        time_points: np.ndarray,
    ) -> FitResult:
        """
        Ajusta el modelo a ``data`` observados en ``time_points``.

        Parametros
        ----------
        data : np.ndarray, forma (n_muestras,)
            Valores de expresion normalizados (tipicamente en [-1, 1]).
        time_points : np.ndarray, forma (n_muestras,)
            Eje temporal circular en [0, 2*pi), producido por CPCA
            (``CPCAResult.circular_scale``).

        Devuelve
        --------
        FitResult
        """

    # ------------------------------------------------------------------
    # Utilidades compartidas por todos los modelos
    # ------------------------------------------------------------------

    @staticmethod
    def _r2(data: np.ndarray, fitted: np.ndarray) -> float:
        """R-cuadrado = 1 - SS_res / SS_tot."""
        ss_res = np.sum((data - fitted) ** 2)
        ss_tot = np.sum((data - data.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _standardise_residuals(residuals: np.ndarray, ddof: int = 1) -> np.ndarray:
        """(res - media) / desv_std  con ``ddof`` grados de libertad.

        ddof=1 corresponde a ``sd()`` de R — el valor por defecto en CIRCUST.
        Devuelve ceros si desv_std == 0 (residuos constantes).
        """
        std = residuals.std(ddof=ddof)
        if std == 0:
            return np.zeros_like(residuals)
        return (residuals - residuals.mean()) / std
