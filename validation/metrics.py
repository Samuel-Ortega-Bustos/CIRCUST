"""
validation/metrics.py
=====================
Metricas de validacion para reconstruccion de orden temporal circular.

Cada estadistico es una funcion independiente con firma uniforme::

    fn(predicted, true, period=24.0) -> float

asi puede usarse tanto con datos sinteticos como con datos reales
(p.ej. tiempos GTEx, baboons con ZT conocido, etc.).

Contenido
---------
Helpers:
    - circular_abs_diff(a, b, period)        distancia circular elemento a elemento
    - mean_resultant_length(theta, period)   R_bar — concentracion (diagnostico)
    - circular_mean(theta, period)           MLE de mu bajo von Mises

Alineacion:
    - align_phases_by_mad(pred, true, period, try_reflection, grid_step)

Estadisticos individuales:
    - median_absolute_error(pred, true, period)        MedAE
    - std_absolute_error(pred, true, period)           SDAE
    - pct_within(pred, true, threshold_h, period)      %< Xh
    - auc_error_cdf(pred, true, period)                AUC del CDF
    - circular_correlation(pred, true, period)         CCC Jammalamadaka-Sarma

Atajo:
    - evaluate_all(pred, true, period, ...)            todas las metricas en dict
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PERIOD_HOURS: float = 24.0


# ===========================================================================
# Helpers numericos
# ===========================================================================

def circular_abs_diff(
    a:      np.ndarray,
    b:      np.ndarray,
    period: float = PERIOD_HOURS,
) -> np.ndarray:
    """
    Distancia circular absoluta elemento a elemento, en ``[0, period/2]``.

    Para puntos en un circulo, la distancia entre p.ej. 23 h y 1 h es 2 h
    (no 22 h). Esta funcion implementa ese atajo:

        d_i = min( |a_i - b_i| mod P,   P - |a_i - b_i| mod P )
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def mean_resultant_length(
    theta:  np.ndarray,
    period: float = PERIOD_HOURS,
) -> float:
    """
    Longitud del vector resultante :math:`\\bar R` de un conjunto de fases.

    Definicion::

        R_bar = sqrt( mean(cos)^2 + mean(sin)^2 )    en [0, 1]

    Interpretacion
    --------------
    - ``R_bar -> 1`` : distribucion **unimodal y concentrada** en torno a
      una direccion preferida.
    - ``R_bar -> 0`` : distribucion **uniforme** en el circulo (sin direccion
      preferida).

    Por que importa
    ---------------
    El CCC de Jammalamadaka depende de las medias circulares ``mu_1`` y
    ``mu_2``, que son inestables cuando ``R_bar -> 0``. Usar esta funcion
    como **diagnostico** para saber si el CCC del dataset es fiable:

    - ``R_bar > 0.3``  el CCC suele ser fiable
    - ``R_bar < 0.1``  el CCC se vuelve inestable y puede dar valores
                       negativos aunque la prediccion sea casi perfecta
    """
    theta = np.asarray(theta, dtype=np.float64)
    rad   = 2.0 * np.pi * theta / period
    c     = float(np.mean(np.cos(rad)))
    s     = float(np.mean(np.sin(rad)))
    return float(np.sqrt(c * c + s * s))


def circular_mean(
    theta:  np.ndarray,
    period: float = PERIOD_HOURS,
) -> float:
    """
    Direccion media circular: MLE de :math:`\\mu` bajo von Mises.

    Formula::

        mu_hat = arctan2( sum sin(theta),  sum cos(theta) )

    Devuelve un escalar en ``[0, period)``. Es numericamente inestable
    cuando :func:`mean_resultant_length` esta cerca de 0.
    """
    theta  = np.asarray(theta, dtype=np.float64)
    rad    = 2.0 * np.pi * theta / period
    s      = float(np.sin(rad).sum())
    c      = float(np.cos(rad).sum())
    mu_rad = np.arctan2(s, c) % (2.0 * np.pi)
    return float(mu_rad * period / (2.0 * np.pi))


# ===========================================================================
# Alineacion: rotacion + reflexion optima por MAD-minimo
# ===========================================================================

@dataclass
class AlignmentResult:
    """
    Resultado de :func:`align_phases_by_mad`.

    Attributes
    ----------
    delta : float
        Desplazamiento optimo en horas (sumado a la prediccion).
    sign : int
        +1 si no se aplico reflexion, -1 si se invirtio el recorrido.
    aligned_pred : np.ndarray
        Prediccion ya alineada, en ``[0, period)``.
    mad : float
        MAD del error tras la alineacion (criterio de optimizacion).
    """
    delta:        float
    sign:         int
    aligned_pred: np.ndarray
    mad:          float


def align_phases_by_mad(
    predicted:      np.ndarray,
    true:           np.ndarray,
    period:         float = PERIOD_HOURS,
    try_reflection: bool  = True,
    grid_step:      float = 0.05,
) -> AlignmentResult:
    """
    Encuentra la rotacion (y opcionalmente reflexion) que minimiza la
    **mediana del error absoluto circular** entre prediccion y verdad.

    Necesario porque algoritmos *unsupervised* (CIRCUST, CYCLOPS, CHIRAL,
    DCPR) producen fases definidas salvo rotacion y/o reflexion del circulo.
    Sin esta alineacion, una prediccion perfecta podria reportar un error
    arbitrariamente grande solo por una eleccion distinta del "origen".

    Parameters
    ----------
    predicted, true
        Fases en horas, asumidas en ``[0, period)``.
    period
        Periodo del ritmo, en horas. Por defecto 24.
    try_reflection
        Si ``True``, prueba tambien ``-predicted`` (recorrido en sentido
        opuesto). Util porque CIRCUST puede invertir la direccion segun
        la asuncion biologica de ARNTL/DBP.
    grid_step
        Paso del grid de busqueda en horas. ``0.05`` da resolucion fina
        sin penalizar el tiempo de calculo.

    Returns
    -------
    AlignmentResult
    """
    predicted = np.asarray(predicted, dtype=np.float64) % period
    true      = np.asarray(true,      dtype=np.float64) % period

    grid  = np.arange(0.0, period, grid_step)
    signs = [+1, -1] if try_reflection else [+1]

    best_mad   = np.inf
    best_delta = 0.0
    best_sign  = +1

    for sign in signs:
        candidate = (sign * predicted) % period
        for delta in grid:
            shifted = (candidate + delta) % period
            mad     = float(np.median(circular_abs_diff(shifted, true, period)))
            if mad < best_mad:
                best_mad   = mad
                best_delta = float(delta)
                best_sign  = sign

    aligned = (best_sign * predicted + best_delta) % period
    return AlignmentResult(
        delta        = best_delta,
        sign         = best_sign,
        aligned_pred = aligned,
        mad          = best_mad,
    )


# ===========================================================================
# Estadisticos individuales (una funcion por metrica)
# ===========================================================================

def median_absolute_error(
    predicted: np.ndarray,
    true:      np.ndarray,
    period:    float = PERIOD_HOURS,
) -> float:
    """
    Mediana del error absoluto circular (MedAE), en horas.

    Robusto a outliers — recomendada como metrica principal del paper DCPR.
    """
    return float(np.median(circular_abs_diff(predicted, true, period)))


def std_absolute_error(
    predicted: np.ndarray,
    true:      np.ndarray,
    period:    float = PERIOD_HOURS,
) -> float:
    """Desviacion estandar del error absoluto circular (SDAE), en horas."""
    return float(np.std(circular_abs_diff(predicted, true, period), ddof=0))


def pct_within(
    predicted:   np.ndarray,
    true:        np.ndarray,
    threshold_h: float,
    period:      float = PERIOD_HOURS,
) -> float:
    """
    Porcentaje de muestras con error absoluto **estrictamente menor** que
    ``threshold_h`` (en horas). Devuelve un valor en ``[0, 100]``.

    Tipicos usados en cronobiologia: ``threshold_h = 1`` y ``threshold_h = 2``.
    """
    errors = circular_abs_diff(predicted, true, period)
    if errors.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(errors < threshold_h))


def auc_error_cdf(
    predicted: np.ndarray,
    true:      np.ndarray,
    period:    float = PERIOD_HOURS,
) -> float:
    """
    Area bajo el CDF empirico del error absoluto, normalizada a ``[0, 1]``.

    Interpretacion:
        - ``AUC = 1.0``: predictor perfecto (todos los errores = 0).
        - ``AUC ~ 0.5``: predictor aleatorio (errores distribuidos
          uniformemente en ``[0, period/2]``).

    Es la metrica principal de resumen en el paper DCPR (Han et al. 2026).
    """
    errors = circular_abs_diff(predicted, true, period)
    n      = errors.size
    if n == 0:
        return float("nan")

    max_err  = period / 2.0
    sorted_e = np.sort(errors)
    cdf_x    = np.concatenate([[0.0], sorted_e, [max_err]])
    cdf_y    = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
    return float(np.trapezoid(cdf_y, cdf_x) / max_err)


def circular_correlation(
    predicted: np.ndarray,
    true:      np.ndarray,
    period:    float = PERIOD_HOURS,
) -> float:
    """
    Coeficiente de correlacion circular **rho** de Jammalamadaka & SenGupta
    (referencia [44] del paper CIRCUST de Larriba et al. 2023):

    .. math::

        \\rho =
        \\frac{\\sum_i \\sin(\\theta_{1i} - \\mu_1)\\, \\sin(\\theta_{2i} - \\mu_2)}
             {\\sqrt{\\sum_i \\sin^2(\\theta_{1i} - \\mu_1)\\,
                     \\sum_i \\sin^2(\\theta_{2i} - \\mu_2)}}

    con :math:`\\mu_1, \\mu_2` estimados por **maxima verosimilitud bajo von Mises**,
    que coincide con la media circular muestral
    :math:`\\hat\\mu = \\arctan2(\\sum \\sin\\theta, \\sum \\cos\\theta)`.

    Interpretacion:
        - ``rho = +1``: prediccion perfecta tras un shift constante.
        - ``rho =  0``: no hay correlacion lineal-circular.
        - ``rho = -1``: prediccion anticorrelacionada (sentido invertido).

    Limitacion (importante)
    -----------------------
    Esta metrica depende de las medias circulares. Cuando los datos cubren
    uniformemente el circulo (``mean_resultant_length -> 0``), las medias
    saltan a posiciones arbitrarias y rho se vuelve numericamente
    inestable, dando valores negativos incluso para predicciones casi
    perfectas. Para diagnosticar la fiabilidad:

        R_bar = mean_resultant_length(true, period)

    - ``R_bar > 0.3``: rho generalmente fiable.
    - ``R_bar < 0.1``: rho inestable; prefiere MedAE / AUC / %within Xh.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    true      = np.asarray(true,      dtype=np.float64)

    p_rad = 2.0 * np.pi * predicted / period
    t_rad = 2.0 * np.pi * true      / period

    mu_p = np.arctan2(np.sin(p_rad).sum(), np.cos(p_rad).sum())
    mu_t = np.arctan2(np.sin(t_rad).sum(), np.cos(t_rad).sum())

    sp = np.sin(p_rad - mu_p)
    st = np.sin(t_rad - mu_t)

    num = float((sp * st).sum())
    den = float(np.sqrt((sp ** 2).sum() * (st ** 2).sum()))
    return num / den if den > 0.0 else 0.0


# ===========================================================================
# API de conveniencia
# ===========================================================================

def evaluate_all(
    predicted:    np.ndarray,
    true:         np.ndarray,
    period:       float          = PERIOD_HOURS,
    thresholds_h: tuple          = (1.0, 2.0),
    align:        bool           = True,
) -> dict:
    """
    Calcula todas las metricas en un solo dict. Util en scripts de validacion.

    Parameters
    ----------
    predicted, true
        Fases en horas.
    period
        Periodo del ritmo, en horas (24 por defecto).
    thresholds_h
        Umbrales (en horas) para ``pct_within``. Por defecto ``(1, 2)``.
    align
        Si ``True`` (recomendado para datos unsupervised), aplica
        :func:`align_phases_by_mad` antes de calcular metricas y anade
        ``delta``, ``sign`` y ``mad`` al dict.

    Returns
    -------
    dict con claves:
        - ``MedAE``, ``SDAE``, ``AUC``, ``CCC``
        - ``pct_within_Xh`` para cada umbral
        - ``R_bar_true``, ``R_bar_pred``   (diagnostico de estabilidad)
        - ``n``                              (numero de muestras evaluadas)
        - ``delta``, ``sign``, ``mad``       (solo si ``align=True``)
    """
    if align:
        alig = align_phases_by_mad(predicted, true, period=period)
        pred = alig.aligned_pred
    else:
        pred = np.asarray(predicted, dtype=np.float64) % period

    true_a = np.asarray(true, dtype=np.float64) % period

    result: dict = {
        "MedAE":      median_absolute_error(pred, true_a, period),
        "SDAE":       std_absolute_error(pred, true_a, period),
        "AUC":        auc_error_cdf(pred, true_a, period),
        "CCC":        circular_correlation(pred, true_a, period),
        "R_bar_true": mean_resultant_length(true_a, period),
        "R_bar_pred": mean_resultant_length(pred,   period),
        "n":          int(true_a.size),
    }
    for th in thresholds_h:
        result[f"pct_within_{th:g}h"] = pct_within(pred, true_a, th, period)

    if align:
        result["delta"] = alig.delta
        result["sign"]  = alig.sign
        result["mad"]   = alig.mad

    return result
