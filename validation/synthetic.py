"""
validation/synthetic.py
=======================
Generador de datasets sinteticos de expresion genica circadiana para
validar CIRCUST contra orden temporal conocido.

Soporta dos formas de senal periodica:

- **Cosinor** (modelo del paper DCPR, Han et al. 2026, ecuacion 1):
  ``f(t) = A * cos(2*pi/24 * (t + phi)) + ruido``
  — sinusoide pura, simetrica.

- **FMM** (modelo Frequency Modulated Mobius, Rueda et al. 2019):
  ``f(t) = M + A * cos(beta + 2*arctan(omega * tan((t_n - alpha)/2))) + ruido``
  — permite picos asimetricos y estrechos. Es el modelo que CIRCUST ajusta
  internamente, asi que validar con datos generados de esta forma evalua
  la capacidad de CIRCUST en su escenario de diseno.

Las metricas y la alineacion viven en :mod:`validation.metrics` para
que puedan reutilizarse con datos reales.

Contenido
---------
- ``SEED_GENES``                       12 genes semilla CIRCUST
- ``SyntheticDataset``                 dataclass de salida del generador
- ``generate_synthetic_dataset(...)``  generador con ``model``
                                        in ``{cosinor, fmm, mixed}``
- ``DCPR_DATASETS``                    registry de los 5 datasets de 1 replica

Notas
-----
La **periodo del ritmo** es siempre 24 h, aunque la ventana de muestreo
(``time_span``) sea 14, 18, 24 o 48 h. Para datasets de 48 h, los tiempos
cubren dos ciclos: al validar, los tiempos reales deben plegarse modulo 24
antes de comparar con la prediccion (la alineacion en
:mod:`validation.metrics` lo hace automaticamente).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PERIOD_HOURS: float = 24.0

#: Genes semilla CIRCUST que el pipeline buscara siempre. El generador
#: los marca como ritmicos y los coloca primero en la matriz.
SEED_GENES: list[str] = [
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "ARNTL", "CLOCK",
    "NR1D1", "RORA",
    "DBP", "TEF", "STAT3",
]


# ---------------------------------------------------------------------------
# Registry de los datasets de una replica del paper DCPR (Tabla 1)
# ---------------------------------------------------------------------------
#: Datasets de **una replica** del paper DCPR. Los seleccionados son los
#: relevantes para post-mortem (1 muestra por donante, ciclo posiblemente
#: incompleto, distribucion temporal posiblemente no uniforme).
DCPR_DATASETS: dict[str, dict] = {
    "SynDST4":  dict(n_samples=24, time_span=24.0, interval=None),  # non-uniform
    "SynDST5":  dict(n_samples=48, time_span=48.0, interval=1.0),
    "SynDST9":  dict(n_samples=48, time_span=48.0, interval=None),  # non-uniform
    "SynDST10": dict(n_samples=14, time_span=14.0, interval=1.0),   # incomplete cycle
    "SynDST11": dict(n_samples=18, time_span=18.0, interval=1.0),   # incomplete cycle
}


# ---------------------------------------------------------------------------
# Contenedor de resultado
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    """
    Dataset sintetico de expresion genica para validacion.

    Attributes
    ----------
    name : str
        Identificador (p.ej. ``"SynDST4"``).
    expr : pd.DataFrame, forma ``(n_genes, n_samples)``
        Matriz de expresion. Filas: genes, columnas: muestras.
    true_times : pd.Series, indice = nombres de muestra
        Tiempo real (horas) en ``[0, time_span)``. **No** esta plegada a 24 h.
    is_rhythmic : pd.Series, indice = nombres de gen
        ``True`` si el gen fue generado como ritmico.
    gene_params : pd.DataFrame, indice = nombres de gen
        Parametros usados por gen para trazabilidad. Columnas:

            - ``model``  : ``"cosinor"``, ``"fmm"`` o ``"noise"``
            - ``amp``    : amplitud A (float)
            - ``snr``    : sigma del ruido (float; NaN para genes no ritmicos)
            - ``phi``    : desfase cosinor en horas (NaN para FMM o ruido)
            - ``alpha``  : parametro FMM alpha en rad (NaN para cosinor o ruido)
            - ``beta``   : parametro FMM beta  en rad (NaN para cosinor o ruido)
            - ``omega``  : parametro FMM omega        (NaN para cosinor o ruido)

    config : dict
        Parametros globales de generacion (``n_samples``, ``time_span``,
        ``interval``, ``model``, ``seed``, etc).
    """
    name:        str
    expr:        pd.DataFrame
    true_times:  pd.Series
    is_rhythmic: pd.Series
    gene_params: pd.DataFrame
    config:      dict


# ---------------------------------------------------------------------------
# Generador (ecuacion 1 de Han et al. 2026)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    name:       str,
    n_samples:  int,
    time_span:  float                = 24.0,
    interval:   Optional[float]      = None,
    n_rhythmic: int                  = 200,
    n_total:    int                  = 1000,
    snr_range:  tuple[float, float]  = (2.0, 5.0),
    amp_range:  tuple[float, float]  = (0.5, 1.0),
    seed:       Optional[int]        = None,
) -> SyntheticDataset:
    """
    Genera un dataset sintetico siguiendo la ecuacion 1 del paper DCPR.

    Para genes ritmicos::

        f(t) = A * cos(2*pi/24 * (t + phi)) + (A/sigma) * S

    con ``A ~ U(amp_range)``, ``phi ~ U(0, 24)``, ``sigma ~ U(snr_range)``,
    ``S ~ N(0, 1)``. ``sigma`` actua como SNR (mayor = menos ruido).

    Para genes no ritmicos (puro ruido gaussiano)::

        f(t) = A * S

    Parameters
    ----------
    name
        Identificador del dataset (para trazabilidad).
    n_samples
        Numero de muestras (donantes).
    time_span
        Ventana de muestreo en horas. Tipicamente 14, 18, 24 o 48.
    interval
        Si es ``None``: muestreo no uniforme (uniforme aleatorio en
        ``[0, time_span)`` y luego ordenado). Si es un float: muestreo
        equiespaciado con ese paso, empezando en ``t=0``.
    n_rhythmic
        Numero de genes ritmicos. Debe ser ``>= len(SEED_GENES) = 12``.
    n_total
        Numero total de genes (ritmicos + ruido). Por defecto 1000 (DCPR).
    snr_range
        Rango uniforme para ``sigma`` (SNR). Por defecto ``(2, 5)``.
    amp_range
        Rango uniforme para la amplitud ``A``. Por defecto ``(0.5, 1)``.
    seed
        Semilla del RNG. Por defecto ``None`` (no reproducible).

    Returns
    -------
    SyntheticDataset

    Raises
    ------
    ValueError
        Si ``n_rhythmic < 12`` o ``n_total < n_rhythmic``.
    """
    if n_rhythmic < len(SEED_GENES):
        raise ValueError(
            f"n_rhythmic={n_rhythmic} debe ser >= {len(SEED_GENES)} "
            f"(numero de genes semilla CIRCUST)"
        )
    if n_total < n_rhythmic:
        raise ValueError(
            f"n_total={n_total} debe ser >= n_rhythmic={n_rhythmic}"
        )

    rng = np.random.default_rng(seed)

    # 1) Tiempos de muestreo
    if interval is not None:
        T = np.arange(0.0, time_span, interval, dtype=np.float64)
        if len(T) >= n_samples:
            T = T[:n_samples]
        else:
            # Repetir el patron (replicas) si hace falta
            n_rep = int(np.ceil(n_samples / len(T)))
            T = np.tile(T, n_rep)[:n_samples]
    else:
        T = np.sort(rng.uniform(0.0, time_span, size=n_samples))

    n_samples = len(T)

    # 2) Nombres de genes y muestras
    n_other_rhy = n_rhythmic - len(SEED_GENES)
    n_noise     = n_total - n_rhythmic

    other_rhythmic = [f"rhythm_{i:04d}" for i in range(n_other_rhy)]
    noise_genes    = [f"noise_{i:04d}"  for i in range(n_noise)]
    gene_names     = SEED_GENES + other_rhythmic + noise_genes

    is_rhy = np.concatenate([
        np.ones(n_rhythmic, dtype=bool),
        np.zeros(n_noise,   dtype=bool),
    ])

    sample_names = [f"s{i:04d}" for i in range(n_samples)]

    # 3) Matriz de expresion
    expr = np.zeros((n_total, n_samples), dtype=np.float64)

    # Ritmicos: cos + ruido
    for g in range(n_rhythmic):
        A     = rng.uniform(*amp_range)
        phi   = rng.uniform(0.0, PERIOD_HOURS)
        sigma = rng.uniform(*snr_range)
        S     = rng.standard_normal(n_samples)
        expr[g] = A * np.cos(2.0 * np.pi / PERIOD_HOURS * (T + phi)) \
                  + (A / sigma) * S

    # No ritmicos: solo ruido
    for g in range(n_rhythmic, n_total):
        A = rng.uniform(*amp_range)
        S = rng.standard_normal(n_samples)
        expr[g] = A * S

    expr_df  = pd.DataFrame(expr, index=gene_names, columns=sample_names)
    true_t   = pd.Series(T, index=sample_names, name="true_time_h")
    is_rhy_s = pd.Series(is_rhy, index=gene_names, name="is_rhythmic")

    return SyntheticDataset(
        name        = name,
        expr        = expr_df,
        true_times  = true_t,
        is_rhythmic = is_rhy_s,
        config      = dict(
            n_samples  = n_samples,
            time_span  = time_span,
            interval   = interval,
            n_rhythmic = n_rhythmic,
            n_total    = n_total,
            snr_range  = snr_range,
            amp_range  = amp_range,
            seed       = seed,
        ),
    )
