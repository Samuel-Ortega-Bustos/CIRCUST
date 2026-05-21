#!/usr/bin/env python3
"""
generate_synthetic.py
=====================
Genera un dataset sintetico de expresion genica circadiana y lo guarda
**en disco** como artefactos que pueden alimentar el resto del flujo.

Soporta dos formas de senal periodica
-------------------------------------
- **Cosinor** (modelo del paper DCPR, Han et al. 2026, ecuacion 1):
  ``f(t) = A * cos(2*pi/24 * (t + phi)) + ruido``
  — sinusoide pura, simetrica.

- **FMM** (modelo Frequency Modulated Mobius, Rueda et al. 2019):
  ``f(t) = M + A * cos(beta + 2*arctan(omega * tan((t_n - alpha)/2))) + ruido``
  — permite picos asimetricos y estrechos. Es el modelo que CIRCUST ajusta
  internamente, asi que validar con datos generados de esta forma evalua
  la capacidad de CIRCUST en su escenario de diseno.

Salidas a disco
---------------
    ``matrix.csv``        Matriz genes x muestras. Formato compatible
                          con ``run_pipeline.py`` (primera columna =
                          gene_symbol, resto = nombres de muestras).

    ``true_times.csv``    Ground truth temporal:
                              sample_name   : nombre de la muestra
                              true_time_h   : tiempo real en horas
                                              (sin plegar — puede exceder
                                              24 h si time_span > 24)
                              true_phase_h  : tiempo real plegado a [0, 24)
                              true_phase_rad: misma fase en [0, 2*pi)
                          Es el archivo que consume ``validate.py``.

    ``gene_params.csv``   Parametros usados por gen al generar (model,
                          amp, snr, phi/alpha/beta/omega). Util para
                          diagnostico post-hoc.

    ``manifest.json``     Configuracion completa del generador
                          (reproducibilidad).

Workflow tipico
---------------
    # 1) Generar
    python scripts/generate_synthetic.py --dataset SynDST4 --seed 42 \\
            -o output/synth/SynDST4

    # 2) Correr CIRCUST sobre la matriz generada
    python scripts/run_pipeline.py \\
            --data output/synth/SynDST4/matrix.csv \\
            -o output/run/SynDST4

    # 3) Validar contra el ground truth
    python scripts/validate.py \\
            --predicted output/run/SynDST4/sample_order.csv \\
            --true      output/synth/SynDST4/true_times.csv \\
            -o output/validate/SynDST4

Notas
-----
- La **periodo del ritmo** es siempre 24 h, aunque la ventana de muestreo
  (``time_span``) sea 14, 18, 24 o 48 h.
- Para datasets de 48 h los tiempos cubren dos ciclos; ``true_times.csv``
  guarda tanto el tiempo bruto (``true_time_h``) como la fase plegada
  modulo 24 (``true_phase_h``, ``true_phase_rad``).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# Constantes
# ===========================================================================

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

#: Datasets de **una replica** del paper DCPR (Han et al. 2026, Tabla 1).
DCPR_DATASETS: dict[str, dict] = {
    "SynDST4":  dict(n_samples=24, time_span=24.0, interval=None),   # non-uniform
    "SynDST5":  dict(n_samples=48, time_span=48.0, interval=1.0),
    "SynDST9":  dict(n_samples=48, time_span=48.0, interval=None),   # non-uniform
    "SynDST10": dict(n_samples=14, time_span=14.0, interval=1.0),    # incomplete cycle
    "SynDST11": dict(n_samples=18, time_span=18.0, interval=1.0),    # incomplete cycle
    "Mod": dict(n_samples=500, time_span=24, interval=None),    # Random

}


# ===========================================================================
# Generador
# ===========================================================================

@dataclass
class SyntheticDataset:
    """Contenedor del resultado del generador (in-memory)."""
    name:        str
    expr:        pd.DataFrame
    true_times:  pd.Series
    is_rhythmic: pd.Series
    gene_params: pd.DataFrame
    config:      dict


def _cosinor_signal(
    t_hours: np.ndarray,
    A:       float,
    phi_h:   float,
    period:  float = PERIOD_HOURS,
) -> np.ndarray:
    """Senal cosinor (DCPR, ecuacion 1): ``A * cos(2*pi/period * (t + phi))``."""
    return A * np.cos(2.0 * np.pi / period * (t_hours + phi_h))


def _fmm_signal(
    t_hours: np.ndarray,
    M:       float,
    A:       float,
    alpha:   float,
    beta:    float,
    omega:   float,
    period:  float = PERIOD_HOURS,
) -> np.ndarray:
    r"""
    Senal del modelo FMM (Rueda et al. 2019)::

        mu(t) = M + A * cos(beta + 2*arctan(omega*tan((t_n - alpha)/2)))

    con :math:`t_n = 2\pi t / period`. Implementado via forma racional
    estable (mismo enfoque que ``mobius_cos_sin`` del FMM estabilizado),
    evitando la singularidad de ``tan()`` cuando :math:`(t_n-\alpha)/2 = \pm\pi/2`.
    """
    t_rad     = 2.0 * np.pi * t_hours / period
    u         = (t_rad - alpha) / 2.0
    cu, su    = np.cos(u), np.sin(u)
    cu2, su2  = cu * cu, su * su
    om2_su2   = (omega * omega) * su2
    den       = cu2 + om2_su2
    den       = np.where(den < 1e-12, 1e-12, den)
    C         = (cu2 - om2_su2) / den
    S         = (2.0 * omega * su * cu) / den
    return M + A * (np.cos(beta) * C - np.sin(beta) * S)


def generate(
    name:        str,
    n_samples:   int,
    time_span:   float                = 24.0,
    interval:    Optional[float]      = None,
    n_rhythmic:  int                  = 200,
    n_total:     int                  = 1000,
    snr_range:   tuple[float, float]  = (2.0, 5.0),
    amp_range:   tuple[float, float]  = (0.5, 1.0),
    seed:        Optional[int]        = None,
    *,
    model:       str                  = "cosinor",
    omega_range: tuple[float, float]  = (0.2, 1.0),
) -> SyntheticDataset:
    """
    Genera un dataset sintetico. Ver docstring del modulo para detalles.

    Para genes ritmicos la forma de la senal depende de ``model``:

    - ``"cosinor"``: ``f(t) = A*cos(2pi/24*(t+phi)) + (A/sigma)*S``
    - ``"fmm"``    : ``f(t) = M + A*cos(beta + 2*atan(omega*tan((t_n-alpha)/2))) + (A/sigma)*S``
    - ``"mixed"``  : 50% cosinor + 50% FMM por gen (i.i.d.)

    Genes no ritmicos: ``f(t) = A*S``.

    Raises
    ------
    ValueError
        Si ``model`` no es valido, ``n_rhythmic < 12`` o ``n_total < n_rhythmic``.
    """
    valid_models = {"cosinor", "fmm", "mixed"}
    if model not in valid_models:
        raise ValueError(
            f"model debe ser uno de {sorted(valid_models)}, "
            f"recibido: {model!r}"
        )
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

    # 3) Pre-allocar matriz y registro de parametros por gen
    expr = np.zeros((n_total, n_samples), dtype=np.float64)
    params_rows: list[dict] = []

    # 4) Generar genes ritmicos
    for g in range(n_rhythmic):
        A     = rng.uniform(*amp_range)
        sigma = rng.uniform(*snr_range)
        noise = (A / sigma) * rng.standard_normal(n_samples)

        if model == "cosinor":
            use_fmm = False
        elif model == "fmm":
            use_fmm = True
        else:
            use_fmm = bool(rng.random() < 0.5)

        if use_fmm:
            alpha  = float(rng.uniform(0.0, 2.0 * np.pi))
            beta   = float(rng.uniform(0.0, 2.0 * np.pi))
            omega  = float(rng.uniform(*omega_range))
            signal = _fmm_signal(T, M=0.0, A=A, alpha=alpha, beta=beta, omega=omega)
            params_rows.append(dict(
                gene  = gene_names[g], model = "fmm",
                amp   = A,             snr   = sigma,
                phi   = np.nan,
                alpha = alpha,         beta  = beta,    omega = omega,
            ))
        else:
            phi    = float(rng.uniform(0.0, PERIOD_HOURS))
            signal = _cosinor_signal(T, A=A, phi_h=phi)
            params_rows.append(dict(
                gene  = gene_names[g], model = "cosinor",
                amp   = A,             snr   = sigma,
                phi   = phi,
                alpha = np.nan,        beta  = np.nan,  omega = np.nan,
            ))

        expr[g] = signal + noise

    # 5) Genes no ritmicos
    for g in range(n_rhythmic, n_total):
        A = rng.uniform(*amp_range)
        expr[g] = A * rng.standard_normal(n_samples)
        params_rows.append(dict(
            gene  = gene_names[g], model = "noise",
            amp   = A,             snr   = np.nan,
            phi   = np.nan,
            alpha = np.nan,        beta  = np.nan,  omega = np.nan,
        ))

    # 6) Empaquetar
    expr_df     = pd.DataFrame(expr, index=gene_names, columns=sample_names)
    true_t      = pd.Series(T, index=sample_names, name="true_time_h")
    is_rhy_s    = pd.Series(is_rhy, index=gene_names, name="is_rhythmic")
    gene_params = pd.DataFrame(params_rows).set_index("gene")

    return SyntheticDataset(
        name        = name,
        expr        = expr_df,
        true_times  = true_t,
        is_rhythmic = is_rhy_s,
        gene_params = gene_params,
        config      = dict(
            n_samples   = n_samples,
            time_span   = time_span,
            interval    = interval,
            n_rhythmic  = n_rhythmic,
            n_total     = n_total,
            snr_range   = snr_range,
            amp_range   = amp_range,
            model       = model,
            omega_range = omega_range,
            seed        = seed,
        ),
    )


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Genera un dataset sintetico de expresion circadiana y lo "
            "guarda en disco como matrix.csv + true_times.csv (consumibles "
            "por run_pipeline.py y validate.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Datasets DCPR disponibles: {list(DCPR_DATASETS)}",
    )
    p.add_argument(
        "--dataset", default="SynDST4",
        help=f"Nombre de dataset DCPR. Disponibles: {list(DCPR_DATASETS)}. "
             "Por defecto: SynDST4.",
    )
    p.add_argument("--seed",       type=int,   default=42,
                   help="Semilla del RNG. Defecto: 42.")
    p.add_argument("--n-rhythmic", type=int,   default=200,
                   help="Genes ritmicos. >= 12. Defecto: 200.")
    p.add_argument("--n-total",    type=int,   default=1000,
                   help="Genes totales (ritmicos + ruido). Defecto: 1000.")
    p.add_argument("--snr-min",    type=float, default=2.0,
                   help="Sigma minimo (SNR bajo). Defecto: 2.")
    p.add_argument("--snr-max",    type=float, default=5.0,
                   help="Sigma maximo (SNR alto). Defecto: 5.")
    p.add_argument("--model",      default="cosinor",
                   choices=["cosinor", "fmm", "mixed"],
                   help="Forma de la senal. Defecto: cosinor.")
    p.add_argument("--omega-min",  type=float, default=0.2,
                   help="Omega minimo para genes FMM. Defecto: 0.2.")
    p.add_argument("--omega-max",  type=float, default=1.0,
                   help="Omega maximo para genes FMM. Defecto: 1.0.")
    p.add_argument("-o", "--output", type=str, required=True,
                   help="Directorio de salida (se crea si no existe).")
    return p.parse_args()


def _resolve_dataset(name: str) -> str:
    matches = [k for k in DCPR_DATASETS if k.lower() == name.lower()]
    if not matches:
        raise SystemExit(
            f"Dataset desconocido: '{name}'. "
            f"Disponibles: {list(DCPR_DATASETS)}"
        )
    return matches[0]


def main() -> None:
    args = _parse_args()
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    name = _resolve_dataset(args.dataset)
    cfg  = DCPR_DATASETS[name]

    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"  Dataset    : {name}")
    print(f"  n_samples  : {cfg['n_samples']}")
    print(f"  time_span  : {cfg['time_span']} h")
    print(f"  interval   : {cfg['interval'] if cfg['interval'] else 'no uniforme'}")
    print(f"  modelo     : {args.model}")
    print(f"  genes      : {args.n_rhythmic} ritmicos / {args.n_total} totales")
    print(f"  SNR sigma  : [{args.snr_min}, {args.snr_max}]")
    if args.model in ("fmm", "mixed"):
        print(f"  omega FMM  : [{args.omega_min}, {args.omega_max}]")
    print(f"  seed       : {args.seed}")
    print(f"  output     : {out.resolve()}")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    # 1) Generar
    ds = generate(
        name        = name,
        seed        = args.seed,
        n_rhythmic  = args.n_rhythmic,
        n_total     = args.n_total,
        snr_range   = (args.snr_min, args.snr_max),
        model       = args.model,
        omega_range = (args.omega_min, args.omega_max),
        **cfg,
    )

    # 2) Matriz en formato consumible por run_pipeline.py
    matrix_path = out / "matrix.csv"
    ds.expr.index.name = "gene_symbol"
    ds.expr.to_csv(matrix_path)

    # 3) Ground truth temporal
    times_rad = (ds.true_times.values % PERIOD_HOURS) * (2.0 * np.pi / PERIOD_HOURS)
    times_df = pd.DataFrame({
        "sample_name":    ds.true_times.index,
        "true_time_h":    ds.true_times.values,
        "true_phase_h":   ds.true_times.values % PERIOD_HOURS,
        "true_phase_rad": times_rad,
    })
    times_path = out / "true_times.csv"
    times_df.to_csv(times_path, index=False)

    # 4) Parametros por gen
    params_path = out / "gene_params.csv"
    ds.gene_params.to_csv(params_path)

    # 5) Manifest
    manifest = {
        "name":        name,
        "model":       args.model,
        "seed":        args.seed,
        "n_rhythmic":  args.n_rhythmic,
        "n_total":     args.n_total,
        "snr_range":   [args.snr_min, args.snr_max],
        "omega_range": [args.omega_min, args.omega_max],
        "dcpr_config": cfg,
        "period_h":    PERIOD_HOURS,
        "outputs": {
            "matrix":      str(matrix_path.name),
            "true_times":  str(times_path.name),
            "gene_params": str(params_path.name),
        },
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    elapsed = time.time() - t_start
    n_fmm   = int((ds.gene_params["model"] == "fmm").sum())
    n_cos   = int((ds.gene_params["model"] == "cosinor").sum())

    print()
    print(f"  ✓ Generado en {elapsed:.1f}s")
    print(f"    matrix      : {ds.expr.shape[0]} genes x {ds.expr.shape[1]} muestras "
          f"({n_cos} cosinor + {n_fmm} fmm + ruido)")
    print(f"    Archivos en {out.resolve()}:")
    print(f"      - matrix.csv")
    print(f"      - true_times.csv")
    print(f"      - gene_params.csv")
    print(f"      - manifest.json")


if __name__ == "__main__":
    main()
