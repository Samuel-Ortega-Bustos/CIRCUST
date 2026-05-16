#!/usr/bin/env python3
"""
validate_synthetic_circust.py
==============================
Valida CIRCUST contra los datasets sinteticos de UNA replica del paper DCPR
(Han et al. 2026, Tabla 1): SynDST4, SynDST5, SynDST9, SynDST10, SynDST11.

Para cada dataset:
  1. Genera la matriz expr + tiempos reales segun la ecuacion 1 del paper
     (``validation.synthetic.generate_synthetic_dataset``).
  2. Ejecuta CIRCUST hasta la Etapa 2 (orden preliminar: CPCA + Synchronizer).
  3. Alinea fases predichas vs reales por MAD-minimo
     (``validation.metrics.align_phases_by_mad``).
  4. Reporta las metricas individuales del modulo ``validation.metrics``:
     MedAE, SDAE, AUC del CDF, CCC, %within 1h, %within 2h, R_bar_true.

Uso
---
    # Los 5 datasets de una replica
    python validation/validate_synthetic_circust.py

    # Uno solo, con semilla fija
    python validation/validate_synthetic_circust.py --dataset SynDST4 --seed 42

    # Personalizar n_rhythmic / n_total / output dir
    python validation/validate_synthetic_circust.py \\
            --n-rhythmic 200 --n-total 1000 \\
            --output output/synth_2026-05-16

Salida
------
    <output>/
    ├── summary.csv                 (1 fila por dataset)
    ├── <DATASET>_matrix.csv        (si --save-matrices)
    ├── <DATASET>_true_times.csv    (si --save-matrices)
    └── <DATASET>_predictions.csv   (predicted, true, error por muestra)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup paths ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "validation"))

# --- CIRCUST pipeline (Etapas 1-2) ---------------------------------------
from circust.preprocessing import Preprocessor
from circust.cpca          import CPCA
from circust.synchronizer  import CircularSynchronizer

# --- Generador de datasets sinteticos ------------------------------------
from synthetic import (
    DCPR_DATASETS,
    PERIOD_HOURS,
    SEED_GENES,
    generate_synthetic_dataset,
)

# --- Metricas de validacion (modulo reutilizable) ------------------------
from metrics import (
    align_phases_by_mad,
    auc_error_cdf,
    circular_abs_diff,
    circular_correlation,
    mean_resultant_length,
    median_absolute_error,
    pct_within,
    std_absolute_error,
)


# ===========================================================================
# Pipeline reducido: hasta Etapa 2 (orden preliminar)
# ===========================================================================

def _run_circust_stage2(expr_raw: pd.DataFrame, verbose: bool = False):
    """
    Ejecuta CIRCUST hasta la salida del Synchronizer (Etapa 2). Devuelve la
    tupla ``(prep, cpca_result, sync_result)``.
    """
    prep = Preprocessor(
        sparse_threshold = 0.3,
        verbose          = verbose,
    ).run(expr_raw)

    cpca_result = CPCA(
        core_genes            = SEED_GENES,
        n_outlier_candidates  = 8,
        tight_radius          = 0.10,
        loose_radius          = 0.15,
        fmm_threshold         = 3.0,
        max_outlier_fraction  = 0.05,
        fmm_length_alpha_grid = 48,
        fmm_length_omega_grid = 24,
        fmm_num_reps          = 3,
        verbose               = verbose,
    ).run(prep.expr_norm)

    sync_result = CircularSynchronizer(
        anchor_gene    = "ARNTL",
        direction_gene = "DBP",
        verbose        = verbose,
    ).run(cpca_result, list(cpca_result.core_genes_found))

    return prep, cpca_result, sync_result


def _extract_predicted_phases_hours(cpca_result, sync_result) -> pd.Series:
    """
    Convierte ``circular_scale`` (rad) en horas en ``[0, 24)`` y devuelve
    una Series indexada por **nombre original de muestra**.

    El Synchronizer resetea los nombres de columna a ``range(n)`` (lineas
    450 y 535 de :mod:`circust.synchronizer`), asi que reconstruimos los
    nombres a partir de ``cpca_result.expr_norm_final.columns`` +
    ``sync.sample_order``::

        nombre_en_posicion_k = cpca_result.expr_norm_final.columns[sync.sample_order[k]]
    """
    clean_names = list(cpca_result.expr_norm_final.columns)
    sample_idx  = np.asarray(sync_result.sample_order, dtype=int)
    names       = [clean_names[i] for i in sample_idx]

    phases_rad = np.asarray(sync_result.circular_scale, dtype=np.float64)
    phases_h   = (phases_rad * PERIOD_HOURS / (2.0 * np.pi)) % PERIOD_HOURS
    return pd.Series(phases_h, index=names, name="predicted_h")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validacion sintetica de CIRCUST (datasets DCPR de 1 replica).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Datasets disponibles: {list(DCPR_DATASETS)}",
    )
    p.add_argument(
        "--dataset", default="all",
        help="Nombre de dataset (case-insensitive) o 'all'. "
             f"Disponibles: {list(DCPR_DATASETS)} | all",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Semilla del generador. Por defecto: 42.",
    )
    p.add_argument(
        "--n-rhythmic", type=int, default=200,
        help="Numero de genes ritmicos. Por defecto: 200.",
    )
    p.add_argument(
        "--n-total", type=int, default=1000,
        help="Numero total de genes (ritmicos + ruido). Por defecto: 1000.",
    )
    p.add_argument(
        "--snr-min", type=float, default=2.0,
        help="Sigma minimo (SNR bajo). Por defecto: 2.",
    )
    p.add_argument(
        "--snr-max", type=float, default=5.0,
        help="Sigma maximo (SNR alto). Por defecto: 5.",
    )
    p.add_argument(
        "-o", "--output", type=str, default="output/synthetic_validation",
        help="Directorio de salida. Por defecto: output/synthetic_validation",
    )
    p.add_argument(
        "--save-matrices", action="store_true",
        help="Guardar las matrices generadas (.csv) ademas del resumen.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Mostrar los logs de cada etapa de CIRCUST.",
    )
    return p.parse_args()


def _resolve_datasets(name: str) -> list[str]:
    if name.lower() == "all":
        return list(DCPR_DATASETS.keys())
    matches = [k for k in DCPR_DATASETS if k.lower() == name.lower()]
    if not matches:
        raise SystemExit(
            f"Dataset desconocido: '{name}'. "
            f"Disponibles: {list(DCPR_DATASETS)} | all"
        )
    return matches


def _banner(msg: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _ccc_reliability(r_bar: float) -> str:
    """Etiqueta cualitativa de fiabilidad del CCC segun R_bar de los true."""
    if   r_bar >= 0.30: return "fiable"
    elif r_bar >= 0.10: return "frontera"
    else:               return "inestable"


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _parse_args()
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    datasets_to_run = _resolve_datasets(args.dataset)

    _banner("Validacion sintetica CIRCUST (datasets DCPR)")
    print(f"  Datasets   : {datasets_to_run}")
    print(f"  Semilla    : {args.seed}")
    print(f"  Genes ritm.: {args.n_rhythmic}")
    print(f"  Genes tot. : {args.n_total}")
    print(f"  SNR sigma  : [{args.snr_min}, {args.snr_max}]")
    print(f"  Salida     : {out.resolve()}")

    rows: list[dict] = []
    t_start = time.time()

    for name in datasets_to_run:
        cfg = DCPR_DATASETS[name]
        _banner(f"Dataset: {name}")
        print(f"  n_samples = {cfg['n_samples']}    "
              f"time_span = {cfg['time_span']} h    "
              f"interval  = {cfg['interval']}")

        # 1) Generar
        ds = generate_synthetic_dataset(
            name       = name,
            seed       = args.seed,
            n_rhythmic = args.n_rhythmic,
            n_total    = args.n_total,
            snr_range  = (args.snr_min, args.snr_max),
            **cfg,
        )
        print(f"  Generado : {ds.expr.shape[0]} genes x {ds.expr.shape[1]} muestras")

        if args.save_matrices:
            ds.expr.to_csv(out / f"{name}_matrix.csv")
            ds.true_times.to_csv(out / f"{name}_true_times.csv")
            print(f"    Guardado: {name}_matrix.csv + true_times.csv")

        # 2) Pipeline CIRCUST (Etapas 1-2)
        try:
            t_pipe = time.time()
            prep, cres, sync = _run_circust_stage2(ds.expr, verbose=args.verbose)
            pipe_secs = time.time() - t_pipe
        except Exception as e:
            print(f"  [ERROR] Pipeline fallo: {type(e).__name__}: {e}")
            rows.append(dict(
                dataset      = name,
                error        = f"{type(e).__name__}: {e}",
                n_samples_in = ds.expr.shape[1],
            ))
            continue

        n_dropped = len(cres.samples_dropped)
        print(f"  Pipeline : {pipe_secs:.1f}s  |  outliers eliminados: {n_dropped}")
        print(f"  Direccion invertida (sync): {sync.direction_flipped}")

        # 3) Extraer fases predichas y emparejar con tiempos reales
        pred_h   = _extract_predicted_phases_hours(cres, sync)
        common   = pred_h.index.intersection(ds.true_times.index)
        pred_arr = pred_h.loc[common].values
        true_arr = ds.true_times.loc[common].values

        # 4) Alinear (rotacion + reflexion optima por MAD-min)
        alig = align_phases_by_mad(pred_arr, true_arr, period=PERIOD_HOURS)

        # 5) Cada estadistico de forma individual (modulo metrics)
        aligned    = alig.aligned_pred
        true_fold  = true_arr % PERIOD_HOURS

        medae      = median_absolute_error(aligned, true_fold, PERIOD_HOURS)
        sdae       = std_absolute_error  (aligned, true_fold, PERIOD_HOURS)
        auc        = auc_error_cdf       (aligned, true_fold, PERIOD_HOURS)
        ccc        = circular_correlation(aligned, true_fold, PERIOD_HOURS)
        pct1       = pct_within          (aligned, true_fold, 1.0, PERIOD_HOURS)
        pct2       = pct_within          (aligned, true_fold, 2.0, PERIOD_HOURS)
        r_bar_true = mean_resultant_length(true_fold, PERIOD_HOURS)
        r_bar_pred = mean_resultant_length(aligned,   PERIOD_HOURS)

        # 6) Guardar predicciones por muestra
        per_sample = pd.DataFrame({
            "sample":         common,
            "true_h":         true_arr,
            "predicted_h":    pred_arr,
            "aligned_pred_h": aligned,
            "abs_error_h":    circular_abs_diff(aligned, true_fold, PERIOD_HOURS),
        })
        per_sample.to_csv(out / f"{name}_predictions.csv", index=False)

        # 7) Resumen por dataset
        row = dict(
            dataset            = name,
            n_samples_in       = ds.expr.shape[1],
            n_samples_used     = len(common),
            n_outliers_dropped = n_dropped,
            time_span_h        = cfg["time_span"],
            interval           = cfg["interval"] if cfg["interval"] else "non-uniform",
            MedAE              = medae,
            SDAE               = sdae,
            AUC                = auc,
            CCC                = ccc,
            CCC_reliability    = _ccc_reliability(r_bar_true),
            R_bar_true         = r_bar_true,
            R_bar_pred         = r_bar_pred,
            pct_within_1h      = pct1,
            pct_within_2h      = pct2,
            alignment_delta_h  = alig.delta,
            alignment_sign     = alig.sign,
            direction_flipped  = sync.direction_flipped,
            pipeline_secs      = round(pipe_secs, 2),
        )
        rows.append(row)

        print(f"  ── Metricas ───────────────────────────")
        print(f"    MedAE        = {medae:7.3f} h")
        print(f"    SDAE         = {sdae:7.3f} h")
        print(f"    AUC          = {auc:7.3f}")
        print(f"    CCC          = {ccc:+7.3f}  "
              f"[R_bar(true)={r_bar_true:.3f} → {_ccc_reliability(r_bar_true)}]")
        print(f"    % within 1h  = {pct1:6.1f}%")
        print(f"    % within 2h  = {pct2:6.1f}%")
        print(f"    alignment    : delta={alig.delta:.3f}h  sign={alig.sign:+d}")

    # ── Resumen global ──────────────────────────────────────────────────
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "summary.csv", index=False)

        _banner("Resumen")
        cols_show = [c for c in [
            "dataset", "n_samples_used", "n_outliers_dropped",
            "MedAE", "SDAE", "AUC", "CCC", "CCC_reliability", "R_bar_true",
            "pct_within_1h", "pct_within_2h",
        ] if c in df.columns]
        print(df[cols_show].to_string(index=False))
        print(f"\n  Tiempo total: {time.time() - t_start:.1f}s")
        print(f"  Guardado    : {out.resolve() / 'summary.csv'}")


if __name__ == "__main__":
    main()
