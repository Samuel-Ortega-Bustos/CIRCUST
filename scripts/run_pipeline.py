#!/usr/bin/env python3
"""
run_pipeline.py
===============
Ejecucion del algoritmo CIRCUST (Etapas 1-4). **Solo** ejecuta el pipeline
y vuelca los resultados a CSV/TXT/JSON para que los consuman otros
programas (scripts de validacion sintetica, generadores de graficos en
Python o R, analisis aguas abajo, etc.).

No produce graficos: el plotting vive en scripts independientes.

Etapas ejecutadas
-----------------
    1.0  Preprocesamiento (min-max, filtrado de genes sparse)
    1.1  CPCA + deteccion de outliers
    2    Sincronizador (ancla ARNTL en pi, direccion via DBP)
    3    Seleccion de genes TOP (R2_ORI, R2_FMM, omega, cobertura)
    4    Estimacion robusta del orden (K repeticiones + agregacion)

Ejemplos
--------
    # Por defecto: data/matrixIn.parquet con genes core de Larriba
    python scripts/run_pipeline.py

    # Entrada CSV
    python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv

    # Preset alternativo de genes core
    python scripts/run_pipeline.py --core-genes zhang

    # Lista personalizada
    python scripts/run_pipeline.py --core-genes ARNTL,DBP,NR1D1,PER1,PER2,PER3

Salida
------
    <output>/
    ├── manifest.json         config + semilla (reproducibilidad)
    │
    ├── sample_order.csv      orden de muestras — UNA FILA POR MUESTRA.
    │                          Es la salida principal del TFG (mejora
    │                          en la inferencia del orden temporal de
    │                          muestras via agregacion de Barragan 2015).
    │                          Se valida contra ground truth (TOD o
    │                          tiempos reales conocidos):
    │                            sample_name     : nombre real de la muestra
    │                            circular_phase  : fase en RADIANES, [0, 2*pi) — canonico
    │                            phase_h         : fase en HORAS,   [0, 24)  — interpretable
    │
    └── circust_atlas.csv     [M^TOP] — atlas circadiano del tejido,
                              UNA FILA POR GEN TOP. Salida cientifica
                              canonica de CIRCUST (Larriba et al. 2023):
                                gene_name      : simbolo del gen TOP
                                r2_fmm         : R^2 del ajuste FMM
                                amplitude      : amplitud A
                                peak_phase_rad : pico t_U en [0, 2*pi)
                                peak_phase_h   : pico en [0, 24) horas
                                omega          : asimetria / agudeza
                                classification : 'day' (t_U in [0, pi))
                                                 o 'night' (t_U in [pi, 2pi))

Solo se generan estos tres archivos. Las metricas y resumenes intermedios
se imprimen por stdout durante la ejecucion pero no se guardan en disco.

Notas sobre la agregacion
-------------------------
La **agregacion opera a nivel de MUESTRAS** (propuesta del TFG, seccion 2:
aplicar la tecnica de Barragan 2015 al orden de muestras de CIRCUST):

- ``--robust-method best_k``: se selecciona la repeticion con mayor
  mediana R^2 y se usan su orden de muestras + sus parametros FMM por gen.
- ``--robust-method aggregate``: TSP3/HODs se aplica a la matriz
  ``(K reps x n_muestras)`` de fases asignadas por muestra para producir
  UN orden de muestras consenso. Despues se re-ajusta FMM por gen sobre
  ese orden consenso para obtener los parametros del atlas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── sys.path para importar el paquete circust ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from circust.preprocessing import load_expression_matrix, Preprocessor
from circust.cpca          import CPCA
from circust.synchronizer  import CircularSynchronizer
from circust.top_genes     import TopGeneSelector
from circust.robust_order  import RobustOrderEstimator
from circust.core_genes    import CoreGeneSelector


# ═══════════════════════════════════════════════════════════════════════════
# Configuracion por defecto — editar aqui o sobreescribir via CLI
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Preprocesamiento
    "sparse_threshold":       0.3,      # fraccion max de ceros/NaN por gen
    # CPCA
    "n_outlier_candidates":   8,        # muestras cercanas al origen a examinar
    "tight_radius":           0.10,     # umbral radial primario
    "loose_radius":           0.15,     # umbral radial alternativo
    # Deteccion de outliers
    "outlier_fmm_threshold":   3.0,     # |residuo estand.| FMM
    "max_outlier_fraction":    0.05,    # limite max de muestras eliminables
    # Ajuste FMM
    "fmm_alpha_grid":         48,       # resolucion de la rejilla alpha
    "fmm_omega_grid":         24,       # resolucion de la rejilla omega
    "fmm_reps":               3,        # iteraciones de refinamiento
    # Anclas biologicas
    "anchor_gene":            "ARNTL",  # gen situado en pi (amanecer)
    "direction_gene":         "DBP",    # gen para verificacion de direccion
}

# Nombre de la columna de genes por extension. Parquet suele tener una
# columna explicita; CSV/TSV usan la primera columna como indice.
_GENE_COL_DEFAULTS = {
    ".parquet": "gene_id",
    ".csv":     None,
    ".tsv":     None,
    ".txt":     None,
    ".xlsx":    None,
    ".xls":     None,
}


def _resolve_gene_column(data_path: Path, override: str | None) -> str | None:
    if override is not None:
        return override
    return _GENE_COL_DEFAULTS.get(data_path.suffix.lower(), None)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecutar el algoritmo CIRCUST (Etapas 1-4) y volcar CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
ejemplos:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv
  python scripts/run_pipeline.py --core-genes zhang
  python scripts/run_pipeline.py --core-genes ARNTL,DBP,NR1D1,PER1,PER2,PER3

conjuntos de genes disponibles: {list(CoreGeneSelector.PRESETS.keys())}
""",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta a la matriz de expresion (CSV, TSV, Parquet, Excel). "
             "Por defecto: data/matrixIn.parquet",
    )
    parser.add_argument(
        "--gene-column", type=str, default=None,
        help="Nombre de la columna identificadora de genes. Se detecta "
             "automaticamente por extension si se omite (Parquet -> 'gene_id', "
             "CSV -> primera col).",
    )
    parser.add_argument(
        "--core-genes", type=str, default=None,
        help=f"Genes core: preset {list(CoreGeneSelector.PRESETS.keys())} o "
             "lista separada por comas. Por defecto: circust.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Etiqueta del dataset (queda en manifest.json). Por defecto: "
             "nombre del archivo.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directorio de salida. Por defecto: output/",
    )

    # ── Parametros del algoritmo (5 esenciales) ──────────────────────────
    parser.add_argument(
        "--fmm-method", choices=["base", "stabilized"], default="base",
        help="Modelo FMM: 'base' (Larriba/Rueda 2019, MLE puro) o "
             "'stabilized' (power-roughness + KDE von Mises). "
             "Cuando se usa 'stabilized', CPCA calibra lambda y se hereda "
             "a TopGenes y RobustOrder. Por defecto: base.",
    )
    parser.add_argument(
        "--robust-method", choices=["best_k", "aggregate"], default="best_k",
        help="Como se elige el orden final entre las K repeticiones: "
             "'best_k' (mejor por R2 mediano) o 'aggregate' (TSP3/HODs). "
             "Por defecto: best_k.",
    )
    parser.add_argument(
        "--aggregation", choices=["tsp3", "hods"], default="tsp3",
        help="Metodo de agregacion de ordenes. Solo aplica si "
             "--robust-method=aggregate. Por defecto: tsp3.",
    )
    parser.add_argument(
        "--n-reps", type=int, default=5,
        help="Numero K de repeticiones aleatorias en RobustOrder. "
             "Por defecto: 5.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla del RNG (para las selecciones aleatorias de TOP "
             "en RobustOrder). Por defecto: 42.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


# Columnas de robust_result.stats_table en el orden esperado.
_STATS_COLS = [
    "fmm_M", "fmm_A", "fmm_alpha", "fmm_beta", "fmm_omega",
    "fmm_pkU", "fmm_pkL", "fmm_pkU_pct", "fmm_pkL_pct",
    "fmm_s", "fmm_mse", "fmm_r2",
    "cos_M", "cos_A", "cos_phi",
    "cos_pk1", "cos_pk2", "cos_pk1_pct", "cos_pk2_pct",
    "cos_s", "cos_mse", "cos_r2",
    "np_s", "np_mse", "np_r2",
]


def _build_atlas(
    robust_result,
    top_result,
    cpca_result,
    fmm_method:  str,
    cfg:         dict,
) -> pd.DataFrame:
    """
    Construye la tabla [M^TOP] del atlas circadiano del tejido
    (Larriba et al. 2023, Step 4): una fila por gen TOP con sus
    parametros FMM finales (R^2, A, t_U, omega) + clasificacion
    day/night, ordenada por R^2 descendente.

    La **agregacion opera a nivel de MUESTRAS** (propuesta del TFG,
    seccion 2: aplicar la tecnica de Barragan 2015 al orden de muestras):
    los K reps producen K orderings de muestras, y se consolida en UN
    orden consenso. Luego se ajusta FMM por gen sobre ese consenso.

    - ``method="best_k"``: usa los parametros de la repeticion seleccionada
      directamente desde ``stats_table``.
    - ``method="aggregate"``: el orden consenso (TSP3/HODs sobre los K
      orderings de muestras) no coincide con ninguna rep individual, asi
      que **se re-fittea FMM por gen** sobre el orden consenso para
      obtener los parametros canonicos.
    """
    n_top     = len(robust_result.top_gene_names)
    top_names = list(robust_result.top_gene_names)

    if robust_result.method_used == "best_k":
        sel   = int(robust_result.selected_rep)
        block = robust_result.stats_table[sel * n_top : (sel + 1) * n_top]
        df    = pd.DataFrame(block, columns=_STATS_COLS)
        raw   = pd.DataFrame({
            "gene_name":      top_names,
            "r2_fmm":         df["fmm_r2"].values,
            "amplitude":      df["fmm_A"].values,
            "peak_phase_rad": df["fmm_pkU"].values,
            "omega":          df["fmm_omega"].values,
        })
    else:
        # Re-fittear FMM sobre el orden consenso de MUESTRAS
        top_matrix_final = top_result.candidate_matrix.iloc[
            :, robust_result.sample_order
        ]
        final_scale = np.asarray(robust_result.circular_scale, dtype=np.float64)

        if fmm_method == "base":
            from circust.fitting.fmm import FMMModel
            fmm = FMMModel(
                length_alpha_grid = cfg["fmm_alpha_grid"],
                length_omega_grid = cfg["fmm_omega_grid"],
                num_reps          = cfg["fmm_reps"],
            )
        else:
            from circust.fitting.fmm_stabilized import (
                StabilizedFMMModel, prepare_dataset,
            )
            stab_kwargs: dict = {}
            if cpca_result.lambda_star is not None:
                stab_kwargs["lambda_star"] = float(cpca_result.lambda_star)
            prepared = prepare_dataset(final_scale, **stab_kwargs)
            fmm      = StabilizedFMMModel(prepared=prepared)

        rows = []
        for gene in top_names:
            y   = top_matrix_final.loc[gene].values.astype(np.float64)
            fit = fmm.fit(y, final_scale)
            rows.append({
                "gene_name":      gene,
                "r2_fmm":         float(fit.r2),
                "amplitude":      float(fit.params["A"]),
                "peak_phase_rad": float(fit.peak_time),
                "omega":          float(fit.params["omega"]),
            })
        raw = pd.DataFrame(rows)

    # Postprocesar: pico en horas + clasificacion day/night
    raw["peak_phase_h"]   = raw["peak_phase_rad"] * 24.0 / (2.0 * np.pi)
    raw["classification"] = np.where(
        raw["peak_phase_rad"] < np.pi, "day", "night",
    )
    atlas = raw[[
        "gene_name", "r2_fmm", "amplitude",
        "peak_phase_rad", "peak_phase_h",
        "omega", "classification",
    ]].sort_values("r2_fmm", ascending=False).reset_index(drop=True)
    return atlas


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    t_start = time.time()

    # ── Resolver entradas ────────────────────────────────────────────────
    data_path = Path(args.data) if args.data else (PROJECT_ROOT / "data" / "matrixIn.parquet")
    gene_column   = _resolve_gene_column(data_path, args.gene_column)
    core_selector = CoreGeneSelector.from_string(args.core_genes)
    label         = args.label if args.label else data_path.stem

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Mostrar configuracion ────────────────────────────────────────────
    _banner("Configuracion")
    print(f"  Archivo de datos : {data_path}")
    print(f"  Columna genes    : {gene_column or '(primera columna)'}")
    print(f"  Preset genes core: {core_selector._preset_name}")
    print(f"  Gen ancla        : {cfg['anchor_gene']}")
    print(f"  Gen direccion    : {cfg['direction_gene']}")
    print(f"  FMM method       : {args.fmm_method}")
    print(f"  Robust method    : {args.robust_method}"
          + (f"  (agregacion: {args.aggregation})"
             if args.robust_method == "aggregate" else ""))
    print(f"  K repeticiones   : {args.n_reps}")
    print(f"  Semilla          : {args.seed}")
    print(f"  Etiqueta         : {label}")
    print(f"  Salida           : {output_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 0 — Cargar datos
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 0: Cargando matriz de expresion")
    raw_matrix = load_expression_matrix(str(data_path), gene_column=gene_column)
    print(f"  Cargado: {raw_matrix.shape[0]} genes x {raw_matrix.shape[1]} muestras")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.0 — Preprocesamiento
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.0: Preprocesamiento")
    prep = Preprocessor(
        sparse_threshold=cfg["sparse_threshold"],
        verbose=True,
    ).run(raw_matrix)

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.1a — Seleccion de genes core
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.1a: Seleccion de Genes Core")
    core_result = core_selector.select(prep.expr_norm)
    core_genes  = core_result.genes

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.1 — CPCA + deteccion de outliers
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.1: PCA Circular")
    cpca_result = CPCA(
        core_genes            = core_genes,
        n_outlier_candidates  = cfg["n_outlier_candidates"],
        tight_radius          = cfg["tight_radius"],
        loose_radius          = cfg["loose_radius"],
        fmm_threshold         = cfg["outlier_fmm_threshold"],
        max_outlier_fraction  = cfg["max_outlier_fraction"],
        fmm_length_alpha_grid = cfg["fmm_alpha_grid"],
        fmm_length_omega_grid = cfg["fmm_omega_grid"],
        fmm_num_reps          = cfg["fmm_reps"],
        fmm_method            = args.fmm_method,
        verbose               = True,
    ).run(prep.expr_norm)

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 2 — Sincronizador (orden preliminar)
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 2: Ordenamiento Circular Preliminar")
    core_genes_found = list(cpca_result.core_genes_found)
    order_result = CircularSynchronizer(
        anchor_gene    = cfg["anchor_gene"],
        direction_gene = cfg["direction_gene"],
        verbose        = True,
    ).run(cpca_result, core_genes_found)

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3 — Seleccion de genes TOP
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3: Seleccion de Genes TOP")
    top_result = TopGeneSelector(
        r2_ori_threshold      = 0.5,
        r2_fmm_threshold      = 0.5,
        omega_min             = 0.1,
        n_sectors             = 4,
        fmm_length_alpha_grid = cfg["fmm_alpha_grid"],
        fmm_length_omega_grid = cfg["fmm_omega_grid"],
        fmm_num_reps          = cfg["fmm_reps"],
        fmm_method            = args.fmm_method,
        lambda_star           = cpca_result.lambda_star,   # heredado si stabilized
        n_jobs                = -1,
        verbose               = True,
    ).run(
        expr_norm      = order_result.expr_ordered,
        circular_scale = order_result.circular_scale,
        seed_genes     = core_genes_found,
        sample_order   = order_result.sample_order,
    )
    print(f"  Matriz TOP: {top_result.candidate_matrix.shape[0]} genes x "
          f"{top_result.candidate_matrix.shape[1]} muestras")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 4 — Estimacion robusta del orden
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 4: Estimacion Robusta del Orden")
    robust_result = RobustOrderEstimator(
        n_reps               = args.n_reps,
        sample_size_fraction = 2.0 / 3.0,
        method               = args.robust_method,
        aggregation_method   = args.aggregation,
        r2_min               = 0.5,
        max_attempts         = 5000,
        anchor_gene          = cfg["anchor_gene"],
        direction_gene       = cfg["direction_gene"],
        fmm_length_alpha_grid= cfg["fmm_alpha_grid"],
        fmm_length_omega_grid= cfg["fmm_omega_grid"],
        fmm_num_reps         = cfg["fmm_reps"],
        fmm_method           = args.fmm_method,
        lambda_star          = cpca_result.lambda_star,   # heredado si stabilized
        n_jobs               = 1,
        seed                 = args.seed,
        verbose              = True,
    ).run(
        top_result     = top_result,
        expr_full_norm = order_result.expr_ordered,
        core_genes     = core_genes_found,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Salida 1: orden de MUESTRAS — UNA FILA POR MUESTRA
    # ─────────────────────────────────────────────────────────────────────
    # Es la salida principal del TFG (mejora del orden de muestras via
    # aggregation de Barragan 2015 aplicada a las K iteraciones de CIRCUST,
    # seccion 2 de la propuesta). Es lo que se valida contra ground truth
    # (tiempos reales conocidos o TOD), como hace el paper original en
    # Fig 5 (skeletal muscle) y Fig 7A (baboons).
    robust_names = [order_result.expr_ordered.columns[i]
                    for i in robust_result.sample_order]
    phases_rad   = np.asarray(robust_result.circular_scale, dtype=np.float64)
    phases_h     = phases_rad * 24.0 / (2.0 * np.pi)
    pd.DataFrame({
        "sample_name":    robust_names,
        "circular_phase": phases_rad,   # radianes en [0, 2*pi) — canonico
        "phase_h":        phases_h,     # horas    en [0, 24)  — interpretable
    }).to_csv(output_dir / "sample_order.csv", index=False)
    print(f"    Saved: sample_order.csv  ({len(phases_rad)} muestras)")

    # ─────────────────────────────────────────────────────────────────────
    # Salida 2: [M^TOP] — atlas circadiano del tejido (UNA FILA POR GEN)
    # ─────────────────────────────────────────────────────────────────────
    # Salida cientifica canonica de CIRCUST (Larriba et al. 2023, Step 4):
    # para cada gen TOP, sus parametros FMM finales (R^2, amplitud, peak
    # phase, omega) y clasificacion day/night. Construido re-ajustando
    # FMM sobre el orden de muestras consenso (en aggregate) o tomando
    # los parametros de la rep seleccionada (en best_k).
    atlas_df = _build_atlas(
        robust_result = robust_result,
        top_result    = top_result,
        cpca_result   = cpca_result,
        fmm_method    = args.fmm_method,
        cfg           = cfg,
    )
    atlas_df.to_csv(output_dir / "circust_atlas.csv", index=False)
    print(f"    Saved: circust_atlas.csv  ({len(atlas_df)} genes TOP)")

    # ─────────────────────────────────────────────────────────────────────
    # Manifest (config + semilla) para reproducibilidad
    # ─────────────────────────────────────────────────────────────────────
    manifest = {
        "input": {
            "data_path":   str(data_path),
            "label":       label,
            "gene_column": gene_column,
            "core_preset": core_selector._preset_name,
            "core_genes":  list(core_genes),
        },
        "config":    cfg,
        "algorithm": {
            "fmm_method":    args.fmm_method,
            "robust_method": args.robust_method,
            "aggregation":   (args.aggregation
                              if args.robust_method == "aggregate" else None),
            "n_reps":        args.n_reps,
            "lambda_star":   cpca_result.lambda_star,
        },
        "seed":      args.seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  Saved: manifest.json")

    # ─────────────────────────────────────────────────────────────────────
    # Informe final
    # ─────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    _banner("Pipeline completado")
    print(f"  Dataset            : {data_path.name}")
    print(f"  Genes (entrada)    : {prep.n_genes_in}")
    print(f"  Genes (filtrados)  : {prep.n_genes_out}")
    print(f"  Muestras (limpias) : {cpca_result.expr_norm_final.shape[1]}")
    print(f"  Outliers eliminados: {len(cpca_result.samples_dropped)}")
    print(f"  Inversion direcc.  : {order_result.direction_flipped}")
    print(f"  Genes tras ORI     : {top_result.n_after_ori}")
    print(f"  Genes tras FMM     : {top_result.n_after_fmm}")
    print(f"  Genes TOP finales  : {len(top_result.gene_names)}")
    print(f"  Metodo orden rob.  : {robust_result.method_used}")
    print(f"  Repeticiones K     : {robust_result.sample_orders.shape[0]}")
    print(f"  Flips de orient.   : {int(robust_result.direction_flipped.sum())}")
    print(f"  Salida             : {output_dir.resolve()}")
    print(f"  Tiempo transcurrido: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
