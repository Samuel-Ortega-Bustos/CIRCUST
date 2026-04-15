#!/usr/bin/env python3
"""
run_pipeline.py
===============
Ejecución de extremo a extremo del pipeline CIRCUST (Etapas 1-2) con
visualizaciones diagnósticas completas. Soporta cualquier formato de
matriz de expresión (CSV, TSV, Parquet, Excel) y conjuntos de genes
core configurables.

Ejemplos
--------
    # Por defecto: matrixIn.parquet con genes core de Larriba
    python scripts/run_pipeline.py

    # Entrada CSV (neuronas glutamatérgicas BA46)
    python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv \\
                                   --label "BA46 glutamatergic"

    # Genes core personalizados (conjunto ratón Zhang et al. 2014)
    python scripts/run_pipeline.py --core-genes ARNTL,DBP,NR1D1,PER1,PER2,PER3

    # Parquet con columna de genes explícita + DPI de publicación
    python scripts/run_pipeline.py --data data/matrixIn.parquet \\
                                   --gene-column gene_id \\
                                   --dpi 300 -o output/pub_run

    # Parámetros del pipeline ajustados
    python scripts/run_pipeline.py --sparse-threshold 0.4 \\
                                   --outlier-uni-threshold 5.0 \\
                                   --fmm-reps 5

Estructura de salida
--------------------
    <output_dir>/
    ├── results/
    │   ├── preprocessing_summary.txt
    │   ├── cpca_summary.txt
    │   ├── outlier_summary.txt
    │   ├── preliminary_order_summary.txt
    │   ├── core_gene_peaks.csv
    │   ├── core_gene_r2.csv
    │   └── sample_order.csv
    └── figures/
        ├── 01_variance_explained.png
        ├── 02_cpca_scatter.png
        ├── 03_cpca_gene_panels.png
        ├── 04_core_gene_fits.png
        ├── 05_residual_strips.png
        ├── 06_residual_heatmap.png
        ├── 07_circular_peaks.png
        ├── 08_ordered_profiles.png
        ├── 09_r2_comparison.png
        ├── 10_day_night_diagram.png
        ├── 11_expression_heatmap.png
        └── 12_pipeline_summary.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Asegurar que la raíz del proyecto está en sys.path ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Importaciones del pipeline CIRCUST ─────────────────────────────────────
from circust.preprocessing import load_expression_matrix, Preprocessor
from circust.cpca import CPCA
from circust.synchronizer import CircularSynchronizer
from circust.nonparametric import NonParametricScorer
from circust.candidate_selection import CandidateSelector
from circust.reference_set import ReferenceSetBuilder
from circust.random_selection import RandomSelector
from circust.top_matrix import build_top_matrix
from circust.robust_estimation import RobustEstimator
from circust.core_genes import CoreGeneSelector

# ── Importaciones de visualización ─────────────────────────────────────────
from circust.visualization import (
    plot_pc_scatter,
    plot_gene_panels,
    plot_core_gene_fits,
    plot_residual_strips,
    plot_residual_heatmap,
    plot_circular_peaks,
    plot_ordered_profiles,
    plot_r2_comparison,
    plot_day_night_diagram,
    plot_pipeline_summary,
    plot_variance_explained,
    plot_expression_overview,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuración — editar esta sección para cambiar los valores por defecto
# ═══════════════════════════════════════════════════════════════════════════

# Parámetros del pipeline — modificar aquí o sobreescribir via CLI.
DEFAULT_CONFIG = {
    # Preprocesamiento
    "sparse_threshold":       0.3,      # fracción máx de ceros/NaN por gen
    # CPCA
    "n_outlier_candidates":   8,        # muestras cercanas al origen a examinar
    "tight_radius":           0.10,     # umbral radial primario
    "loose_radius":           0.15,     # umbral radial alternativo
    # Detección de outliers
    "outlier_fmm_threshold":   3.0,     # |res. estand.| FMM (Sec. 3.3 suplementario)
    "max_outlier_fraction":    0.05,    # límite máximo de muestras eliminables
    # Ajuste FMM
    "fmm_alpha_grid":         48,       # resolución de la rejilla alpha
    "fmm_omega_grid":         24,       # resolución de la rejilla omega
    "fmm_reps":               3,        # iteraciones de refinamiento de la rejilla
    # Anclas biológicas
    "anchor_gene":            "ARNTL",  # gen situado en pi (amanecer)
    "direction_gene":         "DBP",    # gen para verificación de dirección
}

# Nombres de columna de genes por formato de archivo. Los ficheros Parquet
# suelen almacenar nombres de genes en una columna explícita; CSV/TSV/Excel
# usan la primera columna por defecto. Añadir entradas aquí si los archivos
# usan una convención diferente.
_GENE_COL_DEFAULTS = {
    ".parquet": "gene_id",
    ".csv":     None,       # None → first column is the index
    ".tsv":     None,
    ".txt":     None,
    ".xlsx":    None,
    ".xls":     None,
}


def _resolve_gene_column(data_path: Path, override: str | None) -> str | None:
    """Detectar automáticamente gene_column por extensión de archivo, con sobreescritura opcional."""
    if override is not None:
        return override
    return _GENE_COL_DEFAULTS.get(data_path.suffix.lower(), None)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecutar el pipeline CIRCUST (Etapas 1-2) con diagnósticos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
ejemplos:
  # Por defecto (matrixIn.parquet, 12 genes core de Larriba)
  python scripts/run_pipeline.py

  # Entrada CSV — columna de genes detectada automáticamente
  python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv

  # Conjunto de genes con nombre
  python scripts/run_pipeline.py --core-genes zhang

  # Lista de genes personalizada
  python scripts/run_pipeline.py --core-genes PER1,PER2,CRY1,CRY2,ARNTL,DBP

conjuntos de genes disponibles: {list(CoreGeneSelector.PRESETS.keys())}
""",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta a la matriz de expresión (CSV, TSV, Parquet, Excel). "
             "Por defecto: data/matrixIn.parquet",
    )
    parser.add_argument(
        "--gene-column", type=str, default=None,
        help="Nombre de la columna identificadora de genes. Se detecta "
             "automáticamente por extensión si se omite (Parquet → 'gene_id', CSV → primera col).",
    )
    parser.add_argument(
        "--core-genes", type=str, default=None,
        help=f"Genes core: un conjunto con nombre {list(CoreGeneSelector.PRESETS.keys())} o "
             "símbolos separados por comas. Por defecto: larriba",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Etiqueta del dataset para los títulos de los gráficos. Por defecto: nombre del archivo.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directorio de salida. Por defecto: output/",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Resolución de las figuras. Por defecto: 150",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Omitir la generación de gráficos (solo resultados en texto/CSV).",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares
# ═══════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _save_figure(fig, path: Path, dpi: int) -> None:
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"    Saved: {path.name}")


def _save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"    Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    t_start = time.time()

    # ── Resolver entradas ────────────────────────────────────────────────
    data_path = Path(args.data) if args.data else (PROJECT_ROOT / "data" / "matrixIn.parquet")
    gene_column    = _resolve_gene_column(data_path, args.gene_column)
    core_selector  = CoreGeneSelector.from_string(args.core_genes)
    label          = args.label if args.label else data_path.stem

    output_dir = Path(args.output)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Mostrar configuración ────────────────────────────────────────────
    _banner("Configuración")
    print(f"  Archivo de datos : {data_path}")
    print(f"  Columna genes    : {gene_column or '(primera columna)'}")
    print(f"  Preset genes core: {core_selector._preset_name}")
    print(f"  Gen ancla        : {cfg['anchor_gene']}")
    print(f"  Gen dirección    : {cfg['direction_gene']}")
    print(f"  Etiqueta         : {label}")
    print(f"  Salida           : {output_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 0 — Cargar datos
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 0: Cargando matriz de expresión")

    raw_matrix = load_expression_matrix(str(data_path), gene_column=gene_column)
    print(f"  Cargado: {raw_matrix.shape[0]} genes x {raw_matrix.shape[1]} muestras")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.0 — Preprocesamiento
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.0: Preprocesamiento")

    preprocessor = Preprocessor(
        sparse_threshold=cfg["sparse_threshold"],
        verbose=True,
    )
    prep = preprocessor.run(raw_matrix)

    _save_text(prep.summary(), results_dir / "preprocessing_summary.txt")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.1a — Seleccion de genes core
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.1a: Seleccion de Genes Core")

    core_result = core_selector.select(prep.expr_norm)
    core_genes  = core_result.genes

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 1.1 — CPCA (ordenamiento circular inicial)
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 1.1: PCA Circular")

    cpca = CPCA(
        core_genes            = core_genes,
        n_outlier_candidates  = cfg["n_outlier_candidates"],
        tight_radius          = cfg["tight_radius"],
        loose_radius          = cfg["loose_radius"],
        fmm_threshold         = cfg["outlier_fmm_threshold"],
        max_outlier_fraction  = cfg["max_outlier_fraction"],
        fmm_length_alpha_grid = cfg["fmm_alpha_grid"],
        fmm_length_omega_grid = cfg["fmm_omega_grid"],
        fmm_num_reps          = cfg["fmm_reps"],
        verbose               = True,
    )
    cpca_result = cpca.run(prep.expr_norm)

    _save_text(cpca_result.summary(), results_dir / "cpca_summary.txt")

    if not args.no_plots:
        print("  Generando gráficos CPCA ...")

        fig = plot_variance_explained(cpca_result, title=label)
        _save_figure(fig, figures_dir / "01_variance_explained.png", args.dpi)

        fig = plot_pc_scatter(cpca_result, title=label)
        _save_figure(fig, figures_dir / "02_cpca_scatter.png", args.dpi)

        fig = plot_gene_panels(cpca_result, title=label)
        _save_figure(fig, figures_dir / "03_cpca_gene_panels.png", args.dpi)

    if not args.no_plots:
        print("  Generando gráficos diagnósticos de outliers ...")

        fig = plot_core_gene_fits(cpca_result, title=label)
        _save_figure(fig, figures_dir / "04_core_gene_fits.png", args.dpi)

        fig = plot_residual_strips(cpca_result, title=label)
        _save_figure(fig, figures_dir / "05_residual_strips.png", args.dpi)

        fig = plot_residual_heatmap(cpca_result, title=label)
        _save_figure(fig, figures_dir / "06_residual_heatmap.png", args.dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 2 — Ordenamiento preliminar
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 2: Ordenamiento Circular Preliminar")

    core_genes_found = list(cpca_result.core_genes_found)
    estimator = CircularSynchronizer(
        anchor_gene    = cfg["anchor_gene"],
        direction_gene = cfg["direction_gene"],
        verbose        = True,
    )
    order_result = estimator.run(cpca_result, core_genes_found)

    _save_text(order_result.summary(), results_dir / "preliminary_order_summary.txt")

    # ── Exportar resultados clave como CSV ──────────────────────────────
    peak_df = pd.DataFrame({
        "gene": order_result.core_genes,
        "peak_phase": order_result.peak_times,
        "r2_fmm": order_result.r2_fmm,
        "classification": [
            "anchor" if g == "ARNTL"
            else "direction" if g == "DBP"
            else "day" if g in order_result.day_genes
            else "night"
            for g in order_result.core_genes
        ],
    })
    peak_df.to_csv(results_dir / "core_gene_peaks.csv", index=False)
    print(f"    Saved: core_gene_peaks.csv")

    r2_df = pd.DataFrame({
        "gene": order_result.core_genes,
        "r2_fmm": order_result.r2_fmm,
    }).sort_values("r2_fmm", ascending=False)
    r2_df.to_csv(results_dir / "core_gene_r2.csv", index=False)
    print(f"    Saved: core_gene_r2.csv")

    order_df = pd.DataFrame({
        "position": np.arange(len(order_result.circular_scale)),
        "circular_phase": order_result.circular_scale,
        "sample_index": order_result.sample_order,
    })
    order_df.to_csv(results_dir / "sample_order.csv", index=False)
    print(f"    Saved: sample_order.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3.1 — Puntuación de ritmicidad no paramétrica (PAVA)
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3.1: Puntuación de Ritmicidad No Paramétrica")

    np_scorer = NonParametricScorer(verbose=True)
    np_result = np_scorer.run(order_result.expr_ordered)

    _save_text(np_result.summary(), results_dir / "nonparametric_summary.txt")

    np_df = pd.DataFrame({
        "gene":   np_result.gene_names,
        "r2_np":  np_result.r2,
        "mse_np": np_result.mse_np,
    }).sort_values("r2_np", ascending=False)
    np_df.to_csv(results_dir / "nonparametric_r2.csv", index=False)
    print(f"    Saved: nonparametric_r2.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3.2 — Selección de genes candidatos por sectores circulares
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3.2: Selección de Genes Candidatos")

    candidate_selector = CandidateSelector(
        tam=50,
        omega_min=0.1,
        r2_par_floor=0.4,
        fmm_length_alpha_grid=cfg["fmm_alpha_grid"],
        fmm_length_omega_grid=cfg["fmm_omega_grid"],
        fmm_num_reps=cfg["fmm_reps"],
        verbose=True,
    )
    candidate_result = candidate_selector.run(
        r2_np          = np_result.r2,
        expr_norm      = order_result.expr_ordered,
        circular_scale = order_result.circular_scale,
        r2_par_core    = order_result.r2_fmm,
    )

    _save_text(candidate_result.summary(), results_dir / "candidate_summary.txt")

    cand_df = pd.DataFrame({
        "gene":     candidate_result.gene_names,
        "sector":   candidate_result.sector_labels,
        "fmm_peak": candidate_result.fmm_peaks,
        "r2_par":   candidate_result.r2_par,
    }).sort_values(["sector", "r2_par"], ascending=[True, False])
    cand_df.to_csv(results_dir / "candidate_genes.csv", index=False)
    print(f"    Saved: candidate_genes.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3.3 / 3.4 — Conjunto de referencia mínimo
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3.3 / 3.4: Conjunto de Referencia Mínimo")

    refset_builder = ReferenceSetBuilder(
        arntl_gene=cfg["anchor_gene"],
        dbp_gene=cfg["direction_gene"],
        verbose=True,
    )
    refset_result = refset_builder.run(
        candidate      = candidate_result,
        expr_full_norm = order_result.expr_ordered,
        circular_scale = order_result.circular_scale,
        sample_order   = order_result.sample_order,
        r2_np_all      = np_result.r2,
        gene_names_all = np_result.gene_names,
    )

    _save_text(refset_result.summary(), results_dir / "reference_set_summary.txt")
    pd.DataFrame({
        "gene":     refset_result.gene_names,
        "sector":   refset_result.sector_labels,
        "fmm_peak": refset_result.fmm_peaks,
        "r2_par":   refset_result.r2_par,
    }).to_csv(results_dir / "reference_set.csv", index=False)
    print(f"    Saved: reference_set.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3.5 / 3.6 — Selección aleatoria controlada
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3.5 / 3.6: Selección Aleatoria Controlada")

    random_selector = RandomSelector(
        n_reps=5,
        sample_size_fraction=2.0 / 3.0,
        r2_min=0.5,
        max_attempts=5000,
        arntl_gene=cfg["anchor_gene"],
        dbp_gene=cfg["direction_gene"],
        seed=42,
        verbose=True,
    )
    random_result = random_selector.run(
        reference      = refset_result,
        expr_full_norm = order_result.expr_ordered,
    )

    _save_text(random_result.summary(), results_dir / "random_selection_summary.txt")
    sel_df_rows = []
    for k in range(random_result.selection_indices.shape[0]):
        for j in range(random_result.selection_indices.shape[1]):
            sel_df_rows.append({
                "rep":      k + 1,
                "gene":     random_result.selection_names[k, j],
                "fmm_peak": random_result.selection_fmm_peaks[k, j],
                "r2_par":   random_result.selection_r2[k, j],
            })
    pd.DataFrame(sel_df_rows).to_csv(
        results_dir / "random_selections.csv", index=False
    )
    print(f"    Saved: random_selections.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 3.7 — Construcción de la matriz TOP
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 3.7: Matriz TOP (cores + candidatos)")

    top_matrix = build_top_matrix(
        core_genes         = core_genes_found,
        candidate_names    = refset_result.gene_names,
        expr_full_norm     = order_result.expr_ordered,
        candidate_norm_ord = refset_result.candidate_matrix,
        sample_order       = np.arange(order_result.expr_ordered.shape[1]),
        genes_to_exclude   = refset_result.added_genes,
    )
    print(f"  Matriz TOP: {top_matrix.shape[0]} genes x {top_matrix.shape[1]} muestras")
    top_matrix.to_csv(results_dir / "top_matrix.csv")
    print(f"    Saved: top_matrix.csv")

    # ─────────────────────────────────────────────────────────────────────
    # Etapa 4 — Estimación robusta
    # ─────────────────────────────────────────────────────────────────────
    _banner("Etapa 4: Estimación Robusta")

    robust_est = RobustEstimator(
        arntl_gene=cfg["anchor_gene"],
        dbp_gene=cfg["direction_gene"],
        fmm_length_alpha_grid=cfg["fmm_alpha_grid"],
        fmm_length_omega_grid=cfg["fmm_omega_grid"],
        fmm_num_reps=cfg["fmm_reps"],
        verbose=True,
    )
    robust_result = robust_est.run(
        random_selection = random_result,
        top_matrix       = top_matrix,
        expr_full_norm   = order_result.expr_ordered,
        core_genes       = core_genes_found,
    )

    _save_text(robust_result.summary(), results_dir / "robust_estimation_summary.txt")
    pd.DataFrame({
        "sample_position": np.arange(len(robust_result.consensus_phase)),
        "consensus_phase": robust_result.consensus_phase,
    }).to_csv(results_dir / "robust_consensus_phase.csv", index=False)
    print(f"    Saved: robust_consensus_phase.csv")

    stats_cols = (
        ["fmm_M","fmm_A","fmm_alpha","fmm_beta","fmm_omega",
         "fmm_pkU","fmm_pkL","fmm_pkU_pct","fmm_pkL_pct",
         "fmm_s","fmm_mse","fmm_r2",
         "cos_M","cos_A","cos_phi",
         "cos_pk1","cos_pk2","cos_pk1_pct","cos_pk2_pct",
         "cos_s","cos_mse","cos_r2",
         "np_s","np_mse","np_r2"]
    )
    K = robust_result.sample_orders.shape[0]
    n_top = len(robust_result.top_gene_names)
    stats_df = pd.DataFrame(robust_result.stats_table, columns=stats_cols)
    stats_df.insert(0, "gene", np.tile(robust_result.top_gene_names, K))
    stats_df.insert(0, "rep",  np.repeat(np.arange(1, K + 1), n_top))
    stats_df.to_csv(results_dir / "robust_stats_table.csv", index=False)
    print(f"    Saved: robust_stats_table.csv")

    if not args.no_plots:
        print("  Generando gráficos de ordenamiento ...")

        fig = plot_circular_peaks(order_result, title=label)
        _save_figure(fig, figures_dir / "07_circular_peaks.png", args.dpi)

        fig = plot_ordered_profiles(order_result, title=label)
        _save_figure(fig, figures_dir / "08_ordered_profiles.png", args.dpi)

        fig = plot_r2_comparison(order_result, title=label)
        _save_figure(fig, figures_dir / "09_r2_comparison.png", args.dpi)

        fig = plot_day_night_diagram(order_result, title=label)
        _save_figure(fig, figures_dir / "10_day_night_diagram.png", args.dpi)

        fig = plot_expression_overview(
            order_result.expr_ordered,
            core_genes=order_result.core_genes,
            circular_scale=order_result.circular_scale,
            n_top=50,
            title=label,
        )
        _save_figure(fig, figures_dir / "11_expression_heatmap.png", args.dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Resumen compuesto
    # ─────────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _banner("Generando figura resumen del pipeline")

        fig = plot_pipeline_summary(
            cpca_result, order_result,
            title=label,
        )
        _save_figure(fig, figures_dir / "12_pipeline_summary.png", args.dpi)

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
    print(f"  Inversión direcc.  : {order_result.direction_flipped}")
    print(f"  Mediana R² NP      : {float(np.median(np_result.r2)):.3f}")
    print(f"  Genes candidatos   : {len(candidate_result.gene_names)}")
    print(f"  Conjunto referencia: {len(refset_result.gene_names)}")
    print(f"  Repeticiones K     : {robust_result.sample_orders.shape[0]}")
    print(f"  Flips de orient.   : {int(robust_result.direction_flipped.sum())}")
    print(f"  Salida             : {output_dir.resolve()}")
    print(f"  Tiempo transcurrido: {elapsed:.1f}s")
    print()

    # Resumen de picos de genes core
    print("  Picos de genes core (marco biológico):")
    print("  " + "-" * 44)
    for i, gene in enumerate(order_result.core_genes):
        phase = order_result.peak_times[i]
        r2 = order_result.r2_fmm[i]
        cls = peak_df.loc[peak_df["gene"] == gene, "classification"].values[0]
        print(f"    {gene:<8s}  phase={phase:5.3f} rad  R2={r2:.3f}  [{cls}]")
    print()


if __name__ == "__main__":
    main()
