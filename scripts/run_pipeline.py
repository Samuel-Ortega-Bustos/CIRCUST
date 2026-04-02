#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end execution of the CIRCUST pipeline (Stages 1-2) with full
diagnostic visualisations.  Supports any expression matrix format
(CSV, TSV, Parquet, Excel) and configurable core gene sets.

Examples
--------
    # Default: matrixIn.parquet with Larriba core genes
    python scripts/run_pipeline.py

    # CSV input (BA46 glutamatergic neurons)
    python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv \\
                                   --label "BA46 glutamatergic"

    # Custom core genes (Zhang et al. 2014 mouse set)
    python scripts/run_pipeline.py --core-genes ARNTL,DBP,NR1D1,PER1,PER2,PER3

    # Parquet with explicit gene column + publication DPI
    python scripts/run_pipeline.py --data data/matrixIn.parquet \\
                                   --gene-column gene_id \\
                                   --dpi 300 -o output/pub_run

    # Tuned pipeline parameters
    python scripts/run_pipeline.py --sparse-threshold 0.4 \\
                                   --outlier-uni-threshold 5.0 \\
                                   --fmm-reps 5

Output structure
----------------
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

# ── Ensure the project root is on sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── CIRCUST pipeline imports ────────────────────────────────────────────────
from circust.preprocessing import load_expression_matrix, Preprocessor
from circust.cpca import CPCA
from circust.outlier import OutlierRefiner
from circust.preliminary_order import PreliminaryOrderEstimator
from circust.constants import SEED_GENES_DEFAULT, SEED_GENES_ZHANG

# ── Visualisation imports ───────────────────────────────────────────────────
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
# Configuration — edit this section to change defaults
# ═══════════════════════════════════════════════════════════════════════════

# Named core-gene sets.  Use --core-genes <name> on the CLI,
# or add your own set here and reference it by name.
GENE_SETS = {
    "larriba": SEED_GENES_DEFAULT,     # Larriba et al. 2023 (12 genes)
    "zhang":   SEED_GENES_ZHANG,       # Zhang et al. 2014 (10 genes)
}

# Pipeline parameters — modify here or override via CLI.
DEFAULT_CONFIG = {
    # Preprocessing
    "sparse_threshold":       0.3,      # max zero/NaN fraction per gene
    # CPCA
    "n_outlier_candidates":   8,        # near-origin samples to examine
    "tight_radius":           0.10,     # primary radial threshold
    "loose_radius":           0.15,     # fallback radial threshold
    # Outlier detection
    "outlier_multi_threshold": 3.0,     # |std res| for multivariate
    "outlier_uni_threshold":   4.0,     # |std res| for univariate
    "max_outlier_fraction":    0.05,    # cap on removable samples
    # FMM fitting
    "fmm_alpha_grid":         48,       # alpha grid resolution
    "fmm_omega_grid":         24,       # omega grid resolution
    "fmm_reps":               3,        # grid refinement iterations
    # Biological anchors
    "anchor_gene":            "ARNTL",  # gene placed at pi (dawn)
    "direction_gene":         "DBP",    # gene for direction check
}

# Gene column names per file format.  Parquet files often store gene
# names in an explicit column; CSV/TSV/Excel use the first column by
# default.  Add entries here if your files use a different convention.
_GENE_COL_DEFAULTS = {
    ".parquet": "gene_id",
    ".csv":     None,       # None → first column is the index
    ".tsv":     None,
    ".txt":     None,
    ".xlsx":    None,
    ".xls":     None,
}


def _resolve_gene_column(data_path: Path, override: str | None) -> str | None:
    """Auto-detect gene_column from file extension, with optional override."""
    if override is not None:
        return override
    return _GENE_COL_DEFAULTS.get(data_path.suffix.lower(), None)


def _resolve_core_genes(raw: str | None) -> list[str]:
    """Parse --core-genes into a list of gene symbols."""
    if raw is None:
        return list(SEED_GENES_DEFAULT)
    # Named set?
    if raw.lower() in GENE_SETS:
        return list(GENE_SETS[raw.lower()])
    # Comma-separated list
    genes = [g.strip() for g in raw.split(",") if g.strip()]
    if len(genes) < 2:
        raise ValueError(
            f"--core-genes requires at least 2 genes, got: {genes}"
        )
    return genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CIRCUST pipeline (Stages 1-2) with diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  # Default (matrixIn.parquet, Larriba 12 core genes)
  python scripts/run_pipeline.py

  # CSV input — gene column auto-detected
  python scripts/run_pipeline.py --data data/BA46_glut_sample_no_minmax.csv

  # Named gene set
  python scripts/run_pipeline.py --core-genes zhang

  # Custom gene list
  python scripts/run_pipeline.py --core-genes PER1,PER2,CRY1,CRY2,ARNTL,DBP

available gene sets: {list(GENE_SETS.keys())}
""",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to expression matrix (CSV, TSV, Parquet, Excel). "
             "Default: data/matrixIn.parquet",
    )
    parser.add_argument(
        "--gene-column", type=str, default=None,
        help="Gene identifier column name. Auto-detected from file "
             "extension if omitted (Parquet → 'gene_id', CSV → first col).",
    )
    parser.add_argument(
        "--core-genes", type=str, default=None,
        help=f"Core genes: a named set {list(GENE_SETS.keys())} or "
             "comma-separated symbols. Default: larriba",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Dataset label for plot titles. Default: filename stem.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Output directory. Default: output/",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure resolution. Default: 150",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (text/CSV results only).",
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


def _save_figure(fig, path: Path, dpi: int) -> None:
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"    Saved: {path.name}")


def _save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"    Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    t_start = time.time()

    # ── Resolve inputs ───────────────────────────────────────────────────
    data_path = Path(args.data) if args.data else (PROJECT_ROOT / "data" / "matrixIn.parquet")
    gene_column = _resolve_gene_column(data_path, args.gene_column)
    core_genes = _resolve_core_genes(args.core_genes)
    label = args.label if args.label else data_path.stem

    output_dir = Path(args.output)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Print configuration ──────────────────────────────────────────────
    _banner("Configuration")
    print(f"  Data file    : {data_path}")
    print(f"  Gene column  : {gene_column or '(first column)'}")
    print(f"  Core genes   : {core_genes}")
    print(f"  Anchor gene  : {cfg['anchor_gene']}")
    print(f"  Direction    : {cfg['direction_gene']}")
    print(f"  Label        : {label}")
    print(f"  Output       : {output_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 0 — Load data
    # ─────────────────────────────────────────────────────────────────────
    _banner("Stage 0: Loading expression matrix")

    raw_matrix = load_expression_matrix(str(data_path), gene_column=gene_column)
    print(f"  Loaded: {raw_matrix.shape[0]} genes x {raw_matrix.shape[1]} samples")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1.0 — Preprocessing
    # ─────────────────────────────────────────────────────────────────────
    _banner("Stage 1.0: Preprocessing")

    preprocessor = Preprocessor(
        sparse_threshold=cfg["sparse_threshold"],
        verbose=True,
    )
    prep = preprocessor.run(raw_matrix)

    _save_text(prep.summary(), results_dir / "preprocessing_summary.txt")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1.1 — CPCA (initial circular ordering)
    # ─────────────────────────────────────────────────────────────────────
    _banner("Stage 1.1: Circular PCA")

    cpca = CPCA(
        core_genes=core_genes,
        n_outlier_candidates=cfg["n_outlier_candidates"],
        tight_radius=cfg["tight_radius"],
        loose_radius=cfg["loose_radius"],
        verbose=True,
    )
    cpca_result = cpca.run(prep.expr_norm)

    _save_text(cpca_result.summary(), results_dir / "cpca_summary.txt")

    if not args.no_plots:
        print("  Generating CPCA plots ...")

        fig = plot_variance_explained(cpca_result, title=label)
        _save_figure(fig, figures_dir / "01_variance_explained.png", args.dpi)

        fig = plot_pc_scatter(cpca_result, title=label)
        _save_figure(fig, figures_dir / "02_cpca_scatter.png", args.dpi)

        fig = plot_gene_panels(cpca_result, title=label)
        _save_figure(fig, figures_dir / "03_cpca_gene_panels.png", args.dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1.2 — Outlier refinement
    # ─────────────────────────────────────────────────────────────────────
    _banner("Stage 1.2: Outlier Refinement")

    refiner = OutlierRefiner(
        multi_threshold=cfg["outlier_multi_threshold"],
        uni_threshold=cfg["outlier_uni_threshold"],
        max_outlier_fraction=cfg["max_outlier_fraction"],
        fmm_length_alpha_grid=cfg["fmm_alpha_grid"],
        fmm_length_omega_grid=cfg["fmm_omega_grid"],
        fmm_num_reps=cfg["fmm_reps"],
        verbose=True,
    )
    outlier_result = refiner.run(cpca_result, prep.expr_norm)

    _save_text(outlier_result.summary(), results_dir / "outlier_summary.txt")

    if not args.no_plots:
        print("  Generating outlier diagnostic plots ...")

        fig = plot_core_gene_fits(outlier_result, cpca_initial=cpca_result, title=label)
        _save_figure(fig, figures_dir / "04_core_gene_fits.png", args.dpi)

        fig = plot_residual_strips(outlier_result, title=label)
        _save_figure(fig, figures_dir / "05_residual_strips.png", args.dpi)

        fig = plot_residual_heatmap(outlier_result, title=label)
        _save_figure(fig, figures_dir / "06_residual_heatmap.png", args.dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2 — Preliminary ordering
    # ─────────────────────────────────────────────────────────────────────
    _banner("Stage 2: Preliminary Circular Ordering")

    core_genes_found = list(outlier_result.cpca_final.core_genes_found)
    estimator = PreliminaryOrderEstimator(
        arntl_gene=cfg["anchor_gene"],
        dbp_gene=cfg["direction_gene"],
        verbose=True,
    )
    order_result = estimator.run(outlier_result, core_genes_found)

    _save_text(order_result.summary(), results_dir / "preliminary_order_summary.txt")

    # ── Export key results as CSV ────────────────────────────────────────
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

    if not args.no_plots:
        print("  Generating ordering plots ...")

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
    # Composite summary
    # ─────────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _banner("Generating pipeline summary figure")

        fig = plot_pipeline_summary(
            cpca_result, outlier_result, order_result,
            title=label,
        )
        _save_figure(fig, figures_dir / "12_pipeline_summary.png", args.dpi)

    # ─────────────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    _banner("Pipeline complete")
    print(f"  Dataset         : {data_path.name}")
    print(f"  Genes (input)   : {prep.n_genes_in}")
    print(f"  Genes (filtered): {prep.n_genes_out}")
    print(f"  Samples (clean) : {outlier_result.expr_norm_final.shape[1]}")
    print(f"  Outliers removed: {len(outlier_result.samples_dropped)}")
    print(f"  Direction flip  : {order_result.direction_flipped}")
    print(f"  Output          : {output_dir.resolve()}")
    print(f"  Elapsed time    : {elapsed:.1f}s")
    print()

    # Print core gene peak summary
    print("  Core gene peaks (biological frame):")
    print("  " + "-" * 44)
    for i, gene in enumerate(order_result.core_genes):
        phase = order_result.peak_times[i]
        r2 = order_result.r2_fmm[i]
        cls = peak_df.loc[peak_df["gene"] == gene, "classification"].values[0]
        print(f"    {gene:<8s}  phase={phase:5.3f} rad  R2={r2:.3f}  [{cls}]")
    print()


if __name__ == "__main__":
    main()
