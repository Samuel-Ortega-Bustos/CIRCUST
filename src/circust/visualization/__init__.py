"""
circust.visualization
=====================
Publication-quality plots for each stage of the CIRCUST pipeline.

Modules
-------
cpca_plots        — CPCA stage: PC scatter, gene expression panels
outlier_plots     — Outlier refinement: gene fits, residual strips, heatmap
order_plots       — Preliminary ordering: circular peaks, profiles, R² bars
pipeline_summary  — Composite summary, scree plot, expression heatmap

Quick reference
---------------
    from circust.visualization import (
        # CPCA (Stage 1.1)
        plot_pc_scatter,
        plot_gene_panels,
        # Outlier refinement (Stage 1.2)
        plot_core_gene_fits,
        plot_residual_strips,
        plot_residual_heatmap,
        # Preliminary ordering (Stage 2)
        plot_circular_peaks,
        plot_ordered_profiles,
        plot_r2_comparison,
        plot_day_night_diagram,
        # Pipeline summary
        plot_pipeline_summary,
        plot_variance_explained,
        plot_expression_overview,
    )
"""

from circust.visualization.cpca_plots import (
    plot_pc_scatter,
    plot_gene_panels,
)

from circust.visualization.outlier_plots import (
    plot_core_gene_fits,
    plot_residual_strips,
    plot_residual_heatmap,
)

from circust.visualization.order_plots import (
    plot_circular_peaks,
    plot_ordered_profiles,
    plot_r2_comparison,
    plot_day_night_diagram,
)

from circust.visualization.pipeline_summary import (
    plot_pipeline_summary,
    plot_variance_explained,
    plot_expression_overview,
)

__all__ = [
    # CPCA
    "plot_pc_scatter",
    "plot_gene_panels",
    # Outlier
    "plot_core_gene_fits",
    "plot_residual_strips",
    "plot_residual_heatmap",
    # Ordering
    "plot_circular_peaks",
    "plot_ordered_profiles",
    "plot_r2_comparison",
    "plot_day_night_diagram",
    # Summary
    "plot_pipeline_summary",
    "plot_variance_explained",
    "plot_expression_overview",
]
