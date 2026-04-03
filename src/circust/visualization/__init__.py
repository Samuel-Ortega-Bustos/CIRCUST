"""
circust.visualization
=====================
Graficos de calidad para publicacion de cada etapa del pipeline CIRCUST.

Modulos
-------
cpca_plots        — Etapa CPCA: scatter PC, paneles de expresion genica
outlier_plots     — Refinamiento de outliers: ajustes genicos, strips de residuos, heatmap
order_plots       — Ordenamiento preliminar: picos circulares, perfiles, barras R²
pipeline_summary  — Resumen compuesto, scree plot, heatmap de expresion

Referencia rapida
------------------
    from circust.visualization import (
        # CPCA (Etapa 1.1)
        plot_pc_scatter,
        plot_gene_panels,
        # Refinamiento de outliers (Etapa 1.2)
        plot_core_gene_fits,
        plot_residual_strips,
        plot_residual_heatmap,
        # Ordenamiento preliminar (Etapa 2)
        plot_circular_peaks,
        plot_ordered_profiles,
        plot_r2_comparison,
        plot_day_night_diagram,
        # Resumen del pipeline
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
