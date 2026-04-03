"""
circust/visualization/outlier_plots.py
======================================
Visualizaciones para la etapa de refinamiento de outliers (Etapa 1.2).

Tres tipos de gráficos:

  plot_core_gene_fits(result)
      Cuadrícula de trazas de expresión de genes core con ajustes FMM
      y Cosinor superpuestos.  Muestra la calidad de cada modelo en
      cada gen core y los tres eigengenes (PC1/PC2/PC3).
      Equivalente en R: ``outPar_*.png`` en giveMatIniNP_v3_cores.

  plot_residual_strips(result)
      Diagrama de tiras horizontal por gen de residuos FMM
      estandarizados con líneas umbral de ±3 (multivariante) y
      ±4 (univariante).  Las muestras outlier se resaltan.
      Equivalente en R: ``resOutPar_*.png``.

  plot_residual_heatmap(result)
      Mapa de calor de |residuos FMM estandarizados| a través de
      genes × muestras, con cortes de color en los dos umbrales de
      outliers.  Proporciona una vista global de qué muestras son
      problemáticas en los distintos genes.

Todas las funciones devuelven una Figure de matplotlib.  Ninguna llama a plt.show().
"""
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from circust.cpca import CPCAResult
from circust.outlier import OutlierRefinementResult


# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------
_FMM_COLOUR = "#E41A1C"      # red
_COS_COLOUR = "#377EB8"      # blue
_OUTLIER_COLOUR = "#FF7F00"  # orange
_THRESHOLD_COLOURS = {3: "#984EA3", 4: "#E41A1C"}  # purple / red


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Core gene fits (FMM + Cosinor overlay)
# ═══════════════════════════════════════════════════════════════════════════

def plot_core_gene_fits(
    result: OutlierRefinementResult,
    cpca_initial: CPCAResult | None = None,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
    show_eigengenes: bool = True,
) -> Figure:
    """
    Grid of core-gene expression with FMM and Cosinor model overlays.

    Each panel shows the observed expression (grey dots) ordered by
    circular phase, with the FMM fit (red) and Cosinor fit (blue) on
    top.  R² values for both models are annotated.

    Parameters
    ----------
    result : OutlierRefinementResult
        Output of ``OutlierRefiner.run()``.
    cpca_initial : CPCAResult, optional
        The initial CPCA result (before outlier removal).  If provided,
        the initial fits are plotted against the initial ordering.
        If None, the final CPCA result is used (fits must match in size).
    title : str
        Suptitle label.
    figsize : tuple, optional
        Figure size.  Auto-computed if None.
    show_eigengenes : bool
        If True, include PC1/PC2/PC3 panels at the end.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Use the initial CPCA if provided (initial fits have pre-outlier length)
    cpca = cpca_initial if cpca_initial is not None else result.cpca_final
    time_points = cpca.circular_scale
    order = cpca.sample_order
    genes = list(cpca.core_genes_found)

    signals = genes.copy()
    if show_eigengenes:
        signals += ["PC1", "PC2", "PC3"]

    n_panels = len(signals)
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (ncols * 3.2, nrows * 2.6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    # Core matrix ordered by CPCA phase
    cm = cpca.core_matrix
    cm_vals = cm.values if hasattr(cm, "values") else cm

    for i, name in enumerate(signals):
        ax = axes_flat[i]

        # Get observed data in CPCA order
        if name in genes:
            gene_idx = genes.index(name)
            y_obs = cm_vals[gene_idx, order]
        else:
            # Eigengene
            eigenvec = {"PC1": cpca.pc1, "PC2": cpca.pc2, "PC3": cpca.pc3}
            y_obs = eigenvec[name][order]

        # Plot observed
        ax.plot(time_points, y_obs, "o",
                color="#BBBBBB", markersize=1.8, zorder=2)

        # FMM fit (initial fits match initial CPCA size)
        if name in result.fmm_fits_initial:
            fr = result.fmm_fits_initial[name]
            # Generate x-axis matching the fitted array length
            fit_x = np.linspace(0, 2 * np.pi, len(fr.fitted), endpoint=False) \
                    if len(fr.fitted) != len(time_points) else time_points
            ax.plot(fit_x, fr.fitted, "-",
                    color=_FMM_COLOUR, linewidth=1.4, zorder=4)
            r2_fmm = fr.r2
        else:
            r2_fmm = None

        # Cosinor fit
        if name in result.cosinor_fits_initial:
            cr = result.cosinor_fits_initial[name]
            fit_x = np.linspace(0, 2 * np.pi, len(cr.fitted), endpoint=False) \
                    if len(cr.fitted) != len(time_points) else time_points
            ax.plot(fit_x, cr.fitted, "--",
                    color=_COS_COLOUR, linewidth=1.0, zorder=3)
            r2_cos = cr.r2
        else:
            r2_cos = None

        # Annotations
        label = name
        if r2_fmm is not None:
            label += f"\nFMM R\u00b2={r2_fmm:.3f}"
        if r2_cos is not None:
            label += f"  Cos R\u00b2={r2_cos:.3f}"
        ax.set_title(label, fontsize=7, pad=2)

        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=_FMM_COLOUR, lw=1.4, label="FMM"),
        Line2D([0], [0], color=_COS_COLOUR, lw=1.0,
               linestyle="--", label="Cosinor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    suptitle = f"{title}  \u2014 " if title else ""
    fig.suptitle(
        f"{suptitle}Core gene fits (initial CPCA ordering)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2 — Residual strip charts with threshold lines
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_strips(
    result: OutlierRefinementResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Horizontal strip chart of standardised FMM residuals per gene.

    Each row is a core gene (+ eigengenes).  Individual samples are
    plotted as dots; those exceeding the multivariate threshold (|3|)
    or univariate threshold (|4|) are colour-coded.  Vertical dashed
    lines mark the thresholds.

    Purpose: visually identify which samples are driving outlier
    detection and in which genes.

    Parameters
    ----------
    result : OutlierRefinementResult
    title : str
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    std_res = result.std_residuals_fmm  # DataFrame (genes × samples)
    if std_res is None:
        raise ValueError("std_residuals_fmm is None — cannot plot residuals.")

    gene_names = list(std_res.index)
    n_genes = len(gene_names)
    n_samples = std_res.shape[1]

    if figsize is None:
        figsize = (10, max(4, n_genes * 0.55))

    fig, ax = plt.subplots(figsize=figsize)

    for i, gene in enumerate(gene_names):
        row = std_res.loc[gene].values
        y_pos = n_genes - 1 - i  # top gene at top

        # Classify each sample
        normal = np.abs(row) <= 3
        multi = (np.abs(row) > 3) & (np.abs(row) <= 4)
        uni = np.abs(row) > 4

        ax.scatter(row[normal], np.full(normal.sum(), y_pos),
                   s=8, color="#AAAAAA", alpha=0.5, zorder=2, linewidths=0)
        ax.scatter(row[multi], np.full(multi.sum(), y_pos),
                   s=18, color=_THRESHOLD_COLOURS[3], alpha=0.8, zorder=3,
                   linewidths=0, label="|res| > 3" if i == 0 else "")
        ax.scatter(row[uni], np.full(uni.sum(), y_pos),
                   s=24, color=_THRESHOLD_COLOURS[4], alpha=0.9, zorder=4,
                   marker="D", linewidths=0,
                   label="|res| > 4" if i == 0 else "")

    # Threshold lines
    for thresh, colour in _THRESHOLD_COLOURS.items():
        ax.axvline(thresh, color=colour, linestyle="--", linewidth=0.8,
                   alpha=0.7, zorder=1)
        ax.axvline(-thresh, color=colour, linestyle="--", linewidth=0.8,
                   alpha=0.7, zorder=1)
        ax.text(thresh + 0.15, n_genes - 0.3, f"\u00b1{thresh}",
                fontsize=6, color=colour, va="top")

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(list(reversed(gene_names)), fontsize=8)
    ax.set_xlabel("Standardised FMM residual", fontsize=9)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.axvline(0, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend — deduplicated
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(),
                  fontsize=7, loc="upper right", framealpha=0.7)

    n_out = len(result.samples_dropped)
    suptitle = f"{title}  \u2014 " if title else ""
    ax.set_title(
        f"{suptitle}Standardised FMM residuals  "
        f"({n_out} outlier sample{'s' if n_out != 1 else ''} dropped)",
        fontsize=10, pad=8,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3 — Residual heatmap
# ═══════════════════════════════════════════════════════════════════════════

def plot_residual_heatmap(
    result: OutlierRefinementResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Heatmap of |standardised FMM residuals| (genes × ordered samples).

    Colour scale uses three regions:
      - |res| < 3  : white → light blue  (normal)
      - 3 ≤ |res| < 4 : yellow → orange  (multivariate flag)
      - |res| ≥ 4  : red → dark red      (univariate flag)

    Purpose: a global view of residual patterns — useful for spotting
    sample-wide issues (vertical bands) or gene-specific issues (rows).

    Parameters
    ----------
    result : OutlierRefinementResult
    title : str
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    std_res = result.std_residuals_fmm
    if std_res is None:
        raise ValueError("std_residuals_fmm is None.")

    abs_res = np.abs(std_res.values)
    gene_names = list(std_res.index)
    n_genes, n_samples = abs_res.shape

    if figsize is None:
        figsize = (max(8, n_samples * 0.025), max(3, n_genes * 0.4))

    # Custom colourmap with threshold boundaries
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "residuals",
        [
            (0.0, "#F7FBFF"),    # near-zero: almost white
            (0.4, "#6BAED6"),    # moderate: blue
            (0.6, "#FDD49E"),    # threshold region: yellow-orange
            (0.75, "#EF6548"),   # above 3: red
            (1.0, "#A50F15"),    # extreme: dark red
        ],
    )
    vmax = max(5.0, np.percentile(abs_res, 99.5))
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=3.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(abs_res, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_names, fontsize=7)
    ax.set_xlabel("Sample index (CPCA order)", fontsize=9)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|Std. FMM residual|", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Mark thresholds on colour bar
    for thresh in [3, 4]:
        cbar.ax.axhline(thresh, color="black", linewidth=0.8, linestyle="--")
        cbar.ax.text(1.5, thresh, f"{thresh}", fontsize=6, va="center",
                     transform=cbar.ax.get_yaxis_transform())

    suptitle = f"{title}  \u2014 " if title else ""
    ax.set_title(
        f"{suptitle}|Standardised FMM residuals| heatmap",
        fontsize=10, pad=8,
    )
    fig.tight_layout()
    return fig
