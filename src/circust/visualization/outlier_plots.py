r"""
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
      Mapa de calor de \|residuos FMM estandarizados\| a través de
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


# ---------------------------------------------------------------------------
# Paleta compartida
# ---------------------------------------------------------------------------
_FMM_COLOUR = "#E41A1C"      # red
_COS_COLOUR = "#377EB8"      # blue
_OUTLIER_COLOUR = "#FF7F00"  # orange
_THRESHOLD_COLOURS = {3: "#984EA3", 4: "#E41A1C"}  # purple / red


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Core gene fits (FMM + Cosinor overlay)
# ═══════════════════════════════════════════════════════════════════════════

def plot_core_gene_fits(
    result: CPCAResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
    show_eigengenes: bool = True,
) -> Figure:
    """
    Cuadrícula de expresión de genes core con superposición de modelos FMM y Cosinor.

    Cada panel muestra la expresión observada (puntos grises) ordenada por
    fase circular, con el ajuste FMM (rojo) y el ajuste Cosinor (azul)
    superpuestos. Se anotan los valores R² de ambos modelos.

    Parámetros
    ----------
    result : CPCAResult
        Salida de ``CPCA.run()``. Los ajustes iniciales y residuos se
        almacenan en los campos de diagnostico ``fmm_fits_initial``,
        ``cosinor_fits_initial`` y ``std_residuals_fmm``.
    title : str
        Etiqueta del suptítulo.
    figsize : tuple, opcional
        Tamaño de la figura. Se calcula automáticamente si es None.
    show_eigengenes : bool
        Si es True, incluir paneles PC1/PC2/PC3 al final.

    Returns
    -------
    matplotlib.figure.Figure
    """
    time_points = result.circular_scale
    genes       = list(result.core_genes_found)

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

    # Datos observados: matriz core final ya ordenada por fase CPCA
    cm_vals = result.core_norm_final.values

    for i, name in enumerate(signals):
        ax = axes_flat[i]

        # Obtener datos observados
        if name in genes:
            gene_idx = genes.index(name)
            y_obs = cm_vals[gene_idx]
        else:
            # Eigengene (loadings finales, ya ordenados)
            eigenvec = {"PC1": result.pc1, "PC2": result.pc2, "PC3": result.pc3}
            y_obs = eigenvec[name]

        # Graficar observado
        ax.plot(time_points, y_obs, "o",
                color="#BBBBBB", markersize=1.8, zorder=2)

        # Ajuste FMM (los ajustes iniciales coinciden con el tamaño del CPCA inicial)
        if name in result.fmm_fits_initial:
            fr = result.fmm_fits_initial[name]
            # Generar eje x que coincida con la longitud del array ajustado
            fit_x = np.linspace(0, 2 * np.pi, len(fr.fitted), endpoint=False) \
                    if len(fr.fitted) != len(time_points) else time_points
            ax.plot(fit_x, fr.fitted, "-",
                    color=_FMM_COLOUR, linewidth=1.4, zorder=4)
            r2_fmm = fr.r2
        else:
            r2_fmm = None

        # Ajuste Cosinor
        if name in result.cosinor_fits_initial:
            cr = result.cosinor_fits_initial[name]
            fit_x = np.linspace(0, 2 * np.pi, len(cr.fitted), endpoint=False) \
                    if len(cr.fitted) != len(time_points) else time_points
            ax.plot(fit_x, cr.fitted, "--",
                    color=_COS_COLOUR, linewidth=1.0, zorder=3)
            r2_cos = cr.r2
        else:
            r2_cos = None

        # Anotaciones
        label = name
        if r2_fmm is not None:
            label += f"\nFMM R\u00b2={r2_fmm:.3f}"
        if r2_cos is not None:
            label += f"  Cos R\u00b2={r2_cos:.3f}"
        ax.set_title(label, fontsize=7, pad=2)

        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelsize=6, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    # Ocultar paneles no usados
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    # Leyenda compartida
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
    result: CPCAResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    r"""
    Diagrama de tiras horizontal de residuos FMM estandarizados por gen.

    Cada fila es un gen core (+ eigengenes). Las muestras individuales se
    grafican como puntos; las que superan el umbral multivariante (\|3\|)
    o el umbral univariante (\|4\|) se codifican por color. Lineas verticales
    discontinuas marcan los umbrales.

    Propósito: identificar visualmente qué muestras impulsan la detección
    de outliers y en qué genes.

    Parámetros
    ----------
    result : CPCAResult
    title : str
    figsize : tuple, opcional

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

        # Clasificar cada muestra
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

    # Líneas de umbral
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

    # Leyenda — deduplicada
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
    result: CPCAResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    r"""
    Mapa de calor de \|residuos FMM estandarizados\| (genes x muestras ordenadas).

    La escala de color usa tres regiones:

    - \|res\| < 3: blanco a azul claro (normal)
    - 3 <= \|res\| < 4: amarillo a naranja (senal multivariante)
    - \|res\| >= 4: rojo a rojo oscuro (senal univariante)

    Propósito: vista global de patrones de residuos — útil para detectar
    problemas a nivel de muestra (bandas verticales) o de gen (filas).

    Parámetros
    ----------
    result : CPCAResult
    title : str
    figsize : tuple, opcional

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

    # Mapa de color personalizado con límites de umbral
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

    # Barra de color
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("|Std. FMM residual|", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Marcar umbrales en la barra de color
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
