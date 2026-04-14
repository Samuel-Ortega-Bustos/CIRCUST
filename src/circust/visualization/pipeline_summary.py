"""
circust/visualization/pipeline_summary.py
==========================================
Figura resumen de extremo a extremo del pipeline CIRCUST (Etapas 1-2).

  plot_pipeline_summary(cpca, outlier, order)
      Figura única multipanel con cuatro vistas diagnósticas clave:

        A. Dispersión PC1 vs PC2 (etapa CPCA)   — muestra estructura circular
        B. Heatmap de residuos (etapa outlier)   — muestra calidad de muestras
        C. Diagrama circular de picos (ordenación) — muestra programa temporal
        D. Gráfico de barras R² (ordenación)     — muestra calidad del ajuste

      Diseñada para informes de tesis y presentaciones.

  plot_variance_explained(cpca)
      Gráfico de barras tipo scree de la varianza explicada por las
      primeras componentes principales, con línea acumulada superpuesta.

  plot_expression_overview(expr_norm, core_genes, circular_scale, n_top)
      Heatmap de expresión normalizada para los n genes más rítmicos,
      ordenados por fase circular. Los genes core se resaltan.
      Proporciona una visión global de la señal circadiana a nivel genómico.

Todas las funciones devuelven un matplotlib Figure. Ninguna llama a plt.show().
"""
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd

from circust.cpca import CPCAResult
from circust.synchronizer import SynchronizationResult


# ---------------------------------------------------------------------------
# Compartido
# ---------------------------------------------------------------------------
_FMM_COLOUR = "#E41A1C"
_COS_COLOUR = "#377EB8"
_DAY_COLOUR = "#FDB863"
_NIGHT_COLOUR = "#5E4FA2"
_ARNTL_COLOUR = "#E41A1C"
_DBP_COLOUR = "#377EB8"

_HOUR_LABELS_8 = [
    "CT12", "CT15", "CT18", "CT21",
    "CT0", "CT3", "CT6", "CT9",
]


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Pipeline summary (4-panel composite)
# ═══════════════════════════════════════════════════════════════════════════

def plot_pipeline_summary(
    cpca_result: CPCAResult,
    order_result: SynchronizationResult,
    title: str = "",
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """
    Resumen compuesto de cuatro paneles del pipeline CIRCUST.

    Disposición::

        ┌──────────────┬──────────────┐
        │ A. PC scatter │ B. Residuals │
        ├──────────────┼──────────────┤
        │ C. Peaks     │ D. R² bars   │
        └──────────────┴──────────────┘

    Parámetros
    ----------
    cpca_result : CPCAResult
        Salida de ``CPCA.run()`` (incluye CPCA + deteccion de outliers).
    order_result : SynchronizationResult
        Salida de CircularSynchronizer.
    title : str
        Título global de la figura.
    figsize : tuple

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Panel A: dispersión PC1 vs PC2 ──────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_pc_scatter(ax_a, cpca_result)
    ax_a.set_title("A. CPCA — PC1 vs PC2", fontsize=10, fontweight="bold",
                    pad=6)

    # ── Panel B: heatmap de residuos (compacto) ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_residual_compact(ax_b, cpca_result)
    n_out = len(cpca_result.samples_dropped)
    ax_b.set_title(
        f"B. Standardised FMM residuals  ({n_out} outliers)",
        fontsize=10, fontweight="bold", pad=6,
    )

    # ── Panel C: diagrama circular de picos ──────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0], polar=True)
    _draw_polar_peaks(ax_c, order_result)
    ax_c.set_title("C. Core gene peak times", fontsize=10,
                    fontweight="bold", pad=12)

    # ── Panel D: gráfico de barras R² ────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_r2_bars(ax_d, order_result)
    ax_d.set_title("D. FMM R\u00b2 per core gene", fontsize=10,
                    fontweight="bold", pad=6)

    suptitle = f"{title}  \u2014 " if title else ""
    fig.suptitle(
        f"{suptitle}CIRCUST Pipeline Summary (Stages 1\u20132)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# Funciones auxiliares de paneles (dibujan en un Axes existente)
# ---------------------------------------------------------------------------

def _draw_pc_scatter(ax, cpca: "CPCAResult") -> None:
    """Dispersión PC mini para el panel resumen."""
    pc1, pc2 = cpca.pc1, cpca.pc2
    var = cpca.variance_explained
    dropped = set(cpca.samples_dropped)

    # Todas las muestras (normales en gris, eliminadas en rojo)
    normal_mask  = np.array([i not in dropped for i in range(len(pc1))])
    dropped_mask = ~normal_mask

    ax.scatter(pc1[normal_mask], pc2[normal_mask], s=10, color="#CCCCCC",
               edgecolors="#999999", linewidths=0.3, zorder=3)
    if dropped_mask.any():
        ax.scatter(pc1[dropped_mask], pc2[dropped_mask], s=40, marker="x",
                   color="#E41A1C", linewidths=1.4, zorder=5,
                   label=f"{dropped_mask.sum()} outlier(s)")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    # Círculos de umbral radial (0.10 y 0.15)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, colour, lw in [(0.10, "#E41A1C", 1.2), (0.15, "#AAAAAA", 0.6)]:
        ax.plot(np.cos(theta) * r, np.sin(theta) * r,
                "--", color=colour, linewidth=lw, zorder=1)

    ax.axhline(0, color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.set_aspect("equal")
    maxi = round(max(np.abs(pc1).max(), np.abs(pc2).max()), 1) + 0.05
    ax.set_xlim(-maxi, maxi)
    ax.set_ylim(-maxi, maxi)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=8)
    ax.tick_params(labelsize=6)


def _draw_residual_compact(ax, result) -> None:
    """Diagrama de tiras de residuos compacto para el panel resumen."""
    std_res = result.std_residuals_fmm
    if std_res is None:
        ax.text(0.5, 0.5, "No residual data", ha="center", va="center",
                transform=ax.transAxes)
        return

    gene_names = list(std_res.index)
    n_genes = len(gene_names)

    for i, gene in enumerate(gene_names):
        row = std_res.loc[gene].values
        y = n_genes - 1 - i
        normal = np.abs(row) <= 3
        flagged = np.abs(row) > 3

        ax.scatter(row[normal], np.full(normal.sum(), y),
                   s=4, color="#BBBBBB", alpha=0.4, zorder=2, linewidths=0)
        if flagged.any():
            ax.scatter(row[flagged], np.full(flagged.sum(), y),
                       s=12, color="#E41A1C", alpha=0.7, zorder=3,
                       linewidths=0)

    for thresh in [3, 4]:
        col = "#984EA3" if thresh == 3 else "#E41A1C"
        ax.axvline(thresh, color=col, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.axvline(-thresh, color=col, linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(list(reversed(gene_names)), fontsize=6)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.axvline(0, color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.set_xlabel("Std. FMM residual", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_polar_peaks(ax, result) -> None:
    """Diagrama circular de picos mini para el panel resumen."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    genes = result.core_genes
    peaks = result.peak_times

    for i, gene in enumerate(genes):
        theta = peaks[i]
        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
        elif gene == "DBP":
            colour = _DBP_COLOUR
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
        else:
            colour = _NIGHT_COLOUR

        ax.plot(theta, 0.85, "o", color=colour, markersize=7, zorder=5)
        ax.plot([theta, theta], [0, 0.85], "-", color=colour,
                linewidth=0.5, alpha=0.4, zorder=2)
        ax.text(theta, 1.05, gene, fontsize=5.5, fontweight="bold",
                color=colour, ha="center", va="center", zorder=6)

    # Sombreado día/noche
    day_t = np.linspace(0, np.pi, 100)
    night_t = np.linspace(np.pi, 2 * np.pi, 100)
    ax.fill_between(day_t, 0, 0.6, alpha=0.06, color=_DAY_COLOUR)
    ax.fill_between(night_t, 0, 0.6, alpha=0.06, color=_NIGHT_COLOUR)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ax.set_xticks(angles_8)
    ax.set_xticklabels(_HOUR_LABELS_8, fontsize=5.5, color="#666666")


def _draw_r2_bars(ax, result) -> None:
    """Gráfico de barras R² mini para el panel resumen."""
    genes = result.core_genes
    r2 = result.r2_fmm
    order = np.argsort(r2)[::-1]

    for rank, idx in enumerate(order):
        gene = genes[idx]
        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
        elif gene == "DBP":
            colour = _DBP_COLOUR
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
        else:
            colour = _NIGHT_COLOUR

        ax.barh(rank, r2[idx], color=colour, edgecolor="white",
                linewidth=0.3, height=0.65, zorder=3)
        ax.text(r2[idx] + 0.008, rank, f"{r2[idx]:.3f}",
                va="center", fontsize=6, zorder=4)

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([genes[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=0.6, zorder=2)
    ax.set_xlim(0, min(1.0, r2.max() + 0.08))
    ax.set_xlabel("FMM R\u00b2", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 2 — Varianza explicada (scree plot)
# ═══════════════════════════════════════════════════════════════════════════

def plot_variance_explained(
    cpca_result: "CPCAResult",
    n_components: int = 10,
    title: str = "",
    figsize: tuple[float, float] = (6, 4),
) -> Figure:
    """
    Gráfico de barras tipo scree de la varianza explicada por cada CP.

    Muestra las primeras ``n_components`` barras con una línea acumulada
    superpuesta. PC1 y PC2 (usados por CPCA) se resaltan; se marcan los
    umbrales mínimos para PC2 (10%) y PC1+PC2 total (40%).

    Parámetros
    ----------
    cpca_result : CPCAResult
        Debe tener ``variance_explained`` con al menos 3 entradas.
    n_components : int
        Número de CPs a mostrar.
    title : str
    figsize : tuple

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    var = cpca_result.variance_explained
    n_avail = len(var)
    n_show = min(n_components, n_avail)
    var_show = var[:n_show]
    cumvar = np.cumsum(var_show)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Barras
    colours = ["#E41A1C" if i < 2 else "#AAAAAA" for i in range(n_show)]
    bars = ax1.bar(range(1, n_show + 1), var_show * 100,
                   color=colours, edgecolor="white", linewidth=0.5,
                   zorder=3)

    ax1.set_xlabel("Principal component", fontsize=9)
    ax1.set_ylabel("Variance explained (%)", fontsize=9, color="#333333")
    ax1.set_xticks(range(1, n_show + 1))
    ax1.tick_params(labelsize=7)
    ax1.spines[["top", "right"]].set_visible(False)

    # Línea acumulada en eje secundario
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_show + 1), cumvar * 100, "o-",
             color="#377EB8", markersize=5, linewidth=1.2, zorder=4)
    ax2.set_ylabel("Cumulative (%)", fontsize=9, color="#377EB8")
    ax2.tick_params(labelsize=7, colors="#377EB8")
    ax2.spines["right"].set_color("#377EB8")
    ax2.set_ylim(0, 105)

    # Líneas de umbral
    ax1.axhline(10, color="#E41A1C", linestyle=":", linewidth=0.7, alpha=0.5)
    ax1.text(n_show + 0.3, 10, "PC2 min (10%)", fontsize=6,
             color="#E41A1C", va="center")

    ax2.axhline(40, color="#377EB8", linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.text(0.6, 42, "PC1+PC2 min (40%)", fontsize=6,
             color="#377EB8", va="bottom")

    # Anotar PC1 + PC2
    for i in range(min(2, n_show)):
        ax1.text(i + 1, var_show[i] * 100 + 1,
                 f"{var_show[i]*100:.1f}%", ha="center", fontsize=7,
                 fontweight="bold", color="#E41A1C")

    suptitle = f"{title}  \u2014 " if title else ""
    ax1.set_title(
        f"{suptitle}Variance explained by principal components",
        fontsize=10, pad=8,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 3 — Heatmap de expresión global
# ═══════════════════════════════════════════════════════════════════════════

def plot_expression_overview(
    expr_ordered: pd.DataFrame,
    core_genes: list[str],
    circular_scale: np.ndarray,
    n_top: int = 50,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Heatmap de expresión normalizada para los genes más rítmicos.

    Muestra un mapa de calor (genes × muestras en orden circular) donde
    el color representa el nivel de expresión [-1, +1]. Los genes reloj
    core se resaltan con una barra de color lateral.

    Propósito: proporciona una instantánea a nivel genómico de la señal
    circadiana — los genes rítmicos muestran bandas sinusoidales limpias.

    Parámetros
    ----------
    expr_ordered : pd.DataFrame
        Matriz de expresión ya ordenada por fase circular
        (p. ej., ``SynchronizationResult.expr_ordered``).
    core_genes : list[str]
        Símbolos de genes reloj core para resaltar.
    circular_scale : np.ndarray
        Eje de tiempo circular para el eje x.
    n_top : int
        Número de genes top a mostrar (por varianza de fila).
    title : str
    figsize : tuple, opcional

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    # Seleccionar genes top por varianza (más rítmicos)
    row_var = expr_ordered.var(axis=1)
    top_genes = row_var.nlargest(n_top).index.tolist()

    # Asegurar que todos los genes core están incluidos
    for cg in core_genes:
        if cg in expr_ordered.index and cg not in top_genes:
            top_genes.append(cg)

    # Ordenar: genes core primero, luego por posición de pico (argmax)
    def sort_key(gene):
        is_core = gene in core_genes
        peak_pos = np.argmax(expr_ordered.loc[gene].values)
        return (0 if is_core else 1, peak_pos)

    top_genes.sort(key=sort_key)

    mat = expr_ordered.loc[top_genes].values
    n_genes_show = len(top_genes)

    if figsize is None:
        figsize = (10, max(4, n_genes_show * 0.15))

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [0.03, 1], "wspace": 0.02},
    )

    # Barra de color lateral — genes core resaltados
    core_mask = np.array([g in core_genes for g in top_genes])
    side_colours = np.where(core_mask, 1.0, 0.0)
    ax_bar.imshow(side_colours.reshape(-1, 1), aspect="auto",
                  cmap=mcolors.ListedColormap(["#F0F0F0", "#E41A1C"]),
                  interpolation="nearest")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Genes", fontsize=9)

    # Heatmap principal
    cmap = plt.cm.RdBu_r
    im = ax_heat.imshow(mat, aspect="auto", cmap=cmap,
                        vmin=-1, vmax=1, interpolation="nearest")

    # Eje x: mostrar valores de fase
    n_samples = mat.shape[1]
    n_ticks = 8
    tick_positions = np.linspace(0, n_samples - 1, n_ticks, dtype=int)
    tick_labels = [f"{circular_scale[p]:.1f}" for p in tick_positions]
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels(tick_labels, fontsize=6)
    ax_heat.set_xlabel("Circular phase (rad)", fontsize=9)

    # Eje y: nombres de genes (mostrar solo un subconjunto si hay demasiados)
    if n_genes_show <= 30:
        ax_heat.set_yticks(range(n_genes_show))
        ax_heat.set_yticklabels(top_genes, fontsize=5)
        # Colorear etiquetas de genes core
        for i, label in enumerate(ax_heat.get_yticklabels()):
            if top_genes[i] in core_genes:
                label.set_color("#E41A1C")
                label.set_fontweight("bold")
    else:
        # Mostrar solo genes core
        core_positions = [i for i, g in enumerate(top_genes) if g in core_genes]
        ax_heat.set_yticks(core_positions)
        ax_heat.set_yticklabels(
            [top_genes[i] for i in core_positions], fontsize=6,
            color="#E41A1C", fontweight="bold",
        )

    # Barra de color
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.02)
    cbar.set_label("Normalised expression", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Leyenda para la barra lateral
    legend_elements = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="#E41A1C", markersize=8,
               label="Core clock gene"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="#F0F0F0", markeredgecolor="#CCCCCC",
               markersize=8, label="Other gene"),
    ]
    ax_heat.legend(handles=legend_elements, loc="upper right",
                   fontsize=6, framealpha=0.8)

    suptitle = f"{title}  \u2014 " if title else ""
    fig.suptitle(
        f"{suptitle}Expression heatmap (top {n_top} genes, circular order)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    return fig
