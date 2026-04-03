"""
circust/visualization/cpca_plots.py
============================
Visualizaciones para la etapa CPCA del pipeline CIRCUST.

Dos gráficos, equivalentes a la salida de obtainCPCA12/13 (printing=TRUE) en R:

  plot_pc_scatter(result)
      Dispersión PC1 vs PC2 con resaltado de outliers y círculos de umbral.
      Equivalente a la primera ventana x11() en obtainCPCA12/13 de R.

  plot_gene_panels(result)
      Cuadrícula de trazas de expresión de genes reloj ordenadas por fase circular,
      más paneles de eigengenes PC1/PC2/PC3 al final.
      Equivalente a la segunda ventana x11() en obtainCPCA12/13 de R.

Ambas funciones devuelven una Figure de matplotlib para que el llamador pueda
guardarlas, mostrarlas o integrarlas según necesite. Ninguna función llama a
plt.show() — eso es responsabilidad del llamador.

Uso
---
    from circust.cpca import CPCA
    from circust.plots.cpca_plots import plot_pc_scatter, plot_gene_panels

    result = CPCA().run(expr_norm)

    fig1 = plot_pc_scatter(result, title="BA46 glutamatergic")
    fig1.savefig("cpca_scatter.png", dpi=150, bbox_inches="tight")

    fig2 = plot_gene_panels(result)
    fig2.savefig("cpca_genes.png", dpi=150, bbox_inches="tight")
"""
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Colour palette — one colour per outlier, matching R's col=i convention
# ---------------------------------------------------------------------------
_OUTLIER_COLOURS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#66C2A5",  # turquesa (reemplaza gris)
]


def _outlier_colour(i: int) -> str:
    return _OUTLIER_COLOURS[i % len(_OUTLIER_COLOURS)]


# ---------------------------------------------------------------------------
# Plot 1 — PC scatter
# ---------------------------------------------------------------------------

def plot_pc_scatter(
    result,
    title: str = "",
    figsize: tuple[float, float] = (7, 7),
    point_size: float = 18,
    show_loose_circle: bool = True,
) -> Figure:
    """
    Scatter of PC1 vs PC2 loadings (one point per sample).

    Features
    --------
    - Concentric dashed threshold circles at r=0.10, 0.15, 0.20
      (and 0.30 / 0.40 when the data extends that far).
    - Outlier candidates marked with open squares in distinct colours.
    - Confirmed outliers get a filled marker on top.
    - The active radius circle (tight=0.10, or loose=0.15 if rojo8)
      is drawn in red; all others are grey.
    - Axis limits are symmetric and driven by the data range.
    - Variance explained shown in the axis labels.

    Parameters
    ----------
    result : CPCAResult
        Output of CPCA.run().
    title : str
        Dataset / tissue label shown in the plot title.
    figsize : tuple
        Figure size in inches (width, height).
    point_size : float
        Marker area for regular samples (matplotlib `s` parameter).
    show_loose_circle : bool
        Whether to always draw both threshold circles even when only
        the tight one was used.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pc1  = result.pc1
    pc2  = result.pc2
    var  = result.variance_explained
    cand = result.outlier_candidate_idx   # indices in original sample space
    conf = result.outlier_idx
    used_loose = result.used_loose_radius

    # Axis limits — symmetric, slightly padded, driven by data
    maxi = round(max(np.abs(pc1).max(), np.abs(pc2).max()), 1) + 0.05
    maxi = max(maxi, 0.22)   # always show at least the 0.20 circle

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # ── Background circles ──────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 360)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    radii        = [0.10, 0.15, 0.20]
    active_r     = 0.15 if used_loose else 0.10

    if maxi >= 0.30:
        radii.append(0.30)
    if maxi >= 0.40:
        radii.append(0.40)

    for r in radii:
        colour = "#E41A1C" if r == active_r else "#AAAAAA"
        lw     = 1.4       if r == active_r else 0.8
        ax.plot(cos_t * r, sin_t * r,
                linestyle="--", color=colour, linewidth=lw, zorder=1)
        # small radius label at top of each circle
        ax.text(0, r + 0.005, f"{r:.2f}",
                ha="center", va="bottom",
                fontsize=6, color=colour, zorder=2)

    # ── All samples (grey) ──────────────────────────────────────────────────
    ax.scatter(pc1, pc2,
               s=point_size, color="#CCCCCC", edgecolors="#999999",
               linewidths=0.4, zorder=3, label="samples")

    # ── Outlier candidates (open coloured squares) ──────────────────────────
    conf_set = set(conf.tolist())
    for rank, idx in enumerate(cand):
        col = _outlier_colour(rank)
        marker = "s"
        facecolor = col if idx in conf_set else "none"
        ax.scatter(pc1[idx], pc2[idx],
                   s=point_size * 3, marker=marker,
                   facecolors=facecolor, edgecolors=col,
                   linewidths=1.5, zorder=5)

    # ── Legend entries for outliers ─────────────────────────────────────────
    if len(conf) > 0:
        handles = []
        for rank, idx in enumerate(conf):
            # find position of idx in cand to get the colour rank
            cand_rank = np.where(cand == idx)[0]
            rank_used = int(cand_rank[0]) if len(cand_rank) > 0 else rank
            col = _outlier_colour(rank_used)
            handles.append(
                mpatches.Patch(color=col, label=f"outlier sample {idx}")
            )
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.7)

    # ── Origin crosshairs ───────────────────────────────────────────────────
    ax.axhline(0, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.6, zorder=0)

    # ── Labels & title ──────────────────────────────────────────────────────
    ax.set_xlim(-maxi, maxi)
    ax.set_ylim(-maxi, maxi)
    ax.set_xlabel(
        f"PC1  ({var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(
        f"PC2  ({var[1]*100:.1f}% var)", fontsize=11)

    title_str = f"{title}. " if title else ""
    loose_note = "  [loose radius]" if used_loose else ""
    ax.set_title(
        f"{title_str}PC1 vs PC2 — core clock genes\n"
        f"n_outliers={result.n_outliers}{loose_note}",
        fontsize=11,
    )

    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Gene expression panels ordered by circular phase
# ---------------------------------------------------------------------------

def plot_gene_panels(
    result,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Grid of expression traces for each core clock gene, ordered by phi.

    Layout
    ------
    - One subplot per core gene (expression vs circular phase).
    - Three additional subplots at the end: PC1, PC2, PC3 eigengene traces.
    - Outlier samples marked with coloured open squares in every panel.
    - Grid size is the smallest square that fits all panels.

    Parameters
    ----------
    result : CPCAResult
        Must have been produced by a CPCA instance that stored
        ``core_matrix`` (set store_core_matrix=True, which is the default).
    title : str
        Suptitle label.
    figsize : tuple, optional
        Figure size. Auto-computed from grid dimensions if not given.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    AttributeError
        If result.core_matrix is None (CPCA was run with store_core_matrix=False).
    """
    if result.core_matrix is None:
        raise AttributeError(
            "result.core_matrix is None. "
            "Run CPCA with store_core_matrix=True (the default) to enable gene panels."
        )

    genes        = result.core_genes_found
    core_matrix  = result.core_matrix          # shape: (n_genes, n_samples)
    order        = result.sample_order         # indices that sort by phi
    phi          = result.circular_scale       # sorted phi values [0, 2π)
    outs_pos     = result.outlier_positions_in_order   # positions WITHIN order
    n_outliers   = result.n_outliers
    pc1, pc2, pc3 = result.pc1, result.pc2, result.pc3
    var           = result.variance_explained

    # Number of panels: one per gene + three eigengene panels
    n_gene_panels  = len(genes)
    n_eigen_panels = 3
    n_panels       = n_gene_panels + n_eigen_panels

    # Square-ish grid (same logic as R's compPerfSq)
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (ncols * 2.8, nrows * 2.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten()

    # Ordered expression and eigengenes
    # core_matrix may be a DataFrame (gene index × sample columns) — use .iloc
    cm = core_matrix.values if hasattr(core_matrix, 'values') else core_matrix
    core_ordered = cm[:, order]          # (n_genes, n_samples) sorted
    pc1_ordered  = pc1[order]
    pc2_ordered  = pc2[order]
    pc3_ordered  = pc3[order]

    # Outlier colours indexed by their rank in outlier_positions_in_order
    # outs_pos contains positions within order — same rank logic as R's col=i
    outlier_phi_positions = outs_pos  # positions in [0..n_samples-1] in ordered space

    def _add_outlier_marks(ax, y_values):
        """Mark outlier positions in a panel with coloured squares."""
        for rank, pos in enumerate(outlier_phi_positions):
            if 0 <= pos < len(phi):
                col = _outlier_colour(rank)
                ax.scatter(phi[pos], y_values[pos],
                           s=60, marker="s",
                           facecolors="none", edgecolors=col,
                           linewidths=1.5, zorder=5)

    # ── Gene expression panels ───────────────────────────────────────────────
    for i, gene in enumerate(genes):
        ax = axes_flat[i]
        y  = core_ordered[i]
        ax.plot(phi, y, "b-o", markersize=2.5, linewidth=0.8, zorder=3)
        _add_outlier_marks(ax, y)
        ax.set_title(gene, fontsize=8, pad=2)
        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── PC eigengene panels ──────────────────────────────────────────────────
    eigen_panels = [
        (pc1_ordered, f"PC1  ({var[0]*100:.1f}% var)"),
        (pc2_ordered, f"PC2  ({var[1]*100:.1f}% var)"),
        (pc3_ordered, f"PC3  ({var[2]*100:.1f}% var)"),
    ]
    for j, (y_vals, panel_title) in enumerate(eigen_panels):
        ax = axes_flat[n_gene_panels + j]
        ax.plot(phi, y_vals, "k-o", markersize=2.5, linewidth=0.8, zorder=3)
        ax.axhline(0, color="#DDDDDD", linewidth=0.6, zorder=0)
        _add_outlier_marks(ax, y_vals)
        ax.set_title(panel_title, fontsize=8, pad=2)
        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Hide unused axes ─────────────────────────────────────────────────────
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    # ── Shared x-label on bottom row ────────────────────────────────────────
    fig.text(0.5, 0.01, "Circular phase φ  (0 → 2π)", ha="center", fontsize=9)

    suptitle = f"{title}  — " if title else ""
    fig.suptitle(
        f"{suptitle}Core clock genes ordered by CPCA phase",
        fontsize=10, y=1.01,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig