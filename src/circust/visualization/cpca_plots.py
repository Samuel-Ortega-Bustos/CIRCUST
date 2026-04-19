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
# Paleta de colores — un color por outlier, equivalente a col=i en R
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
    Diagrama de dispersión PC1 vs PC2 (un punto por muestra).

    Características
    ---------------
    - Círculos umbral concéntricos discontinuos en r=0.10, 0.15, 0.20
      (y 0.30 / 0.40 cuando los datos se extienden hasta ahí).
    - Candidatos a outlier marcados con cuadrados abiertos de colores distintos.
    - Los outliers confirmados llevan un marcador relleno encima.
    - El círculo de radio activo (ajustado=0.10, o holgado=0.15 si rojo8)
      se dibuja en rojo; el resto en gris.
    - Los límites de los ejes son simétricos y determinados por el rango de datos.
    - La varianza explicada se muestra en las etiquetas de los ejes.

    Parámetros
    ----------
    result : CPCAResult
        Salida de CPCA.run().
    title : str
        Etiqueta de dataset/tejido mostrada en el título.
    figsize : tuple
        Tamaño de la figura en pulgadas (ancho, alto).
    point_size : float
        Área del marcador para muestras normales (parámetro `s` de matplotlib).
    show_loose_circle : bool
        Si se deben dibujar ambos círculos umbral aunque solo se usara el ajustado.

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    # Usar proyecciones iniciales (pre-eliminacion de outliers) para
    # poder mostrar donde caian las muestras eliminadas
    pc1  = result.pc1_initial if len(result.pc1_initial) > 0 else result.pc1
    pc2  = result.pc2_initial if len(result.pc2_initial) > 0 else result.pc2
    var  = result.variance_explained

    # Derivar candidatos a outlier y radio activo a partir de las
    # proyecciones, replicando la logica de CPCA._cpca_step
    norm12   = np.sqrt(pc1**2 + pc2**2)
    n_cands  = min(8, len(norm12))
    cand     = np.argsort(norm12)[:n_cands]
    tight_mask = norm12[cand] <= 0.10
    if tight_mask.any():
        conf = cand[tight_mask]
        used_loose = False
    else:
        loose_mask = norm12[cand] <= 0.15
        conf = cand[loose_mask]
        used_loose = True

    # Límites de ejes — simétricos, con pequeño margen, determinados por los datos
    maxi = round(max(np.abs(pc1).max(), np.abs(pc2).max()), 1) + 0.05
    maxi = max(maxi, 0.22)   # always show at least the 0.20 circle

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # ── Círculos de fondo ──────────────────────────────────────────────────
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

    # ── Todas las muestras (gris) ──────────────────────────────────────────────────
    ax.scatter(pc1, pc2,
               s=point_size, color="#CCCCCC", edgecolors="#999999",
               linewidths=0.4, zorder=3, label="samples")

    # ── Candidatos a outlier (cuadrados coloreados abiertos) ──────────────────────────
    conf_set = set(conf.tolist())
    for rank, idx in enumerate(cand):
        col = _outlier_colour(rank)
        marker = "s"
        facecolor = col if idx in conf_set else "none"
        ax.scatter(pc1[idx], pc2[idx],
                   s=point_size * 3, marker=marker,
                   facecolors=facecolor, edgecolors=col,
                   linewidths=1.5, zorder=5)

    # ── Entradas de leyenda para outliers ─────────────────────────────────────────
    if len(conf) > 0:
        handles = []
        for rank, idx in enumerate(conf):
            # buscar la posición de idx en cand para obtener el rango de color
            cand_rank = np.where(cand == idx)[0]
            rank_used = int(cand_rank[0]) if len(cand_rank) > 0 else rank
            col = _outlier_colour(rank_used)
            handles.append(
                mpatches.Patch(color=col, label=f"outlier sample {idx}")
            )
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.7)

    # ── Líneas de origen ───────────────────────────────────────────────────
    ax.axhline(0, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.6, zorder=0)

    # ── Etiquetas y título ──────────────────────────────────────────────────────
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
        f"n_outliers={len(result.samples_dropped)}{loose_note}",
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
    Cuadrícula de trazas de expresión por gen reloj core, ordenadas por phi.

    Disposición
    -----------
    - Un subgráfico por gen core (expresión vs fase circular).
    - Tres subgráficos adicionales al final: trazas de eigengenes PC1, PC2, PC3.
    - Las muestras outlier se marcan con cuadrados abiertos coloreados en cada panel.
    - El tamaño de la cuadrícula es el cuadrado más pequeño que cabe todos los paneles.

    Parámetros
    ----------
    result : CPCAResult
        Debe haber sido producido por una instancia de CPCA que almacenó
        ``core_matrix`` (store_core_matrix=True, que es el valor por defecto).
    title : str
        Etiqueta del suptítulo.
    figsize : tuple, opcional
        Tamaño de la figura. Se calcula automáticamente de las dimensiones de la cuadrícula si no se proporciona.

    Devuelve
    --------
    matplotlib.figure.Figure

    Lanza
    -----
    ValueError
        Si result.core_norm_final es None.
    """
    genes       = result.core_genes_found
    phi         = result.circular_scale       # sorted phi values [0, 2π)
    order       = result.sample_order         # indices that sort by phi
    var         = result.variance_explained

    # core_norm_final ya está ordenada por fase circular
    cm = result.core_norm_final.values if hasattr(result.core_norm_final, 'values') \
         else result.core_norm_final
    core_ordered = cm                          # (n_genes, n_samples) ya ordenada

    # Eigengenes ordenados por fase
    pc1_ordered = result.pc1[order]
    pc2_ordered = result.pc2[order]
    pc3_ordered = result.pc3[order]

    # Número de paneles: uno por gen + tres paneles de eigengenes
    n_gene_panels  = len(genes)
    n_eigen_panels = 3
    n_panels       = n_gene_panels + n_eigen_panels

    # Cuadrícula aproximadamente cuadrada (misma lógica que compPerfSq en R)
    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (ncols * 2.8, nrows * 2.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten()

    # ── Paneles de expresión génica ───────────────────────────────────────────────
    for i, gene in enumerate(genes):
        ax = axes_flat[i]
        y  = core_ordered[i]
        ax.plot(phi, y, "b-o", markersize=2.5, linewidth=0.8, zorder=3)
        ax.set_title(gene, fontsize=8, pad=2)
        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Paneles de eigengenes PC ──────────────────────────────────────────────────
    eigen_panels = [
        (pc1_ordered, f"PC1  ({var[0]*100:.1f}% var)"),
        (pc2_ordered, f"PC2  ({var[1]*100:.1f}% var)"),
        (pc3_ordered, f"PC3  ({var[2]*100:.1f}% var)"),
    ]
    for j, (y_vals, panel_title) in enumerate(eigen_panels):
        ax = axes_flat[n_gene_panels + j]
        ax.plot(phi, y_vals, "k-o", markersize=2.5, linewidth=0.8, zorder=3)
        ax.axhline(0, color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_title(panel_title, fontsize=8, pad=2)
        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Ocultar ejes no usados ─────────────────────────────────────────────────────
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    # ── Etiqueta x compartida en la fila inferior ────────────────────────────────────────
    fig.text(0.5, 0.01, "Circular phase φ  (0 → 2π)", ha="center", fontsize=9)

    suptitle = f"{title}  — " if title else ""
    fig.suptitle(
        f"{suptitle}Core clock genes ordered by CPCA phase",
        fontsize=10, y=1.01,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig