"""
circust/visualization/order_plots.py
=====================================
Visualizaciones para la etapa de ordenación preliminar (Etapa 2).

Cuatro tipos de gráficos:

  plot_circular_peaks(result)
      Gráfico polar de los tiempos de pico FMM para cada gen reloj
      principal en un reloj de 24 horas. La visualización más
      emblemática de CIRCUST — muestra el programa temporal del
      reloj circadiano.
      Equivalente en R: ``peaksFMMCoresAfter_*.png``, ``12PeaksPre_*.png``.

  plot_ordered_profiles(result, expr)
      Cuadrícula de perfiles de expresión de genes principales en el
      orden biológico final, con la curva del modelo FMM superpuesta.
      Muestra la forma de onda de cada gen tras el anclaje de ARNTL
      y la corrección de dirección.
      Equivalente en R: ``12CorePre_*.png``.

  plot_r2_comparison(result)
      Gráfico de barras horizontales del R² FMM para cada gen
      principal, codificado por color según la clasificación
      día/noche. Muestra rápidamente qué genes tienen ajustes
      rítmicos fuertes vs débiles.

  plot_day_night_diagram(result)
      Diagrama circular de sectores que divide el ciclo de 24 horas
      en día [0, π) y noche [π, 2π), con los nombres de los genes
      situados en sus posiciones de pico. Proporciona un resumen
      intuitivo de qué genes alcanzan su pico en cada fase biológica.

Todas las funciones devuelven una Figure de matplotlib. Ninguna llama a plt.show().
"""
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from circust.synchronizer import SynchronizationResult


# ---------------------------------------------------------------------------
# Constantes compartidas
# ---------------------------------------------------------------------------
_DAY_COLOUR = "#FDB863"     # naranja cálido
_NIGHT_COLOUR = "#5E4FA2"   # morado intenso
_ARNTL_COLOUR = "#E41A1C"   # rojo
_DBP_COLOUR = "#377EB8"     # azul
_FMM_COLOUR = "#E41A1C"     # rojo (línea de modelo)

# Etiquetas de tiempo biológico: π → CT0 (ARNTL), 0 → CT12
# En el marco CIRCUST: 0 = mediodía subjetivo, π = medianoche/amanecer subjetivo
_HOUR_LABELS_8 = [
    "CT12", "CT15", "CT18", "CT21",
    "CT0", "CT3", "CT6", "CT9",
]


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 1 — Diagrama de picos circulares (polar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_circular_peaks(
    result: SynchronizationResult,
    title: str = "",
    figsize: tuple[float, float] = (6, 6),
    show_ct_labels: bool = True,
) -> Figure:
    """
    Gráfico polar de los tiempos de pico FMM para cada gen reloj core.

    Los genes se sitúan alrededor de un círculo en su fase de pico estimada.
    ARNTL está anclado en π (CT0/amanecer), y DBP aparece en la primera
    mitad [0, π) si la dirección es correcta.

    Parámetros
    ----------
    result : SynchronizationResult
        Salida de ``CircularSynchronizer.run()``.
    title : str
        Etiqueta del título del gráfico.
    figsize : tuple
        Tamaño de la figura en pulgadas.
    show_ct_labels : bool
        Si es True, mostrar etiquetas de tiempo circadiano (CT0, CT6, …)
        alrededor del anillo exterior.

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    genes = result.core_genes
    peaks = result.peak_times
    n_core = len(genes)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Configuración polar: 0 en la parte superior (12 en punto), sentido horario
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Graficar cada gen como marcador radial + etiqueta
    for i, gene in enumerate(genes):
        theta = peaks[i]
        r = 0.85  # radial position for marker

        # Color: ARNTL/DBP especiales, luego día/noche
        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
        elif gene == "DBP":
            colour = _DBP_COLOUR
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
        else:
            colour = _NIGHT_COLOUR

        # Marcador
        ax.plot(theta, r, "o", color=colour, markersize=10, zorder=5)

        # Radio (línea fina desde el centro al marcador)
        ax.plot([theta, theta], [0, r], "-",
                color=colour, linewidth=0.8, alpha=0.5, zorder=2)

        # Etiqueta — ligeramente fuera del marcador
        label_r = 1.02
        ha = "left" if 0 <= theta < np.pi else "right"
        if abs(theta - np.pi / 2) < 0.3 or abs(theta - 3 * np.pi / 2) < 0.3:
            ha = "center"

        ax.text(theta, label_r, gene, fontsize=7.5, fontweight="bold",
                color=colour, ha=ha, va="center", zorder=6)

    # Cuadrícula radial: anillo único en r=0.85
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])

    # Marcas angulares: 8 divisiones (cada π/4)
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ax.set_xticks(angles_8)

    if show_ct_labels:
        ax.set_xticklabels(_HOUR_LABELS_8, fontsize=8, color="#555555")
    else:
        ax.set_xticklabels([f"{int(np.degrees(a))}\u00b0" for a in angles_8],
                           fontsize=7)

    # Sombreado día/noche
    # Día = [0, π), Noche = [π, 2π)
    day_theta = np.linspace(0, np.pi, 100)
    night_theta = np.linspace(np.pi, 2 * np.pi, 100)
    ax.fill_between(day_theta, 0, 0.65, alpha=0.06,
                    color=_DAY_COLOUR, zorder=0)
    ax.fill_between(night_theta, 0, 0.65, alpha=0.06,
                    color=_NIGHT_COLOUR, zorder=0)

    # Líneas de límite de fase en 0 y π
    ax.plot([0, 0], [0, 0.65], "-", color="#AAAAAA",
            linewidth=0.8, zorder=1)
    ax.plot([np.pi, np.pi], [0, 0.65], "-", color="#AAAAAA",
            linewidth=0.8, zorder=1)

    # Etiquetas de fase
    ax.text(np.pi / 2, 0.35, "DAY", fontsize=9, ha="center", va="center",
            color=_DAY_COLOUR, alpha=0.4, fontweight="bold", zorder=1)
    ax.text(3 * np.pi / 2, 0.35, "NIGHT", fontsize=9, ha="center",
            va="center", color=_NIGHT_COLOUR, alpha=0.4,
            fontweight="bold", zorder=1)

    # Leyenda
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ARNTL_COLOUR,
               markersize=8, label="ARNTL (anchor, \u03c0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_DBP_COLOUR,
               markersize=8, label="DBP (direction)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_DAY_COLOUR,
               markersize=8, label="Day genes [0, \u03c0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_NIGHT_COLOUR,
               markersize=8, label="Night genes [\u03c0, 2\u03c0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=7, framealpha=0.8,
              bbox_to_anchor=(1.25, -0.05))

    suptitle = f"{title}  \u2014 " if title else ""
    flip_note = " [direction flipped]" if result.direction_flipped else ""
    fig.suptitle(
        f"{suptitle}Core gene FMM peak times{flip_note}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 2 — Perfiles de expresión ordenados con superposición FMM
# ═══════════════════════════════════════════════════════════════════════════

def plot_ordered_profiles(
    result: SynchronizationResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Cuadrícula de perfiles de expresión de genes core en el orden biológico final.

    Cada panel muestra los valores de expresión (puntos grises) vs fase circular,
    con la curva del modelo FMM (línea roja) superpuesta y el tiempo de pico
    marcado con una línea vertical discontinua.

    Parámetros
    ----------
    result : SynchronizationResult
        Salida de ``CircularSynchronizer.run()``.
    title : str
        Etiqueta del titulo de la figura.
    figsize : tuple, opcional
        Tamano de la figura en pulgadas (ancho, alto).

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    from circust.fitting.fmm import FMMModel

    genes = result.core_genes
    n_core = len(genes)
    esc = result.circular_scale
    expr = result.expr_ordered  # full matrix in final order

    ncols = math.ceil(math.sqrt(n_core))
    nrows = math.ceil(n_core / ncols)

    if figsize is None:
        figsize = (ncols * 3.0, nrows * 2.4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n_core > 1 else [axes]

    # Generar curvas FMM suaves para superponer
    t_smooth = np.linspace(0, 2 * np.pi, 300)

    for i, gene in enumerate(genes):
        ax = axes_flat[i]

        if gene not in expr.index:
            ax.set_visible(False)
            continue

        y_obs = expr.loc[gene].values

        # Expresión observada
        ax.plot(esc, y_obs, "o", color="#BBBBBB", markersize=1.5, zorder=2)

        # Curva del modelo FMM a partir de los parámetros
        par = result.fmm_params[i]  # [M, A, α, β, ω]
        M, A, alpha, beta, omega = par
        if A != 0:
            fmm_smooth = M + A * np.cos(
                beta + 2 * np.arctan(omega * np.tan((t_smooth - alpha) / 2))
            )
            ax.plot(t_smooth, fmm_smooth, "-",
                    color=_FMM_COLOUR, linewidth=1.2, zorder=4)

        # Marcador de tiempo de pico
        peak = result.peak_times[i]
        ax.axvline(peak, color=_FMM_COLOUR, linestyle=":",
                   linewidth=0.8, alpha=0.6, zorder=3)

        # Color por día/noche
        if gene in result.day_genes:
            bg = _DAY_COLOUR
        elif gene in result.night_genes:
            bg = _NIGHT_COLOUR
        elif gene == "ARNTL":
            bg = _ARNTL_COLOUR
        elif gene == "DBP":
            bg = _DBP_COLOUR
        else:
            bg = "#666666"

        r2 = result.r2_fmm[i]
        ax.set_title(f"{gene}  R\u00b2={r2:.3f}", fontsize=7.5, pad=2,
                     color=bg, fontweight="bold")
        ax.set_xlim(0, 2 * np.pi)
        ax.tick_params(labelsize=5, length=2)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes_flat[n_core:]:
        ax.set_visible(False)

    fig.text(0.5, 0.01, "Circular phase (0 \u2192 2\u03c0)", ha="center",
             fontsize=9)

    suptitle = f"{title}  \u2014 " if title else ""
    fig.suptitle(
        f"{suptitle}Core gene profiles (biological order)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 3 — Gráfico de barras de comparación de R²
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_comparison(
    result: SynchronizationResult,
    title: str = "",
    figsize: tuple[float, float] = (7, 4),
) -> Figure:
    """
    Gráfico de barras horizontal del R² FMM para cada gen core.

    Las barras se codifican por color según la clasificación día/noche. Una línea
    vertical discontinua en R²=0.5 marca el umbral convencional para un ajuste
    rítmico "bueno".

    Parámetros
    ----------
    result : SynchronizationResult
    title : str
    figsize : tuple

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    genes = result.core_genes
    r2 = result.r2_fmm
    n_core = len(genes)

    # Ordenar por R² descendente para mayor legibilidad
    order = np.argsort(r2)[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    for rank, idx in enumerate(order):
        gene = genes[idx]
        val = r2[idx]

        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
        elif gene == "DBP":
            colour = _DBP_COLOUR
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
        else:
            colour = _NIGHT_COLOUR

        ax.barh(rank, val, color=colour, edgecolor="white",
                linewidth=0.5, height=0.7, zorder=3)

        # Etiqueta de valor
        ax.text(val + 0.01, rank, f"{val:.3f}", va="center",
                fontsize=7, zorder=4)

    ax.set_yticks(range(n_core))
    ax.set_yticklabels([genes[i] for i in order], fontsize=8)
    ax.invert_yaxis()

    # Línea de umbral
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8,
               zorder=2, label="R\u00b2 = 0.5 threshold")

    ax.set_xlim(0, min(1.0, r2.max() + 0.1))
    ax.set_xlabel("FMM R\u00b2", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    suptitle = f"{title}  \u2014 " if title else ""
    ax.set_title(
        f"{suptitle}FMM goodness-of-fit per core gene",
        fontsize=10, pad=8,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Gráfico 4 — Diagrama de sectores circulares día/noche
# ═══════════════════════════════════════════════════════════════════════════

def plot_day_night_diagram(
    result: SynchronizationResult,
    title: str = "",
    figsize: tuple[float, float] = (5.5, 5.5),
) -> Figure:
    """
    Diagrama de sectores circulares con mitades día [0, π) y noche [π, 2π).

    Los nombres de los genes se sitúan en su posición de fase de pico en un anillo,
    con los genes de día en naranja cálido y los de noche en morado intenso.
    ARNTL y DBP se resaltan como los genes de anclaje y dirección.

    Propósito: resumen visual rápido del programa circadiano —
    qué genes se co-expresan y la separación temporal entre
    activadores y represores.

    Parámetros
    ----------
    result : SynchronizationResult
    title : str
    figsize : tuple

    Devuelve
    --------
    matplotlib.figure.Figure
    """
    genes = result.core_genes
    peaks = result.peak_times
    n_core = len(genes)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    # Dibujar los dos semicírculos
    theta_day = np.linspace(np.pi / 2, -np.pi / 2, 100)  # mitad derecha = día [0,π)
    theta_night = np.linspace(-np.pi / 2, -3 * np.pi / 2, 100)  # izquierda = noche

    # Mapear fase CIRCUST a ángulo visual: visual = π/2 - fase
    # Así: fase=0 → arriba, fase=π/2 → derecha, fase=π → abajo
    def phase_to_xy(phase, radius=1.0):
        visual = np.pi / 2 - phase
        return radius * np.cos(visual), radius * np.sin(visual)

    # Semicírculos rellenos
    r = 1.0
    day_arc = np.linspace(0, np.pi, 100)
    night_arc = np.linspace(np.pi, 2 * np.pi, 100)

    day_xs = [0] + [phase_to_xy(t, r)[0] for t in day_arc] + [0]
    day_ys = [0] + [phase_to_xy(t, r)[1] for t in day_arc] + [0]
    night_xs = [0] + [phase_to_xy(t, r)[0] for t in night_arc] + [0]
    night_ys = [0] + [phase_to_xy(t, r)[1] for t in night_arc] + [0]

    ax.fill(day_xs, day_ys, color=_DAY_COLOUR, alpha=0.12, zorder=0)
    ax.fill(night_xs, night_ys, color=_NIGHT_COLOUR, alpha=0.12, zorder=0)

    # Círculo exterior
    circle_t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(circle_t), r * np.sin(circle_t), "-",
            color="#999999", linewidth=0.8, zorder=1)

    # Línea de diámetro (límite día/noche)
    x0, y0 = phase_to_xy(0, r)
    xpi, ypi = phase_to_xy(np.pi, r)
    ax.plot([x0, xpi], [y0, ypi], "-", color="#AAAAAA",
            linewidth=0.6, zorder=1)

    # Etiquetas de fase
    ax.text(*phase_to_xy(np.pi / 2, 0.45), "DAY", fontsize=12,
            ha="center", va="center", color=_DAY_COLOUR,
            fontweight="bold", alpha=0.35, zorder=1)
    ax.text(*phase_to_xy(3 * np.pi / 2, 0.45), "NIGHT", fontsize=12,
            ha="center", va="center", color=_NIGHT_COLOUR,
            fontweight="bold", alpha=0.35, zorder=1)

    # Posicionar genes
    label_radius = 1.22
    marker_radius = 1.0

    for i, gene in enumerate(genes):
        phase = peaks[i]
        mx, my = phase_to_xy(phase, marker_radius)
        lx, ly = phase_to_xy(phase, label_radius)

        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
            marker = "s"
            ms = 9
        elif gene == "DBP":
            colour = _DBP_COLOUR
            marker = "D"
            ms = 8
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
            marker = "o"
            ms = 7
        else:
            colour = _NIGHT_COLOUR
            marker = "o"
            ms = 7

        ax.plot(mx, my, marker, color=colour, markersize=ms, zorder=5)
        ax.plot([0, mx], [0, my], "-", color=colour,
                linewidth=0.4, alpha=0.3, zorder=2)

        # Alineación inteligente de etiquetas
        if abs(lx) < 0.2:
            ha = "center"
        elif lx > 0:
            ha = "left"
        else:
            ha = "right"

        ax.text(lx, ly, gene, fontsize=7.5, fontweight="bold",
                color=colour, ha=ha, va="center", zorder=6)

    # Marcadores de fase alrededor del borde
    for phase_val, label in [(0, "0"), (np.pi / 2, "\u03c0/2"),
                              (np.pi, "\u03c0"), (3 * np.pi / 2, "3\u03c0/2")]:
        px, py = phase_to_xy(phase_val, 1.45)
        ax.text(px, py, label, fontsize=7, ha="center", va="center",
                color="#888888")

    suptitle = f"{title}  \u2014 " if title else ""
    ax.set_title(
        f"{suptitle}Day/Night gene classification",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    return fig
