"""
circust/visualization/order_plots.py
=====================================
Visualisations for the preliminary ordering stage (Stage 2).

Four plot types:

  plot_circular_peaks(result)
      Polar plot of FMM peak times for each core gene on a 24-hour
      clock face.  The most iconic CIRCUST visualisation — shows the
      temporal programme of the circadian clock.
      R equivalent: ``peaksFMMCoresAfter_*.png``, ``12PeaksPre_*.png``.

  plot_ordered_profiles(result, expr)
      Grid of core-gene expression profiles in the final biological
      order, with the FMM model curve overlaid.  Shows each gene's
      waveform after ARNTL anchoring and direction correction.
      R equivalent: ``12CorePre_*.png``.

  plot_r2_comparison(result)
      Horizontal bar chart of FMM R² for each core gene, colour-coded
      by day/night classification.  Quickly shows which genes have
      strong vs weak rhythmic fits.

  plot_day_night_diagram(result)
      Circular sector diagram partitioning the 24-hour cycle into
      day [0, π) and night [π, 2π), with gene names placed at their
      peak positions.  Provides an intuitive summary of which genes
      peak in each biological phase.

All functions return a matplotlib Figure.  None call plt.show().
"""
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from circust.preliminary_order import PreliminaryOrderResult


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_DAY_COLOUR = "#FDB863"     # warm orange
_NIGHT_COLOUR = "#5E4FA2"   # deep purple
_ARNTL_COLOUR = "#E41A1C"   # red
_DBP_COLOUR = "#377EB8"     # blue
_FMM_COLOUR = "#E41A1C"     # red (model line)

# Biological-time labels: π → CT0 (ARNTL), 0 → CT12
# In the CIRCUST frame: 0 = subjective noon, π = subjective midnight/dawn
_HOUR_LABELS_8 = [
    "CT12", "CT15", "CT18", "CT21",
    "CT0", "CT3", "CT6", "CT9",
]


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Circular peak diagram (polar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_circular_peaks(
    result: PreliminaryOrderResult,
    title: str = "",
    figsize: tuple[float, float] = (6, 6),
    show_ct_labels: bool = True,
) -> Figure:
    """
    Polar plot of FMM peak times for each core clock gene.

    Genes are placed around a circle at their estimated peak phase.
    ARNTL is anchored at π (CT0/dawn), and DBP appears in the first
    half [0, π) if the direction is correct.

    Parameters
    ----------
    result : PreliminaryOrderResult
        Output of ``PreliminaryOrderEstimator.run()``.
    title : str
        Plot title label.
    figsize : tuple
        Figure size in inches.
    show_ct_labels : bool
        If True, show circadian-time labels (CT0, CT6, …) around the
        outer ring.

    Returns
    -------
    matplotlib.figure.Figure
    """
    genes = result.core_genes
    peaks = result.peak_times
    n_core = len(genes)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Polar config: 0 at top (12 o'clock), clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Plot each gene as a radial marker + label
    for i, gene in enumerate(genes):
        theta = peaks[i]
        r = 0.85  # radial position for marker

        # Colour: ARNTL/DBP special, then day/night
        if gene == "ARNTL":
            colour = _ARNTL_COLOUR
        elif gene == "DBP":
            colour = _DBP_COLOUR
        elif gene in result.day_genes:
            colour = _DAY_COLOUR
        else:
            colour = _NIGHT_COLOUR

        # Marker
        ax.plot(theta, r, "o", color=colour, markersize=10, zorder=5)

        # Radial spoke (thin line from centre to marker)
        ax.plot([theta, theta], [0, r], "-",
                color=colour, linewidth=0.8, alpha=0.5, zorder=2)

        # Label — slightly outside the marker
        label_r = 1.02
        ha = "left" if 0 <= theta < np.pi else "right"
        if abs(theta - np.pi / 2) < 0.3 or abs(theta - 3 * np.pi / 2) < 0.3:
            ha = "center"

        ax.text(theta, label_r, gene, fontsize=7.5, fontweight="bold",
                color=colour, ha=ha, va="center", zorder=6)

    # Radial grid: single ring at r=0.85
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])

    # Angular ticks: 8 divisions (every π/4)
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ax.set_xticks(angles_8)

    if show_ct_labels:
        ax.set_xticklabels(_HOUR_LABELS_8, fontsize=8, color="#555555")
    else:
        ax.set_xticklabels([f"{int(np.degrees(a))}\u00b0" for a in angles_8],
                           fontsize=7)

    # Day/night shading
    # Day = [0, π), Night = [π, 2π)
    day_theta = np.linspace(0, np.pi, 100)
    night_theta = np.linspace(np.pi, 2 * np.pi, 100)
    ax.fill_between(day_theta, 0, 0.65, alpha=0.06,
                    color=_DAY_COLOUR, zorder=0)
    ax.fill_between(night_theta, 0, 0.65, alpha=0.06,
                    color=_NIGHT_COLOUR, zorder=0)

    # Phase boundary lines at 0 and π
    ax.plot([0, 0], [0, 0.65], "-", color="#AAAAAA",
            linewidth=0.8, zorder=1)
    ax.plot([np.pi, np.pi], [0, 0.65], "-", color="#AAAAAA",
            linewidth=0.8, zorder=1)

    # Phase labels
    ax.text(np.pi / 2, 0.35, "DAY", fontsize=9, ha="center", va="center",
            color=_DAY_COLOUR, alpha=0.4, fontweight="bold", zorder=1)
    ax.text(3 * np.pi / 2, 0.35, "NIGHT", fontsize=9, ha="center",
            va="center", color=_NIGHT_COLOUR, alpha=0.4,
            fontweight="bold", zorder=1)

    # Legend
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
# Plot 2 — Ordered expression profiles with FMM overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_ordered_profiles(
    result: PreliminaryOrderResult,
    title: str = "",
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """
    Grid of core-gene expression profiles in the final biological order.

    Each panel shows expression values (grey dots) vs circular phase,
    with the FMM model curve (red line) overlaid and the peak time
    marked with a vertical dashed line.

    Parameters
    ----------
    result : PreliminaryOrderResult
        Output of ``PreliminaryOrderEstimator.run()``.
    title : str
    figsize : tuple, optional

    Returns
    -------
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

    # Generate smooth FMM curves for overlay
    t_smooth = np.linspace(0, 2 * np.pi, 300)

    for i, gene in enumerate(genes):
        ax = axes_flat[i]

        if gene not in expr.index:
            ax.set_visible(False)
            continue

        y_obs = expr.loc[gene].values

        # Observed expression
        ax.plot(esc, y_obs, "o", color="#BBBBBB", markersize=1.5, zorder=2)

        # FMM model curve from parameters
        par = result.fmm_params[i]  # [M, A, α, β, ω]
        M, A, alpha, beta, omega = par
        if A != 0:
            fmm_smooth = M + A * np.cos(
                beta + 2 * np.arctan(omega * np.tan((t_smooth - alpha) / 2))
            )
            ax.plot(t_smooth, fmm_smooth, "-",
                    color=_FMM_COLOUR, linewidth=1.2, zorder=4)

        # Peak time marker
        peak = result.peak_times[i]
        ax.axvline(peak, color=_FMM_COLOUR, linestyle=":",
                   linewidth=0.8, alpha=0.6, zorder=3)

        # Colour by day/night
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
# Plot 3 — R² comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_comparison(
    result: PreliminaryOrderResult,
    title: str = "",
    figsize: tuple[float, float] = (7, 4),
) -> Figure:
    """
    Horizontal bar chart of FMM R² for each core gene.

    Bars are colour-coded by day/night classification.  A vertical
    dashed line at R²=0.5 marks the conventional threshold for a
    "good" rhythmic fit.

    Parameters
    ----------
    result : PreliminaryOrderResult
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    genes = result.core_genes
    r2 = result.r2_fmm
    n_core = len(genes)

    # Sort by R² descending for readability
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

        # Value label
        ax.text(val + 0.01, rank, f"{val:.3f}", va="center",
                fontsize=7, zorder=4)

    ax.set_yticks(range(n_core))
    ax.set_yticklabels([genes[i] for i in order], fontsize=8)
    ax.invert_yaxis()

    # Threshold line
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
# Plot 4 — Day/night circular sector diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_day_night_diagram(
    result: PreliminaryOrderResult,
    title: str = "",
    figsize: tuple[float, float] = (5.5, 5.5),
) -> Figure:
    """
    Circular sector diagram with day [0, π) and night [π, 2π) halves.

    Gene names are placed at their peak phase position on a ring,
    with day genes in warm orange and night genes in deep purple.
    ARNTL and DBP are highlighted as the anchor and direction genes.

    Purpose: quick visual summary of the circadian programme —
    which genes are co-expressed and the temporal separation between
    activators and repressors.

    Parameters
    ----------
    result : PreliminaryOrderResult
    title : str
    figsize : tuple

    Returns
    -------
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

    # Draw the two semicircles
    theta_day = np.linspace(np.pi / 2, -np.pi / 2, 100)  # right half = day [0,π)
    theta_night = np.linspace(-np.pi / 2, -3 * np.pi / 2, 100)  # left = night

    # Map CIRCUST phase to visual angle: visual = π/2 - phase
    # So phase=0 → top, phase=π/2 → right, phase=π → bottom
    def phase_to_xy(phase, radius=1.0):
        visual = np.pi / 2 - phase
        return radius * np.cos(visual), radius * np.sin(visual)

    # Filled semicircles
    r = 1.0
    day_arc = np.linspace(0, np.pi, 100)
    night_arc = np.linspace(np.pi, 2 * np.pi, 100)

    day_xs = [0] + [phase_to_xy(t, r)[0] for t in day_arc] + [0]
    day_ys = [0] + [phase_to_xy(t, r)[1] for t in day_arc] + [0]
    night_xs = [0] + [phase_to_xy(t, r)[0] for t in night_arc] + [0]
    night_ys = [0] + [phase_to_xy(t, r)[1] for t in night_arc] + [0]

    ax.fill(day_xs, day_ys, color=_DAY_COLOUR, alpha=0.12, zorder=0)
    ax.fill(night_xs, night_ys, color=_NIGHT_COLOUR, alpha=0.12, zorder=0)

    # Outer circle
    circle_t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(circle_t), r * np.sin(circle_t), "-",
            color="#999999", linewidth=0.8, zorder=1)

    # Diameter line (day/night boundary)
    x0, y0 = phase_to_xy(0, r)
    xpi, ypi = phase_to_xy(np.pi, r)
    ax.plot([x0, xpi], [y0, ypi], "-", color="#AAAAAA",
            linewidth=0.6, zorder=1)

    # Phase labels
    ax.text(*phase_to_xy(np.pi / 2, 0.45), "DAY", fontsize=12,
            ha="center", va="center", color=_DAY_COLOUR,
            fontweight="bold", alpha=0.35, zorder=1)
    ax.text(*phase_to_xy(3 * np.pi / 2, 0.45), "NIGHT", fontsize=12,
            ha="center", va="center", color=_NIGHT_COLOUR,
            fontweight="bold", alpha=0.35, zorder=1)

    # Place genes
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

        # Smart label alignment
        if abs(lx) < 0.2:
            ha = "center"
        elif lx > 0:
            ha = "left"
        else:
            ha = "right"

        ax.text(lx, ly, gene, fontsize=7.5, fontweight="bold",
                color=colour, ha=ha, va="center", zorder=6)

    # Phase markers around the edge
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
