"""Matplotlib drawing helpers for the convolution simulation animation.

This module is GUI-free. All functions accept Matplotlib Axes objects and
return nothing (they modify the axes in place). The GUI layer (convolution_window.py)
owns the Figure, FigureCanvas, and QTimer; this module only handles
what goes on the canvas.

Three-panel layout
------------------
Panel 0 (top)    x(τ)             Fixed signal — drawn once.
Panel 1 (middle) h(t−τ)           Reversed and shifted kernel — updated each frame.
Panel 2 (bottom) y(t) = x * h     Progressively built output — updated each frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------
# Initial plot setup — called once when signals are prepared
# ---------------------------------------------------------------------------

def setup_convolution_figure(
    figure: Figure,
    axes: list[Axes],
    tau: np.ndarray,
    x_resampled: np.ndarray,
    h_resampled: np.ndarray,
    output_t: np.ndarray,
    x_name: str = "x(t)",
    h_name: str = "h(t)",
) -> dict[str, object]:
    """Configure all three panels and return a dict of live artist handles.

    The returned handles dictionary contains every artist that changes
    during the animation. The static backgrounds (x plot, axis labels, grid)
    are drawn here and never updated.

    Parameters
    ----------
    figure
        Matplotlib Figure containing the three axes.
    axes
        List of exactly three Axes objects (top, middle, bottom).
    tau, x_resampled, h_resampled
        Common tau grid and resampled signal values.
    output_t
        Output time axis (2*N-1 samples).
    x_name, h_name
        Labels used in plot titles and legends.

    Returns
    -------
    dict
        Keys: ``"h_line"``, ``"overlap_fill"``, ``"output_line"``,
        ``"output_marker"``, ``"info_text"``.
    """
    ax_x, ax_h, ax_y = axes[0], axes[1], axes[2]

    # ----- Panel 0: static x(τ) -----
    ax_x.clear()
    ax_x.plot(tau, x_resampled, color="tab:blue", linewidth=1.8, label=f"{x_name}")
    ax_x.fill_between(tau, x_resampled, alpha=0.15, color="tab:blue")
    ax_x.axhline(0.0, color="0.6", linewidth=0.8)
    ax_x.axvline(0.0, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
    y_max = float(np.max(np.abs(x_resampled)))
    _set_limits(ax_x, tau, y_max or 1.0)
    ax_x.set_title(f"Fixed signal   x(τ)", fontsize=10, loc="left")
    ax_x.set_ylabel("Amplitude")
    ax_x.legend(loc="upper right", fontsize=8)
    ax_x.grid(True, alpha=0.25)

    # ----- Panel 1: h(t−τ) — updated each frame -----
    ax_h.clear()
    ax_h.axhline(0.0, color="0.6", linewidth=0.8)
    ax_h.axvline(0.0, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
    # Ghost of original h(τ) for reference
    ax_h.plot(
        tau, h_resampled,
        color="0.7", linewidth=1.0, linestyle=":", label=f"Original {h_name}",
        alpha=0.6,
    )
    # Live shifted signal — starts as empty
    (h_line,) = ax_h.plot([], [], color="tab:orange", linewidth=1.8,
                           label=f"h(t−τ)  [reversed & shifted]")
    # Also show x faintly in middle panel for overlap comparison
    ax_h.plot(tau, x_resampled, color="tab:blue", linewidth=1.0,
              alpha=0.30, linestyle="-", label="x(τ) reference")
    # Overlap fill — will be updated each frame
    overlap_fill = ax_h.fill_between(
        tau, np.zeros_like(tau), np.zeros_like(tau),
        color="tab:green", alpha=0.30, label="Overlap x(τ)·h(t−τ)",
    )
    y_max_h = float(np.max(np.abs(h_resampled))) or 1.0
    combined_max = max(y_max or 1.0, y_max_h)
    _set_limits(ax_h, tau, combined_max)
    ax_h.set_title("Reversed & shifted kernel   h(t−τ)", fontsize=10, loc="left")
    ax_h.set_ylabel("Amplitude")
    ax_h.legend(loc="upper right", fontsize=8)
    ax_h.grid(True, alpha=0.25)

    # ----- Panel 2: progressive y(t) output -----
    ax_y.clear()
    ax_y.axhline(0.0, color="0.6", linewidth=0.8)
    ax_y.axvline(0.0, color="0.6", linewidth=0.8, linestyle="--", alpha=0.6)
    (output_line,) = ax_y.plot([], [], color="tab:purple", linewidth=1.8,
                                label="y(t) = x(t) * h(t)")
    (output_marker,) = ax_y.plot([], [], "o", color="tab:red", markersize=7,
                                  zorder=5, label="Current y(t)")
    _set_output_limits(ax_y, output_t)
    ax_y.set_title("Convolution output   y(t) = x(t) ∗ h(t)", fontsize=10, loc="left")
    ax_y.set_xlabel("t")
    ax_y.set_ylabel("y(t)")
    ax_y.legend(loc="upper right", fontsize=8)
    ax_y.grid(True, alpha=0.25)

    figure.tight_layout(h_pad=1.2)

    return {
        "h_line": h_line,
        "overlap_fill": overlap_fill,
        "output_line": output_line,
        "output_marker": output_marker,
    }


# ---------------------------------------------------------------------------
# Per-frame update — called on every animation tick
# ---------------------------------------------------------------------------

def update_convolution_frame(
    axes: list[Axes],
    handles: dict[str, object],
    tau: np.ndarray,
    h_shifted_values: np.ndarray,
    x_resampled: np.ndarray,
    output_t_so_far: np.ndarray,
    y_so_far: np.ndarray,
    current_t: float,
    current_y: float,
    canvas,
) -> None:
    """Update the middle and bottom panels for a single animation frame.

    Parameters
    ----------
    axes
        The three Axes objects.
    handles
        Artist handles from :func:`setup_convolution_figure`.
    tau
        Common tau grid.
    h_shifted_values
        h(current_t − τ) evaluated on tau. Shape matches tau.
    x_resampled
        x(τ) values. Used to compute the overlap product for shading.
    output_t_so_far
        Output time values computed so far (up to and including current frame).
    y_so_far
        Convolution output values computed so far.
    current_t
        The current shift value t (scalar).
    current_y
        The convolution value y(current_t) (scalar).
    canvas
        FigureCanvasQTAgg — ``draw_idle()`` is called at the end.
    """
    ax_h = axes[1]
    ax_y = axes[2]

    h_line: Line2D = handles["h_line"]
    output_line: Line2D = handles["output_line"]
    output_marker: Line2D = handles["output_marker"]

    # ----- Middle panel: update h(t−τ) -----
    h_line.set_data(tau, h_shifted_values)

    # Remove old overlap fill and redraw
    old_fill = handles["overlap_fill"]
    try:
        old_fill.remove()
    except ValueError:
        pass

    overlap_product = x_resampled * h_shifted_values
    # Shade the overlap between the two signals
    new_fill = ax_h.fill_between(
        tau,
        0.0,
        overlap_product,
        where=np.abs(overlap_product) > 1e-10,
        color="tab:green",
        alpha=0.35,
        label="_nolegend_",
    )
    handles["overlap_fill"] = new_fill

    # Vertical marker for current shift t on middle panel
    _update_shift_marker(ax_h, current_t)

    # ----- Bottom panel: progressively build y(t) -----
    if output_t_so_far.size > 0:
        output_line.set_data(output_t_so_far, y_so_far)
    output_marker.set_data([current_t], [current_y])

    # Adjust y-axis limits dynamically as the output grows
    if y_so_far.size > 1:
        y_span = float(np.max(np.abs(y_so_far))) or 1.0
        margin = y_span * 0.15
        ax_y.set_ylim(-y_span - margin, y_span + margin)

    canvas.draw_idle()


def draw_final_result(
    axes: list[Axes],
    handles: dict[str, object],
    output_t: np.ndarray,
    y_reference: np.ndarray,
    canvas,
) -> None:
    """Render the complete convolution output after the animation finishes."""
    output_line: Line2D = handles["output_line"]
    output_marker: Line2D = handles["output_marker"]
    output_line.set_data(output_t, y_reference)
    output_marker.set_data([], [])
    y_span = float(np.max(np.abs(y_reference))) or 1.0
    margin = y_span * 0.15
    axes[2].set_ylim(-y_span - margin, y_span + margin)
    canvas.draw_idle()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _set_limits(ax: Axes, tau: np.ndarray, y_max: float) -> None:
    """Set symmetric axis limits with a small margin."""
    tau_span = float(tau[-1] - tau[0])
    margin_x = tau_span * 0.06
    margin_y = y_max * 0.20
    ax.set_xlim(float(tau[0]) - margin_x, float(tau[-1]) + margin_x)
    ax.set_ylim(-y_max - margin_y, y_max + margin_y)


def _set_output_limits(ax: Axes, output_t: np.ndarray) -> None:
    """Set x-axis limits for the output panel; y-axis will auto-expand."""
    span = float(output_t[-1] - output_t[0])
    margin = span * 0.04
    ax.set_xlim(float(output_t[0]) - margin, float(output_t[-1]) + margin)
    ax.set_ylim(-1.5, 1.5)  # placeholder; will be updated dynamically


def _update_shift_marker(ax: Axes, t_shift: float) -> None:
    """Draw or update a vertical dashed line at the current shift position."""
    # Remove any previous shift marker lines tagged with our label
    for line in list(ax.lines):
        if getattr(line, "_conv_shift_marker", False):
            line.remove()
    vl = ax.axvline(t_shift, color="tab:red", linewidth=1.2, linestyle="--", alpha=0.7)
    vl._conv_shift_marker = True  # type: ignore[attr-defined]
