"""Reusable Qt Matplotlib canvas for 1D signal drawing.

This canvas reuses the project's established 1D mouse-drawing patterns:
  - Coordinate mapping from Qt mouse events via Matplotlib's transData inverted transform
  - Stroke point preparation and duplicate elimination via signal.sampler.prepare_signal_points
  - In-place baseline signal updating across the stroke's time range
  - Zero-centered grid lines and clear visual guides
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget

from signal.sampler import prepare_signal_points


class SignalDrawingCanvas(FigureCanvasQTAgg):
    """Embeds a single 1D drawing panel in a Qt widget.

    Allows freehand drawing of a 1D continuous/sampled signal using left-mouse
    click and drag. Updates the signal baseline in real time.
    """

    drawing_changed = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        num_samples: int = 256,
        t_range: tuple[float, float] = (-1.0, 1.0),
        amp_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.figure = Figure(figsize=(7, 4.5), constrained_layout=True)
        self.ax = self.figure.add_subplot(1, 1, 1)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._num_samples = num_samples
        self._t_min, self._t_max = t_range
        self._amp_min, self._amp_max = amp_range

        self._mouse_pressed = False
        self._is_active = True
        self._has_drawn = False
        self._title = "Draw 1D Signal"

        # Baseline sampled signal
        self._drawn_t: np.ndarray = np.linspace(
            self._t_min, self._t_max, self._num_samples
        )
        self._drawn_x: np.ndarray = np.zeros(self._num_samples, dtype=float)

        # Current mouse stroke points
        self._stroke_t: list[float] = []
        self._stroke_x: list[float] = []

        self._configure_axes()
        self._render_drawing()

    # ------------------------------------------------------------------
    # Configuration and Rendering
    # ------------------------------------------------------------------

    def _configure_axes(self) -> None:
        """Set up axis bounds, zero lines, labels, and grid."""
        self.ax.clear()
        self.ax.set_xlim(self._t_min, self._t_max)
        self.ax.set_ylim(self._amp_min, self._amp_max)
        self.ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        if self._t_min <= 0.0 <= self._t_max:
            self.ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Time (t)", fontsize=10)
        self.ax.set_ylabel("Amplitude", fontsize=10)
        self.ax.set_title(self._title, fontsize=11, fontweight="bold", pad=10)

    def _render_drawing(self) -> None:
        """Render the baseline signal curve and any active stroke points."""
        self._configure_axes()
        self.ax.plot(
            self._drawn_t,
            self._drawn_x,
            color="#1f77b4",
            linewidth=2.0,
            label="Signal",
        )
        if len(self._stroke_t) > 0:
            self.ax.scatter(
                self._stroke_t[-1],
                self._stroke_x[-1],
                color="#ff7f0e",
                s=28,
                zorder=4,
            )
        self.draw_idle()

    # ------------------------------------------------------------------
    # Public Control API
    # ------------------------------------------------------------------

    def start_drawing(
        self,
        title: str = "Draw 1D Signal",
        initial_signal: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Prepare the canvas for user drawing.

        Parameters
        ----------
        title : str
            Title displayed above the plot (e.g. "Drawing Signal x(t)").
        initial_signal : tuple of (t, x), optional
            If provided, preloads the signal for further editing.
        """
        self._title = title
        self._mouse_pressed = False
        self._stroke_t.clear()
        self._stroke_x.clear()

        if initial_signal is not None:
            t_init, x_init = initial_signal
            t_init = np.asarray(t_init, dtype=float)
            x_init = np.asarray(x_init, dtype=float)
            self._drawn_x = np.interp(
                self._drawn_t, t_init, x_init, left=0.0, right=0.0
            )
            self._has_drawn = not np.allclose(self._drawn_x, 0.0)
        else:
            self._drawn_x = np.zeros(self._num_samples, dtype=float)
            self._has_drawn = False

        self._render_drawing()
        self.setFocus()

    def finish_drawing(self) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return the completed sampled signal.

        Returns
        -------
        t : np.ndarray
            Monotonically increasing uniform time grid of length `num_samples`.
        x : np.ndarray
            Sampled amplitudes of length `num_samples`.

        Raises
        ------
        ValueError
            If no valid non-zero signal points were drawn.
        """
        has_content = self._has_drawn or len(self._stroke_t) >= 2 or not np.allclose(self._drawn_x, 0.0)
        if not has_content:
            raise ValueError(
                "Please draw a signal by clicking and dragging across the canvas."
            )
        self._mouse_pressed = False
        return self._drawn_t.copy(), self._drawn_x.copy()

    def clear_drawing(self) -> None:
        """Reset the signal curve to zero across the entire domain."""
        self._mouse_pressed = False
        self._stroke_t.clear()
        self._stroke_x.clear()
        self._drawn_x = np.zeros(self._num_samples, dtype=float)
        self._has_drawn = False
        self._render_drawing()
        self.drawing_changed.emit()

    def has_content(self) -> bool:
        """Return True if non-zero signal content has been drawn."""
        return self._has_drawn or not np.allclose(self._drawn_x, 0.0)

    # ------------------------------------------------------------------
    # Mouse Event Handling
    # ------------------------------------------------------------------

    def _event_to_signal_coords(
        self, event: QMouseEvent
    ) -> tuple[float, float] | None:
        """Convert a Qt mouse event position into (time, amplitude) data coordinates."""
        dpr = float(self.devicePixelRatioF())
        canvas_x = float(event.position().x()) * dpr
        canvas_y = float(event.position().y()) * dpr
        display_point = (canvas_x, float(self.figure.bbox.height) - canvas_y)

        if not self.ax.bbox.contains(*display_point):
            return None

        data_x, data_y = self.ax.transData.inverted().transform(display_point)
        data_x = float(np.clip(data_x, self._t_min, self._t_max))
        data_y = float(np.clip(data_y, self._amp_min, self._amp_max))
        return data_x, data_y

    def _update_drawn_signal(self) -> None:
        """Splice the current mouse stroke into the baseline signal."""
        if len(self._stroke_t) < 2:
            return
        stroke_t, stroke_x = prepare_signal_points(self._stroke_t, self._stroke_x)
        left, right = stroke_t[0], stroke_t[-1]
        covered = (self._drawn_t >= left) & (self._drawn_t <= right)
        if np.any(covered):
            self._drawn_x[covered] = np.interp(
                self._drawn_t[covered], stroke_t, stroke_x
            )
            self._has_drawn = True

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self._event_to_signal_coords(event)
            if pt is not None:
                self._stroke_t.clear()
                self._stroke_x.clear()
                self._mouse_pressed = True
                self._stroke_t.append(pt[0])
                self._stroke_x.append(pt[1])
                self._render_drawing()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mouse_pressed:
            pt = self._event_to_signal_coords(event)
            if pt is not None:
                self._stroke_t.append(pt[0])
                self._stroke_x.append(pt[1])
                self._update_drawn_signal()
                self._render_drawing()
                self.drawing_changed.emit()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mouse_pressed:
            self._mouse_pressed = False
            if len(self._stroke_t) >= 2:
                self._update_drawn_signal()
                self._render_drawing()
                self.drawing_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)
