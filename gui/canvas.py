"""Qt canvas wrapper for plots and one-dimensional mouse drawing."""

from collections.abc import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget

from gui.drawing import pixel_to_signal_coordinates
from signal.curve2d import Curve2D
from signal.sampler import prepare_signal_points
from visualization.error_plot import plot_error
from visualization.error_plot import plot_error_vs_harmonics
from visualization.signal_plot import plot_signal
from visualization.spectrum_plot import plot_fourier_vs_fft_spectrum


class SignalCanvas(FigureCanvasQTAgg):
    """Embed time-domain and convergence Matplotlib axes in a Qt widget."""

    drawing_finished = Signal(object, object)
    curve_finished = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(8, 10), constrained_layout=True)
        self.axes = self.figure.subplots(5, 1, sharex=False)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drawing_active = False
        self._drawing_mode: str | None = None
        self._mouse_pressed = False
        self._drawn_t: list[float] = []
        self._drawn_x: list[float] = []
        self._stroke_t: list[float] = []
        self._stroke_x: list[float] = []
        self._curve_points: list[tuple[float, float]] = []
        self._curve: Curve2D | None = None
        self.show_empty_state()

    def _clear_axes(self) -> None:
        """Clear all panels before drawing the next application state."""
        for axis in self.axes:
            axis.clear()

    def show_empty_state(self) -> None:
        """Display explanatory empty panels before a signal is generated."""
        self._clear_axes()
        for axis in self.axes:
            axis.set_visible(True)
        titles = (
            "Original Signal",
            "Reconstructed Signal",
            "Reconstruction Error",
            "Error vs Number of Harmonics",
            "Fourier Series vs FFT",
        )
        for axis, title in zip(self.axes, titles):
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.set_ylabel("Amplitude")
        self.axes[2].set_xlabel("Time")
        self.axes[2].set_ylabel("Error")
        self.axes[3].set_xlabel("Number of Harmonics (N)")
        self.axes[3].set_ylabel("Mean Squared Error")
        self.axes[4].set_xlabel("Harmonic Number")
        self.axes[4].set_ylabel("Magnitude")
        self.draw_idle()

    def start_drawing(self) -> None:
        """Activate the first panel as a bounded one-period drawing area."""
        self._drawing_active = True
        self._drawing_mode = "signal"
        self._mouse_pressed = False
        self._drawn_t.clear()
        self._drawn_x.clear()
        self._stroke_t.clear()
        self._stroke_x.clear()
        self._drawn_t.extend(np.linspace(0.0, 1.0, 1001))
        self._drawn_x.extend(np.zeros(1001))
        self._configure_drawing_axes()
        self.setFocus()
        self.draw()

    def finish_drawing(self) -> tuple[np.ndarray, np.ndarray]:
        """Stop drawing and return the accumulated editable signal."""
        has_edited_signal = len(self._stroke_t) >= 2 or (
            len(self._drawn_x) >= 2
            and not np.allclose(np.asarray(self._drawn_x, dtype=float), 0.0)
        )
        if not has_edited_signal:
            raise ValueError("Please draw a signal by dragging across the canvas.")
        self._drawing_active = False
        self._drawing_mode = None
        self._mouse_pressed = False
        return np.asarray(self._drawn_t, dtype=float), np.asarray(
            self._drawn_x, dtype=float
        )

    def clear_drawing(self) -> None:
        """Discard raw drawing points and return to the empty plot state."""
        self._drawing_active = False
        self._drawing_mode = None
        self._mouse_pressed = False
        self._drawn_t.clear()
        self._drawn_x.clear()
        self._stroke_t.clear()
        self._stroke_x.clear()
        self._curve_points.clear()
        self._curve = None
        self.show_empty_state()

    def start_curve_drawing(self) -> None:
        """Activate the dedicated equal-aspect 2D curve drawing area."""
        self._drawing_active = True
        self._drawing_mode = "curve"
        self._mouse_pressed = False
        self._curve_points.clear()
        self._curve = None
        self._configure_curve_axes()
        self.setFocus()
        self.draw()

    def finish_curve_drawing(self) -> Curve2D:
        """Close and return the current curve without duplicating its first point."""
        if self._curve is None or not self._curve.is_valid:
            raise ValueError("Draw a 2D curve with at least three distinct points.")
        self._drawing_active = False
        self._drawing_mode = None
        self._mouse_pressed = False
        self._render_curve(self._curve)
        return self._curve

    def clear_curve(self) -> None:
        """Clear the 2D curve and leave the curve drawing area active."""
        self._curve_points.clear()
        self._curve = None
        if self._drawing_mode == "curve":
            self._configure_curve_axes()
            self.draw_idle()

    def _configure_curve_axes(self) -> None:
        """Configure the first panel as an equal-aspect 2D drawing area."""
        self._clear_axes()
        axis = self.axes[0]
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("Draw 2D Closed Curve", pad=12)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.text(
            0.5,
            1.02,
            "Press and drag with the left mouse button, then click Finish / Close Curve.",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        axis.grid(True, alpha=0.3)
        for other_axis in self.axes[1:]:
            other_axis.set_visible(False)

    def _event_to_axes_data(
        self, event: QMouseEvent, axis: object
    ) -> tuple[float, float] | None:
        """Map a Qt event through the rendered Matplotlib axis transform."""
        canvas_x = float(event.position().x()) * float(self.devicePixelRatioF())
        canvas_y = float(event.position().y()) * float(self.devicePixelRatioF())
        display_point = (
            canvas_x,
            float(self.figure.bbox.height) - canvas_y,
        )
        if not axis.bbox.contains(*display_point):
            return None
        data_x, data_y = axis.transData.inverted().transform(display_point)
        point = np.asarray([data_x, data_y], dtype=float)
        if not np.all(np.isfinite(point)):
            return None
        return float(data_x), float(data_y)

    def _event_to_curve_point(self, event: QMouseEvent) -> tuple[float, float] | None:
        """Convert a mouse event to finite coordinates on the 2D plot."""
        point = self._event_to_axes_data(event, self.axes[0])
        if point is None:
            return None
        data_x, data_y = point
        if not (-1.0 <= data_x <= 1.0 and -1.0 <= data_y <= 1.0):
            return None
        return float(data_x), float(data_y)

    def _render_curve(self, curve: Curve2D | None = None) -> None:
        """Render an open stroke or its logically closed representation."""
        self._configure_curve_axes()
        axis = self.axes[0]
        if curve is not None and curve.cleaned_points.shape[0] > 0:
            points = curve.closed_points
            axis.plot(points[:, 0], points[:, 1], color="tab:blue", linewidth=1.5)
            axis.scatter(
                [points[0, 0]], [points[0, 1]], color="tab:orange", s=24, zorder=3
            )
            min_x, max_x, min_y, max_y = curve.bounding_box or (0, 0, 0, 0)
            axis.text(
                0.02,
                0.02,
                f"Points: {curve.point_count} | Status: {'Closed' if curve.is_closed else 'Needs more points'}\n"
                f"X: {min_x:.3g} to {max_x:.3g} | Y: {min_y:.3g} to {max_y:.3g}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        elif self._curve_points:
            points = np.asarray(self._curve_points, dtype=float)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color="tab:blue",
                linewidth=1.5,
                marker=".",
                markersize=2,
            )
            axis.scatter(
                [points[-1, 0]], [points[-1, 1]],
                color="tab:orange", s=24, zorder=3,
            )
        self.draw()

    def _configure_drawing_axes(self) -> None:
        """Set the first panel's time and amplitude drawing bounds."""
        self._clear_axes()
        for other_axis in self.axes:
            other_axis.set_visible(True)
        axis = self.axes[0]
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_title(
            "Custom Drawing: One Period (t = 0 to 1)",
            y=1.08,
            pad=8,
        )
        axis.set_xlabel("Time")
        axis.set_ylabel("Amplitude")
        axis.text(
            0.5, 1.015,
            "Drag across the line as many times as needed, then click Finish Drawing.",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        axis.grid(True, alpha=0.3)
        for other_axis, title in zip(
            self.axes[1:],
            (
                "Reconstructed Signal",
                "Reconstruction Error",
                "Error vs Number of Harmonics",
                "Fourier Series vs FFT",
            ),
        ):
            other_axis.set_title(title)
            other_axis.grid(True, alpha=0.3)

    def _event_to_signal(self, event: QMouseEvent) -> tuple[float, float] | None:
        """Convert a Qt mouse event to bounded data coordinates."""
        point = self._event_to_axes_data(event, self.axes[0])
        if point is None:
            return None
        data_x, data_y = point
        if not 0.0 <= data_x <= 1.0:
            return None
        return pixel_to_signal_coordinates(
            data_x,
            0.5 - np.clip(data_y, -1.0, 1.0) / 2.0,
            1.0,
            1.0,
        )

    def _render_live_drawing(self) -> None:
        """Render the editable baseline and current stroke immediately."""
        self._configure_drawing_axes()
        if self._drawn_t:
            self.axes[0].plot(self._drawn_t, self._drawn_x, color="tab:blue", linewidth=1.2)
        self.draw_idle()

    def _update_drawn_signal(self) -> None:
        """Apply the current stroke to the accumulated signal by x-position."""
        if len(self._stroke_t) < 2:
            return
        stroke_t, stroke_x = prepare_signal_points(self._stroke_t, self._stroke_x)
        baseline_t = np.asarray(self._drawn_t, dtype=float)
        baseline_x = np.asarray(self._drawn_x, dtype=float)
        left, right = stroke_t[0], stroke_t[-1]
        covered = (baseline_t >= left) & (baseline_t <= right)
        baseline_x[covered] = np.interp(
            baseline_t[covered], stroke_t, stroke_x
        )
        self._drawn_x[:] = baseline_x.tolist()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start collecting points when drawing mode receives left-click."""
        if (
            self._drawing_active
            and self._drawing_mode == "signal"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self._event_to_signal(event)
            if point is not None:
                self._stroke_t.clear()
                self._stroke_x.clear()
                self._mouse_pressed = True
                self._stroke_t.append(point[0])
                self._stroke_x.append(point[1])
                self._render_live_drawing()
                event.accept()
                return
        if (
            self._drawing_active
            and self._drawing_mode == "curve"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self._event_to_curve_point(event)
            if point is not None:
                self._curve_points = [point]
                self._curve = None
                self._mouse_pressed = True
                self._render_curve()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Collect and render points while the left button is held."""
        if (
            self._drawing_active
            and self._drawing_mode == "signal"
            and self._mouse_pressed
        ):
            point = self._event_to_signal(event)
            if point is not None:
                self._stroke_t.append(point[0])
                self._stroke_x.append(point[1])
                self._update_drawn_signal()
                self._render_live_drawing()
                event.accept()
                return
        if (
            self._drawing_active
            and self._drawing_mode == "curve"
            and self._mouse_pressed
        ):
            point = self._event_to_curve_point(event)
            if point is not None and point != self._curve_points[-1]:
                self._curve_points.append(point)
                self._render_curve()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the current edit stroke without processing the signal."""
        if (
            self._drawing_active
            and self._drawing_mode == "signal"
            and event.button() == Qt.MouseButton.LeftButton
            and self._mouse_pressed
        ):
            self._mouse_pressed = False
            if len(self._stroke_t) >= 2:
                self._update_drawn_signal()
            event.accept()
            return
        if (
            self._drawing_active
            and self._drawing_mode == "curve"
            and event.button() == Qt.MouseButton.LeftButton
            and self._mouse_pressed
        ):
            self._mouse_pressed = False
            self._curve = Curve2D.from_points(self._curve_points)
            self._render_curve(self._curve)
            self.curve_finished.emit(self._curve)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def show_original(self, t: Sequence[float], original: Sequence[float]) -> None:
        """Display a newly generated original signal and clear old results."""
        self._clear_axes()
        for axis in self.axes:
            axis.set_visible(True)
        plot_signal(t, original, ax=self.axes[0], title="Original Signal")
        self.axes[1].set_title("Reconstructed Signal")
        self.axes[2].set_title("Reconstruction Error")
        self.axes[3].set_title("Error vs Number of Harmonics")
        self.axes[4].set_title("Fourier Series vs FFT")
        for axis in self.axes[1:]:
            axis.grid(True, alpha=0.3)
        self.axes[0].set_xlabel("")
        self.axes[1].set_xlabel("")
        self.axes[2].set_xlabel("Time")
        self.axes[2].set_ylabel("Error")
        self.axes[3].set_xlabel("Number of Harmonics (N)")
        self.axes[3].set_ylabel("Mean Squared Error")
        self.axes[4].set_xlabel("Harmonic Number")
        self.axes[4].set_ylabel("Magnitude")
        self.draw_idle()

    def show_reconstruction(
        self,
        t: Sequence[float],
        original: Sequence[float],
        reconstructed: Sequence[float],
        error: Sequence[float],
        num_harmonics: int,
        harmonic_counts: Sequence[int] | None = None,
        mse_values: Sequence[float] | None = None,
    ) -> None:
        """Display signals and convergence error on the Matplotlib axes."""
        self._clear_axes()
        for axis in self.axes:
            axis.set_visible(True)
        plot_signal(t, original, ax=self.axes[0], title="Original Signal")
        plot_signal(
            t,
            reconstructed,
            ax=self.axes[1],
            title=f"Reconstructed Signal (N={num_harmonics})",
        )
        plot_error(t, error, ax=self.axes[2])
        if harmonic_counts is not None and mse_values is not None:
            plot_error_vs_harmonics(
                harmonic_counts, mse_values, ax=self.axes[3]
            )
        else:
            self.axes[3].set_title("Error vs Number of Harmonics")
            self.axes[3].set_xlabel("Number of Harmonics (N)")
            self.axes[3].set_ylabel("Mean Squared Error")
            self.axes[3].grid(True, alpha=0.3)
        self.axes[4].set_title("Fourier Series vs FFT")
        self.axes[4].set_xlabel("Harmonic Number")
        self.axes[4].set_ylabel("Magnitude")
        self.axes[4].grid(True, alpha=0.3)
        self.axes[0].set_xlabel("")
        self.axes[1].set_xlabel("")
        self.axes[2].set_xlabel("Time")
        self.draw_idle()

    def show_fft_comparison(
        self,
        harmonics: Sequence[int],
        fourier_magnitudes: Sequence[float],
        fft_magnitudes: Sequence[float],
    ) -> None:
        """Display a precomputed Fourier Series versus FFT spectrum."""
        self.axes[4].clear()
        plot_fourier_vs_fft_spectrum(
            harmonics,
            fourier_magnitudes,
            fft_magnitudes,
            ax=self.axes[4],
        )
        self.draw_idle()
