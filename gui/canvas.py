"""Qt canvas wrapper for plots and one-dimensional mouse drawing."""

from collections.abc import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget

from gui.drawing import pixel_to_signal_coordinates
from signal.sampler import prepare_signal_points
from visualization.error_plot import plot_error
from visualization.error_plot import plot_error_vs_harmonics
from visualization.signal_plot import plot_signal
from visualization.spectrum_plot import plot_fourier_vs_fft_spectrum


class SignalCanvas(FigureCanvasQTAgg):
    """Embed time-domain and convergence Matplotlib axes in a Qt widget."""

    drawing_finished = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(8, 10), constrained_layout=True)
        self.axes = self.figure.subplots(5, 1, sharex=False)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drawing_active = False
        self._mouse_pressed = False
        self._drawn_t: list[float] = []
        self._drawn_x: list[float] = []
        self._stroke_t: list[float] = []
        self._stroke_x: list[float] = []
        self.show_empty_state()

    def _clear_axes(self) -> None:
        """Clear all panels before drawing the next application state."""
        for axis in self.axes:
            axis.clear()

    def show_empty_state(self) -> None:
        """Display explanatory empty panels before a signal is generated."""
        self._clear_axes()
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
        self._mouse_pressed = False
        self._drawn_t.clear()
        self._drawn_x.clear()
        self._stroke_t.clear()
        self._stroke_x.clear()
        self._drawn_t.extend(np.linspace(0.0, 1.0, 1001))
        self._drawn_x.extend(np.zeros(1001))
        self._configure_drawing_axes()
        self.setFocus()
        self.draw_idle()

    def finish_drawing(self) -> tuple[np.ndarray, np.ndarray]:
        """Stop drawing and return the accumulated editable signal."""
        has_edited_signal = len(self._stroke_t) >= 2 or (
            len(self._drawn_x) >= 2
            and not np.allclose(np.asarray(self._drawn_x, dtype=float), 0.0)
        )
        if not has_edited_signal:
            raise ValueError("Please draw a signal by dragging across the canvas.")
        self._drawing_active = False
        self._mouse_pressed = False
        return np.asarray(self._drawn_t, dtype=float), np.asarray(
            self._drawn_x, dtype=float
        )

    def clear_drawing(self) -> None:
        """Discard raw drawing points and return to the empty plot state."""
        self._drawing_active = False
        self._mouse_pressed = False
        self._drawn_t.clear()
        self._drawn_x.clear()
        self._stroke_t.clear()
        self._stroke_x.clear()
        self.show_empty_state()

    def _configure_drawing_axes(self) -> None:
        """Set the first panel's time and amplitude drawing bounds."""
        self._clear_axes()
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
        canvas_x = float(event.position().x())
        canvas_y = float(event.position().y())
        figure_height = float(self.figure.bbox.height)
        display_x = canvas_x
        display_y = figure_height - canvas_y
        data_x, data_y = self.axes[0].transData.inverted().transform(
            (display_x, display_y)
        )
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
        if self._drawing_active and event.button() == Qt.MouseButton.LeftButton:
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
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Collect and render points while the left button is held."""
        if self._drawing_active and self._mouse_pressed:
            point = self._event_to_signal(event)
            if point is not None:
                self._stroke_t.append(point[0])
                self._stroke_x.append(point[1])
                self._update_drawn_signal()
                self._render_live_drawing()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the current edit stroke without processing the signal."""
        if (
            self._drawing_active
            and event.button() == Qt.MouseButton.LeftButton
            and self._mouse_pressed
        ):
            self._mouse_pressed = False
            if len(self._stroke_t) >= 2:
                self._update_drawn_signal()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def show_original(self, t: Sequence[float], original: Sequence[float]) -> None:
        """Display a newly generated original signal and clear old results."""
        self._clear_axes()
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
