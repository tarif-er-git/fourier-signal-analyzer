"""Interactive Matplotlib window for 2D Fourier epicycles."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fourier.curve2d_analysis import CurveFourierResult
from visualization.curve2d_epicycle import Curve2DEpicycle


class Curve2DEpicycleWindow(QMainWindow):
    """Show rotating planar Fourier vectors and their endpoint trace."""

    FRAME_INTERVAL_MS = 30
    PARAMETER_STEP = 1.0 / 240.0

    def __init__(
        self,
        coefficients: CurveFourierResult,
        harmonic_count: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Fourier Epicycles")
        self.resize(950, 700)
        self.model = Curve2DEpicycle(coefficients)
        self.coefficients = coefficients
        self.parameter = 0.0
        self._trace: list[complex] = []
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._build_layout(harmonic_count)
        self._configure_plot(harmonic_count)
        self._create_artists(harmonic_count)
        self._draw_frame()
        self.play()

    def _build_layout(self, harmonic_count: int) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(QLabel("Harmonics N:"))
        self.harmonic_slider = QSlider(Qt.Orientation.Horizontal)
        self.harmonic_slider.setRange(0, self.model.available_harmonics)
        self.harmonic_slider.setValue(
            max(0, min(int(harmonic_count), self.model.available_harmonics))
        )
        self.harmonic_slider.valueChanged.connect(self._harmonics_changed)
        controls.addWidget(self.harmonic_slider, 1)
        self.harmonic_label = QLabel(str(self.harmonic_slider.value()))
        controls.addWidget(self.harmonic_label)
        layout.addLayout(controls)
        self.info_label = QLabel(
            "Each rotating vector represents a Fourier harmonic. "
            "The final endpoint traces the reconstructed curve."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.setCentralWidget(central)

    def _configure_plot(self, harmonic_count: int) -> None:
        self.axes.clear()
        source = np.column_stack((self.coefficients.x_samples, self.coefficients.y_samples))
        self.axes.plot(
            source[:, 0], source[:, 1],
            color="0.55", linewidth=1.0, alpha=0.55, label="Original curve",
        )
        span = max(
            float(np.ptp(source[:, 0])), float(np.ptp(source[:, 1])), 1e-6
        )
        center = np.mean(source, axis=0)
        limit = span * 0.65
        self.axes.set_xlim(center[0] - limit, center[0] + limit)
        self.axes.set_ylim(center[1] - limit, center[1] + limit)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_title(f"2D Fourier Epicycles (N={harmonic_count})")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(loc="best")

    def _create_artists(self, harmonic_count: int) -> None:
        count = len(self.model.selected_harmonics(harmonic_count))
        self._circles = [Circle((0.0, 0.0), 0.0, fill=False, color="tab:blue", alpha=0.35) for _ in range(count)]
        for circle in self._circles:
            self.axes.add_patch(circle)
        self._vectors = [self.axes.plot([], [], color="tab:blue", linewidth=1.1)[0] for _ in range(count)]
        self._endpoint_marker, = self.axes.plot([], [], "o", color="tab:red", markersize=5)
        self._trace_line, = self.axes.plot([], [], color="tab:red", linewidth=1.3, label="Endpoint trace")
        self.axes.legend(loc="best")

    def _draw_frame(self) -> None:
        harmonic_count = self.harmonic_slider.value()
        frame = self.model.frame(self.parameter, harmonic_count)
        for circle, vector, center, endpoint, radius in zip(
            self._circles,
            self._vectors,
            frame.centers,
            frame.endpoints,
            frame.radii,
        ):
            circle.center = (center[0], center[1])
            circle.radius = float(radius)
            vector.set_data([center[0], endpoint[0]], [center[1], endpoint[1]])
        self._endpoint_marker.set_data([frame.endpoint.real], [frame.endpoint.imag])
        if self._trace:
            trace = np.asarray(self._trace, dtype=complex)
            self._trace_line.set_data(trace.real, trace.imag)
        else:
            self._trace_line.set_data([], [])
        self.canvas.draw_idle()

    def _advance(self) -> None:
        self.parameter = (self.parameter + self.PARAMETER_STEP) % 1.0
        self._trace.append(self.model.endpoint(self.parameter, self.harmonic_slider.value()))
        self._draw_frame()

    def _harmonics_changed(self, value: int) -> None:
        self.harmonic_label.setText(str(value))
        self.parameter = 0.0
        self._trace.clear()
        self._rebuild_for_harmonics(value)
        self._draw_frame()

    def _rebuild_for_harmonics(self, harmonic_count: int) -> None:
        for circle in self._circles:
            circle.remove()
        for vector in self._vectors:
            vector.remove()
        self._endpoint_marker.remove()
        self._trace_line.remove()
        self._configure_plot(harmonic_count)
        self._create_artists(harmonic_count)

    def play(self) -> None:
        """Start or resume the Qt animation timer."""
        self._timer.start()

    def pause(self) -> None:
        """Pause the animation timer."""
        self._timer.stop()

    def reset(self) -> None:
        """Pause, return to parameter zero, and clear the endpoint trace."""
        self.pause()
        self.parameter = 0.0
        self._trace.clear()
        self._draw_frame()

    def closeEvent(self, event) -> None:
        """Stop the timer before closing the window."""
        self.pause()
        super().closeEvent(event)
