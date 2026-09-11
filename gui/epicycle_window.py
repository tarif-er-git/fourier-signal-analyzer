"""Dedicated PySide6 window for interactive epicycle animation."""

from collections.abc import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from visualization.epicycle import EpicycleVisualizer


class EpicycleWindow(QMainWindow):
    """Show rotating Fourier vectors without disrupting the main window."""

    def __init__(
        self,
        a0: float,
        a: Sequence[float],
        b: Sequence[float],
        omega0: float,
        num_harmonics: int,
        signal_name: str = "Current Signal",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fourier Epicycles")
        self.resize(900, 650)
        self.visualizer = EpicycleVisualizer(a0, a, b, omega0)
        self.signal_name = signal_name
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.animation = None
        self._build_layout(num_harmonics)
        self._configure_axes(num_harmonics)
        self._start_animation(num_harmonics)

    def _build_layout(self, num_harmonics: int) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
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
        self.harmonic_slider.setRange(1, self.visualizer.available_harmonics)
        self.harmonic_slider.setValue(
            max(1, min(num_harmonics, self.visualizer.available_harmonics))
        )
        self.harmonic_slider.valueChanged.connect(self._harmonics_changed)
        controls.addWidget(self.harmonic_slider, 1)
        self.harmonic_label = QLabel(str(self.harmonic_slider.value()))
        controls.addWidget(self.harmonic_label)
        layout.addLayout(controls)
        self.setCentralWidget(central)

    def _configure_axes(self, num_harmonics: int) -> None:
        """Configure a centered complex-plane view for the vector chain."""
        self.axes.clear()
        total_radius = abs(self.visualizer.dc_value) + float(
            self.visualizer.radii[:num_harmonics].sum()
        )
        limit = max(1.0, total_radius * 1.15)
        self.axes.set_xlim(-limit, limit)
        self.axes.set_ylim(-limit, limit)
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.axhline(0.0, color="0.75", linewidth=0.8)
        self.axes.axvline(0.0, color="0.75", linewidth=0.8)
        self.axes.set_xlabel("Real component / reconstructed signal")
        self.axes.set_ylabel("Rotating-vector component")
        self.axes.set_title(
            f"{self.signal_name} | Fourier Epicycles | Harmonics: {num_harmonics}"
        )
        self.axes.grid(True, alpha=0.25)

    def _start_animation(self, num_harmonics: int) -> None:
        """Stop any previous animation and create one for the selected N."""
        self.pause()
        self._configure_axes(num_harmonics)
        self.animation = self.visualizer.make_animation(
            self.figure,
            self.axes,
            num_harmonics=num_harmonics,
        )
        self.play()

    def _harmonics_changed(self, value: int) -> None:
        self.harmonic_label.setText(str(value))
        self._start_animation(value)

    def play(self) -> None:
        """Start or resume the current animation timer."""
        if self.animation is not None:
            self.animation.event_source.start()

    def pause(self) -> None:
        """Stop the current animation timer if one exists."""
        if self.animation is not None:
            self.animation.event_source.stop()

    def reset(self) -> None:
        """Reset the animation to its first frame."""
        self.pause()
        if self.animation is not None:
            self.animation._draw_next_frame(0, blit=False)
        self.canvas.draw_idle()

    def closeEvent(self, event) -> None:
        """Stop animation resources before closing the dedicated window."""
        self.pause()
        self.animation = None
        super().closeEvent(event)
