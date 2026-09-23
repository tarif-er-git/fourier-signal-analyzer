"""Reusable controls for the Signal Sketch and Decompose window."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class ControlPanel(QWidget):
    """Signal selection, reconstruction controls, and metric display."""

    generate_requested = Signal()
    reconstruct_requested = Signal()
    reset_requested = Signal()
    draw_custom_requested = Signal()
    finish_drawing_requested = Signal()
    draw_curve_requested = Signal()
    finish_curve_requested = Signal()
    clear_curve_requested = Signal()
    analyze_curve_requested = Signal()
    curve_epicycle_requested = Signal()
    curve_spectrum_requested = Signal()
    curve_error_requested = Signal()
    curve_harmonic_changed = Signal(int)
    harmonic_changed = Signal(int)
    fft_comparison_requested = Signal()
    epicycle_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signal_selector = QComboBox()
        self.signal_selector.addItem("Sine", "sine")
        self.signal_selector.addItem("Square", "square")
        self.signal_selector.addItem("Triangle", "triangle")
        self.signal_selector.addItem("Sawtooth", "sawtooth")

        self.generate_button = QPushButton("Generate")
        self.clear_button = QPushButton("Clear / Reset")
        self.draw_button = QPushButton("Draw Custom Signal")
        self.finish_drawing_button = QPushButton("Finish Drawing")
        self.draw_curve_button = QPushButton("Draw 2D Curve")
        self.finish_curve_button = QPushButton("Finish / Close Curve")
        self.clear_curve_button = QPushButton("Clear Curve")
        self.analyze_curve_button = QPushButton("Analyze Curve")
        self.curve_epicycle_button = QPushButton("2D Epicycle View")
        self.curve_spectrum_button = QPushButton("2D Harmonic Spectrum")
        self.curve_error_button = QPushButton("2D Error & Convergence")
        self.curve_analysis_label = QLabel("2D curve analysis: --")
        self.curve_analysis_label.setWordWrap(True)
        self.curve_harmonic_slider = QSlider(Qt.Orientation.Horizontal)
        self.curve_harmonic_slider.setRange(0, 0)
        self.curve_harmonic_slider.setValue(0)
        self.curve_harmonic_value = QLabel("0")
        self.curve_harmonic_value.setMinimumWidth(24)
        self.curve_harmonic_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.curve_stats_label = QLabel("2D Fourier Reconstruction\n--")
        self.curve_stats_label.setWordWrap(True)
        self.reconstruct_button = QPushButton("Reconstruct")
        self.compare_fft_button = QPushButton("Compare with FFT")
        self.epicycle_button = QPushButton("Show Epicycles")
        self.current_signal_label = QLabel("Current signal: None")
        self.current_signal_label.setWordWrap(True)

        self.harmonic_slider = QSlider(Qt.Orientation.Horizontal)
        self.harmonic_slider.setRange(1, 50)
        self.harmonic_slider.setValue(10)
        self.harmonic_value = QLabel("10")
        self.harmonic_value.setMinimumWidth(24)
        self.harmonic_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.harmonic_slider.valueChanged.connect(self._harmonic_value_changed)

        self.metric_labels: dict[str, QLabel] = {}
        for metric_name in ("MSE", "RMSE", "MAE", "Max Error"):
            self.metric_labels[metric_name] = QLabel("--")
        self.gibbs_status = QLabel("Gibbs analysis: --")
        self.gibbs_status.setWordWrap(True)
        self.gibbs_labels: dict[str, QLabel] = {}
        for metric_name in ("Overshoot", "Undershoot", "Theoretical"):
            self.gibbs_labels[metric_name] = QLabel("--")
        self.fft_timing_labels = {
            "Fourier": QLabel("--"),
            "FFT": QLabel("--"),
        }

        self._build_layout()
        self.generate_button.clicked.connect(self.generate_requested)
        self.reconstruct_button.clicked.connect(self.reconstruct_requested)
        self.clear_button.clicked.connect(self.reset_requested)
        self.draw_button.clicked.connect(self.draw_custom_requested)
        self.finish_drawing_button.clicked.connect(self.finish_drawing_requested)
        self.draw_curve_button.clicked.connect(self.draw_curve_requested)
        self.finish_curve_button.clicked.connect(self.finish_curve_requested)
        self.clear_curve_button.clicked.connect(self.clear_curve_requested)
        self.analyze_curve_button.clicked.connect(self.analyze_curve_requested)
        self.curve_epicycle_button.clicked.connect(self.curve_epicycle_requested)
        self.curve_spectrum_button.clicked.connect(self.curve_spectrum_requested)
        self.curve_error_button.clicked.connect(self.curve_error_requested)
        self.curve_harmonic_slider.valueChanged.connect(
            self._curve_harmonic_value_changed
        )
        self.compare_fft_button.clicked.connect(self.fft_comparison_requested)
        self.epicycle_button.clicked.connect(self.epicycle_requested)
        self._signal_available = False
        self.set_drawing_active(False)
        self.set_curve_active(False)
        self.set_curve_analysis_available(False)
        self.set_signal_available(False)
        self.set_reconstruction_available(False)

    @property
    def selected_signal(self) -> str:
        """Return the internal name of the selected preset signal."""
        return str(self.signal_selector.currentData())

    @property
    def num_harmonics(self) -> int:
        """Return the selected number of harmonics."""
        return self.harmonic_slider.value()

    def set_metrics(
        self, mse: float, rmse: float, mae: float, max_error: float
    ) -> None:
        """Display computed metric values supplied by the controller."""
        values = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Max Error": max_error,
        }
        for name, value in values.items():
            self.metric_labels[name].setText(f"{value:.6g}")

    def reset_metrics(self) -> None:
        """Restore placeholder metric labels."""
        for label in self.metric_labels.values():
            label.setText("--")

    def set_gibbs_result(
        self,
        status: str,
        overshoot: float | None = None,
        undershoot: float | None = None,
        theoretical: float | None = None,
    ) -> None:
        """Display Gibbs values supplied by the analysis controller."""
        self.gibbs_status.setText(f"Gibbs analysis: {status}")
        values = {
            "Overshoot": overshoot,
            "Undershoot": undershoot,
            "Theoretical": theoretical,
        }
        for name, value in values.items():
            self.gibbs_labels[name].setText("--" if value is None else f"{value:.6g}")

    def reset_gibbs(self) -> None:
        """Restore the Gibbs placeholder section."""
        self.set_gibbs_result("--")

    def set_fft_timing(self, fourier_seconds: float, fft_seconds: float) -> None:
        """Display timing values supplied by the FFT comparison controller."""
        self.fft_timing_labels["Fourier"].setText(f"{fourier_seconds * 1000:.3f} ms")
        self.fft_timing_labels["FFT"].setText(f"{fft_seconds * 1000:.3f} ms")

    def reset_fft_timing(self) -> None:
        """Restore FFT timing placeholders."""
        for label in self.fft_timing_labels.values():
            label.setText("--")

    def set_drawing_active(self, active: bool) -> None:
        """Enable only the controls relevant while collecting mouse points."""
        self.draw_button.setEnabled(not active)
        self.finish_drawing_button.setEnabled(active)
        self.generate_button.setEnabled(not active)
        self.reconstruct_button.setEnabled(not active and getattr(self, "_signal_available", False))

    def set_curve_active(self, active: bool) -> None:
        """Enable only the controls relevant while collecting a 2D curve."""
        self.draw_curve_button.setEnabled(not active)
        self.finish_curve_button.setEnabled(active)
        self.clear_curve_button.setEnabled(active)
        self.analyze_curve_button.setEnabled(False)
        self.curve_epicycle_button.setEnabled(False)
        self.curve_spectrum_button.setEnabled(False)
        self.curve_error_button.setEnabled(False)
        self.curve_harmonic_slider.setEnabled(False)
        self.generate_button.setEnabled(not active)
        self.draw_button.setEnabled(not active)
        self.finish_drawing_button.setEnabled(False if active else self.finish_drawing_button.isEnabled())
        self.reconstruct_button.setEnabled(not active and getattr(self, "_signal_available", False))

    def set_curve_analysis_available(self, available: bool) -> None:
        """Enable the Analyze Curve action after a valid curve is finalized."""
        self.analyze_curve_button.setEnabled(available)

    def set_curve_epicycle_available(self, available: bool) -> None:
        """Enable the 2D epicycle view after Fourier analysis succeeds."""
        self.curve_epicycle_button.setEnabled(available)

    def set_curve_spectrum_available(self, available: bool) -> None:
        """Enable the 2D harmonic spectrum after Fourier analysis succeeds."""
        self.curve_spectrum_button.setEnabled(available)

    def set_curve_error_available(self, available: bool) -> None:
        """Enable error/convergence analysis after Fourier analysis succeeds."""
        self.curve_error_button.setEnabled(available)

    def set_curve_reconstruction_available(self, available: bool) -> None:
        """Enable harmonic reconstruction controls after analysis succeeds."""
        self.curve_harmonic_slider.setEnabled(available)

    def set_curve_analysis_status(self, status: str) -> None:
        """Display a compact 2D curve analysis result."""
        self.curve_analysis_label.setText(status)

    def set_curve_harmonic_range(self, maximum: int, value: int = 0) -> None:
        """Set the available 2D harmonic range and selected value."""
        self.curve_harmonic_slider.blockSignals(True)
        self.curve_harmonic_slider.setRange(0, maximum)
        self.curve_harmonic_slider.setValue(value)
        self.curve_harmonic_slider.blockSignals(False)
        self.curve_harmonic_value.setText(str(value))

    def set_curve_stats(self, stats: str) -> None:
        """Display current 2D reconstruction statistics."""
        self.curve_stats_label.setText(stats)

    def reset_curve_controls(self) -> None:
        """Disable all 2D curve analysis controls and restore placeholders."""
        self.set_curve_active(False)
        self.set_curve_analysis_available(False)
        self.set_curve_reconstruction_available(False)
        self.set_curve_epicycle_available(False)
        self.set_curve_spectrum_available(False)
        self.set_curve_error_available(False)
        self.set_curve_analysis_status("2D curve analysis: --")
        self.set_curve_harmonic_range(0, 0)
        self.set_curve_stats("2D Fourier Reconstruction\n--")

    def set_signal_available(self, available: bool) -> None:
        """Enable actions that require a current signal."""
        self._signal_available = bool(available)
        self.clear_button.setEnabled(available)
        self.reconstruct_button.setEnabled(available)
        self.compare_fft_button.setEnabled(False)
        self.epicycle_button.setEnabled(False)

    def set_reconstruction_available(self, available: bool) -> None:
        """Enable actions that require Fourier coefficients or reconstruction."""
        self.compare_fft_button.setEnabled(available)
        self.epicycle_button.setEnabled(available)

    def set_current_signal(self, name: str) -> None:
        """Show which signal source is currently active."""
        self.current_signal_label.setText(f"Current signal: {name}")

    def _harmonic_value_changed(self, value: int) -> None:
        """Update the visible N and notify the controller for live updates."""
        self.harmonic_value.setText(str(value))
        self.harmonic_changed.emit(value)

    def _curve_harmonic_value_changed(self, value: int) -> None:
        """Update the visible 2D harmonic count and notify the controller."""
        self.curve_harmonic_value.setText(str(value))
        self.curve_harmonic_changed.emit(value)

    def _build_layout(self) -> None:
        """Create the compact control-panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        signal_group = QGroupBox("Signal Input")
        signal_layout = QVBoxLayout(signal_group)
        signal_form = QFormLayout()
        signal_form.addRow("Signal:", self.signal_selector)
        signal_layout.addLayout(signal_form)
        signal_layout.addWidget(self.current_signal_label)

        button_row = QHBoxLayout()
        button_row.addWidget(self.generate_button)
        button_row.addWidget(self.clear_button)
        signal_layout.addLayout(button_row)
        drawing_row = QHBoxLayout()
        drawing_row.addWidget(self.draw_button)
        drawing_row.addWidget(self.finish_drawing_button)
        signal_layout.addLayout(drawing_row)
        curve_row = QHBoxLayout()
        curve_row.addWidget(self.draw_curve_button)
        curve_row.addWidget(self.finish_curve_button)
        curve_row.addWidget(self.clear_curve_button)
        signal_layout.addLayout(curve_row)
        signal_layout.addWidget(self.analyze_curve_button)
        signal_layout.addWidget(self.curve_epicycle_button)
        signal_layout.addWidget(self.curve_spectrum_button)
        signal_layout.addWidget(self.curve_error_button)
        signal_layout.addWidget(self.curve_analysis_label)
        curve_harmonic_row = QHBoxLayout()
        curve_harmonic_row.addWidget(QLabel("2D Harmonics N:"))
        curve_harmonic_row.addWidget(self.curve_harmonic_slider)
        curve_harmonic_row.addWidget(self.curve_harmonic_value)
        signal_layout.addLayout(curve_harmonic_row)
        signal_layout.addWidget(self.curve_stats_label)
        layout.addWidget(signal_group)

        analysis_group = QGroupBox("Fourier Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        harmonic_row = QHBoxLayout()
        harmonic_row.addWidget(self.harmonic_slider)
        harmonic_row.addWidget(self.harmonic_value)
        harmonic_form = QFormLayout()
        harmonic_form.addRow("Harmonics N:", harmonic_row)
        analysis_layout.addLayout(harmonic_form)
        analysis_layout.addWidget(self.reconstruct_button)
        layout.addWidget(analysis_group)

        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(10)
        results_layout.addWidget(QLabel("Metrics"))
        metrics_form = QFormLayout()
        metrics_form.setVerticalSpacing(8)
        metrics_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for name, label in self.metric_labels.items():
            metrics_form.addRow(f"{name}:", label)
        results_layout.addLayout(metrics_form)
        results_layout.addWidget(QLabel("Gibbs Analysis"))
        results_layout.addWidget(self.gibbs_status)
        gibbs_form = QFormLayout()
        gibbs_form.setVerticalSpacing(8)
        gibbs_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for name, label in self.gibbs_labels.items():
            gibbs_form.addRow(f"{name}:", label)
        results_layout.addLayout(gibbs_form)
        results_layout.addWidget(QLabel("FFT Comparison Timing"))
        timing_form = QFormLayout()
        timing_form.setVerticalSpacing(8)
        timing_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        timing_form.addRow("Fourier:", self.fft_timing_labels["Fourier"])
        timing_form.addRow("FFT:", self.fft_timing_labels["FFT"])
        results_layout.addLayout(timing_form)
        layout.addWidget(results_group)

        advanced_group = QGroupBox("Advanced Visualization")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.addWidget(self.compare_fft_button)
        advanced_layout.addWidget(self.epicycle_button)
        layout.addWidget(advanced_group)
        layout.addStretch()
