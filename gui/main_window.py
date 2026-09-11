"""Main PySide6 controller for the Signal Sketch and Decompose app."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
    QFileDialog,
)
from PySide6.QtGui import QAction, QKeySequence

from fourier.analysis import FourierAnalyzer
from fourier.fft_comparison import (
    FourierFFTComparison,
    compare_fourier_and_fft,
    compare_performance,
    prepare_fft_period_samples,
)
from fourier.synthesis import FourierSynthesizer
from fourier.spectrum import FourierSpectrum
from gui.canvas import SignalCanvas
from gui.controls import ControlPanel
from gui.drawing import prepare_custom_signal
from gui.epicycle_window import EpicycleWindow
from metrics.convergence import ConvergenceResult, analyze_convergence
from metrics.error import (
    calculate_mae,
    calculate_max_error,
    calculate_mse,
    calculate_rmse,
)
from metrics.gibbs import GibbsResult, analyze_gibbs, theoretical_gibbs_overshoot
from presets.signals import get_preset_signal
from signal_io import (
    export_analysis_report,
    export_error,
    export_reconstruction,
    export_spectrum,
    load_signal,
    save_signal,
)


class MainWindow(QMainWindow):
    """Coordinate user controls, mathematical modules, and plot updates."""

    MAX_HARMONICS = 50

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signal Sketch and Decompose")
        self.resize(1350, 900)

        self.controls = ControlPanel()
        self.canvas = SignalCanvas()
        self.status_label = QLabel("Choose a preset and click Generate.")

        self.t: np.ndarray | None = None
        self.original_signal: np.ndarray | None = None
        self.active_signal_data: tuple[np.ndarray, np.ndarray] | None = None
        self.current_signal_data: tuple[np.ndarray, np.ndarray] | None = None
        self.active_signal_type: str | None = None
        self.drawn_t: np.ndarray | None = None
        self.drawn_x: np.ndarray | None = None
        self.analysis_result: FourierAnalyzer | None = None
        self.reconstructed_signal: np.ndarray | None = None
        self.error: np.ndarray | None = None
        self.convergence_result: ConvergenceResult | None = None
        self.gibbs_result: GibbsResult | None = None
        self.fft_comparison: FourierFFTComparison | None = None
        self.epicycle_window: EpicycleWindow | None = None
        self.file_actions: dict[str, QAction] = {}

        self._build_layout()
        self.controls.generate_requested.connect(self.generate_signal)
        self.controls.reconstruct_requested.connect(self.reconstruct_signal)
        self.controls.reset_requested.connect(self.reset)
        self.controls.draw_custom_requested.connect(self.start_custom_drawing)
        self.controls.finish_drawing_requested.connect(self.finish_custom_drawing)
        self.controls.harmonic_changed.connect(self.reconstruct_if_ready)
        self.controls.fft_comparison_requested.connect(self.compare_with_fft)
        self.controls.epicycle_requested.connect(self.show_epicycles)
        self._update_action_state()

    def _build_layout(self) -> None:
        """Build the top-level controls and visualization layout."""
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        controls_scroll.setMinimumWidth(310)
        controls_scroll.setWidget(self.controls)
        self.controls_scroll = controls_scroll
        main_layout.addWidget(controls_scroll, 0)
        main_layout.addWidget(self.canvas, 1)
        self.setCentralWidget(central_widget)
        self.statusBar().addWidget(self.status_label)
        self._build_file_menu()

    def _build_file_menu(self) -> None:
        """Create the small File menu for persistence and numerical exports."""
        file_menu = self.menuBar().addMenu("File")
        self.file_actions["load"] = file_menu.addAction("Load Signal", self.load_signal_file)
        self.file_actions["save"] = file_menu.addAction("Save Signal", self.save_signal_file)
        self.file_actions["load"].setShortcut(QKeySequence.StandardKey.Open)
        self.file_actions["save"].setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addSeparator()
        self.file_actions["reconstruction"] = file_menu.addAction(
            "Export Reconstruction", self.export_reconstruction_file
        )
        self.file_actions["spectrum"] = file_menu.addAction(
            "Export Spectrum", self.export_spectrum_file
        )
        self.file_actions["error"] = file_menu.addAction(
            "Export Error", self.export_error_file
        )
        self.file_actions["report"] = file_menu.addAction(
            "Export Analysis Report", self.export_report_file
        )
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit", self.close)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def _update_action_state(self) -> None:
        """Enable file actions according to the current application state."""
        has_signal = self.t is not None and self.original_signal is not None
        has_reconstruction = self.reconstructed_signal is not None
        self.file_actions["save"].setEnabled(has_signal)
        self.file_actions["reconstruction"].setEnabled(has_reconstruction)
        self.file_actions["spectrum"].setEnabled(self.analysis_result is not None)
        self.file_actions["error"].setEnabled(self.error is not None)
        self.file_actions["report"].setEnabled(has_signal)

    def show_about(self) -> None:
        """Show a concise educational project description."""
        QMessageBox.about(
            self,
            "About Signal Sketch and Decompose",
            "Signal Sketch and Decompose\n\n"
            "An educational Fourier Series visualization tool.\n\n"
            "Explore Fourier analysis, synthesis, harmonic components, "
            "reconstruction error, the Gibbs phenomenon, FFT comparison, "
            "and epicycle visualization.",
        )

    def _set_active_signal(
        self,
        time_values: np.ndarray,
        signal_values: np.ndarray,
        signal_type: str,
        display_name: str,
    ) -> None:
        """Store the single signal used by every downstream analysis action."""
        self.t = time_values
        self.original_signal = signal_values
        self.active_signal_data = (time_values, signal_values)
        self.current_signal_data = self.active_signal_data
        self.active_signal_type = signal_type
        self.controls.set_current_signal(display_name)

    def _clear_active_signal(self) -> None:
        """Clear the active signal while preserving no stale source state."""
        self.t = None
        self.original_signal = None
        self.active_signal_data = None
        self.current_signal_data = None
        self.active_signal_type = None

    def generate_signal(self) -> None:
        """Generate and display the selected preset signal."""
        self._close_epicycle_window()
        self.canvas.clear_drawing()
        time_values = np.linspace(0.0, 1.0, 1001)
        generator = get_preset_signal(self.controls.selected_signal)
        signal_values = generator(time_values, amplitude=1.0, frequency=1.0)
        self._set_active_signal(
            time_values,
            signal_values,
            self.controls.selected_signal,
            self.controls.signal_selector.currentText(),
        )
        self.analysis_result = None
        self.reconstructed_signal = None
        self.error = None
        self.convergence_result = None
        self.gibbs_result = None
        self.fft_comparison = None
        self.drawn_t = None
        self.drawn_x = None
        self.controls.reset_metrics()
        self.controls.reset_gibbs()
        self.controls.reset_fft_timing()
        self.controls.set_drawing_active(False)
        self.controls.set_signal_available(True)
        self.controls.set_reconstruction_available(False)
        self.canvas.show_original(self.t, self.original_signal)
        self._update_action_state()
        self.status_label.setText(
            f"Generated {self.controls.signal_selector.currentText()} signal."
        )

    def save_signal_file(self) -> None:
        """Save the current signal through a JSON file dialog."""
        if self.t is None or self.original_signal is None:
            QMessageBox.warning(self, "No signal", "Generate or load a signal first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Signal", "signal.json", "Signal JSON (*.json)"
        )
        if not path:
            return
        try:
            save_signal(
                path,
                self.t,
                self.original_signal,
                {
                    "name": self.controls.current_signal_label.text(),
                    "type": "signal",
                    "period": float(self.t[-1] - self.t[0]),
                    "sampling_rate": float(1.0 / np.median(np.diff(self.t))),
                },
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return
        self.status_label.setText(f"Saved signal to {path}")
        QMessageBox.information(self, "Signal saved", "The signal was saved successfully.")

    def load_signal_file(self) -> None:
        """Load a JSON signal, clear derived state, and display it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Signal", "", "Signal JSON (*.json)"
        )
        if not path:
            return
        try:
            loaded = load_signal(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Load failed", str(error))
            return
        self._close_epicycle_window()
        self._set_active_signal(loaded.time, loaded.signal, "loaded", "Loaded")
        self.drawn_t = None
        self.drawn_x = None
        self.analysis_result = None
        self.reconstructed_signal = None
        self.error = None
        self.convergence_result = None
        self.gibbs_result = None
        self.fft_comparison = None
        self.controls.reset_metrics()
        self.controls.reset_gibbs()
        self.controls.reset_fft_timing()
        self.controls.set_drawing_active(False)
        self.controls.set_signal_available(True)
        self.controls.set_reconstruction_available(False)
        self.canvas.show_original(self.t, self.original_signal)
        self._update_action_state()
        self.status_label.setText(f"Loaded signal from {path}")

    def _choose_export_path(self, title: str, filename: str) -> str:
        """Return a CSV destination selected by the user, or an empty string."""
        path, _ = QFileDialog.getSaveFileName(self, title, filename, "CSV (*.csv)")
        return path

    def export_reconstruction_file(self) -> None:
        """Export the current reconstructed signal as CSV."""
        if self.t is None or self.reconstructed_signal is None:
            QMessageBox.warning(self, "No reconstruction", "Reconstruct a signal first.")
            return
        path = self._choose_export_path("Export Reconstruction", "reconstruction.csv")
        if not path:
            return
        try:
            export_reconstruction(
                path,
                self.t,
                self.reconstructed_signal,
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.status_label.setText(f"Exported reconstruction to {path}")

    def export_spectrum_file(self) -> None:
        """Export calculated Fourier magnitude and phase values as CSV."""
        if self.analysis_result is None:
            QMessageBox.warning(self, "No spectrum", "Reconstruct a signal first.")
            return
        path = self._choose_export_path("Export Spectrum", "spectrum.csv")
        if not path:
            return
        spectrum = FourierSpectrum(
            self.analysis_result.a0,
            self.analysis_result.a,
            self.analysis_result.b,
        )
        try:
            export_spectrum(path, spectrum.harmonics, spectrum.magnitude(), spectrum.phase())
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.status_label.setText(f"Exported spectrum to {path}")

    def export_error_file(self) -> None:
        """Export the current point-wise reconstruction error as CSV."""
        if self.t is None or self.error is None:
            QMessageBox.warning(self, "No error", "Reconstruct a signal first.")
            return
        path = self._choose_export_path("Export Error", "error.csv")
        if not path:
            return
        try:
            export_error(path, self.t, self.error)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.status_label.setText(f"Exported error to {path}")

    def export_report_file(self) -> None:
        """Export a compact JSON report from already-computed application state."""
        if self.t is None or self.original_signal is None:
            QMessageBox.warning(self, "No analysis", "Generate or load a signal first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Report", "analysis_report.json", "JSON (*.json)"
        )
        if not path:
            return
        report: dict[str, object] = {
            "number_of_samples": int(self.t.size),
            "period": float(self.t[-1] - self.t[0]),
            "harmonics": self.controls.num_harmonics,
        }
        if self.reconstructed_signal is not None:
            report.update(
                {
                    "mse": calculate_mse(self.original_signal, self.reconstructed_signal),
                    "rmse": calculate_rmse(self.original_signal, self.reconstructed_signal),
                    "mae": calculate_mae(self.original_signal, self.reconstructed_signal),
                    "max_error": calculate_max_error(self.original_signal, self.reconstructed_signal),
                }
            )
        if self.gibbs_result is not None:
            report["gibbs"] = {
                "detected": self.gibbs_result.detected,
                "message": self.gibbs_result.message,
                "overshoot": self.gibbs_result.overshoot,
                "undershoot": self.gibbs_result.undershoot,
                "theoretical": (
                    theoretical_gibbs_overshoot(self.gibbs_result.jump_magnitude)
                    if self.gibbs_result.jump_magnitude is not None
                    else None
                ),
            }
        try:
            export_analysis_report(path, report)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.status_label.setText(f"Exported analysis report to {path}")

    def start_custom_drawing(self) -> None:
        """Clear stale results and activate the canvas drawing region."""
        self._close_epicycle_window()
        self._clear_active_signal()
        self.analysis_result = None
        self.reconstructed_signal = None
        self.error = None
        self.convergence_result = None
        self.gibbs_result = None
        self.fft_comparison = None
        self.drawn_t = None
        self.drawn_x = None
        self.controls.reset_metrics()
        self.controls.reset_gibbs()
        self.controls.reset_fft_timing()
        self.controls.set_drawing_active(True)
        self.controls.set_signal_available(False)
        self.controls.set_current_signal("Drawing custom signal...")
        self.canvas.start_drawing()
        self._update_action_state()
        self.status_label.setText("Draw while holding the left mouse button, then finish.")

    def finish_custom_drawing(self) -> None:
        """Process completed raw points into the application's signal grid."""
        try:
            drawn_t, drawn_x = self.canvas.finish_drawing()
            custom_time, custom_signal = prepare_custom_signal(
                drawn_t, drawn_x, num_samples=1001
            )
        except ValueError as error:
            QMessageBox.warning(self, "Drawing not ready", str(error))
            return

        self.drawn_t = drawn_t
        self.drawn_x = drawn_x
        self._set_active_signal(custom_time, custom_signal, "custom", "Custom")
        self.analysis_result = None
        self.reconstructed_signal = None
        self.error = None
        self.convergence_result = None
        self.gibbs_result = None
        self.fft_comparison = None
        self.controls.reset_metrics()
        self.controls.reset_gibbs()
        self.controls.reset_fft_timing()
        self.controls.set_drawing_active(False)
        self.controls.set_signal_available(True)
        self.controls.set_reconstruction_available(False)
        self.canvas.show_original(self.t, self.original_signal)
        self._update_action_state()
        self.status_label.setText("Custom signal ready for reconstruction.")

    def reconstruct_signal(self, show_warning: bool = True) -> None:
        """Analyze, reconstruct, score, and display the current signal."""
        if self.active_signal_data is None:
            if show_warning:
                QMessageBox.warning(
                    self,
                    "No signal generated",
                    "Generate or finish drawing a signal before reconstructing it.",
                )
            return

        active_time, active_signal = self.active_signal_data
        num_harmonics = self.controls.num_harmonics
        self.analysis_result = FourierAnalyzer(
            active_time, active_signal, num_harmonics=self.MAX_HARMONICS
        ).analyze()
        synthesizer = FourierSynthesizer(
            active_time,
            self.analysis_result.a0,
            self.analysis_result.a,
            self.analysis_result.b,
            self.analysis_result.omega0,
        )
        self.reconstructed_signal = synthesizer.reconstruct(num_harmonics)
        self.error = active_signal - self.reconstructed_signal
        self.convergence_result = analyze_convergence(
            active_signal,
            synthesizer,
            np.arange(1, self.MAX_HARMONICS + 1),
        )
        self.gibbs_result = analyze_gibbs(
            active_time, active_signal, self.reconstructed_signal
        )
        self.canvas.show_reconstruction(
            active_time,
            active_signal,
            self.reconstructed_signal,
            self.error,
            num_harmonics,
            self.convergence_result.harmonic_counts,
            self.convergence_result.mse_values,
        )
        self.controls.set_metrics(
            calculate_mse(active_signal, self.reconstructed_signal),
            calculate_rmse(active_signal, self.reconstructed_signal),
            calculate_mae(active_signal, self.reconstructed_signal),
            calculate_max_error(active_signal, self.reconstructed_signal),
        )
        if self.gibbs_result.detected:
            jump = self.gibbs_result.jump_magnitude or 0.0
            self.controls.set_gibbs_result(
                self.gibbs_result.message,
                self.gibbs_result.overshoot,
                self.gibbs_result.undershoot,
                theoretical_gibbs_overshoot(jump),
            )
        else:
            self.controls.set_gibbs_result(self.gibbs_result.message)
        self.controls.set_reconstruction_available(True)
        self._update_action_state()
        self.status_label.setText(
            f"Reconstructed with {num_harmonics} harmonics."
        )

    def reconstruct_if_ready(self, _value: int) -> None:
        """Live-update reconstruction when N changes for an active signal."""
        if self.t is not None and self.original_signal is not None:
            self.reconstruct_signal(show_warning=False)

    def compare_with_fft(self) -> None:
        """Compare the current signal with NumPy's FFT without changing reconstruction."""
        if self.t is None or self.original_signal is None:
            QMessageBox.warning(
                self,
                "No signal generated",
                "Generate or finish drawing a signal before comparing with FFT.",
            )
            return
        if self.analysis_result is None:
            self.analysis_result = FourierAnalyzer(
                self.t, self.original_signal, num_harmonics=self.MAX_HARMONICS
            ).analyze()
        self.fft_comparison = compare_fourier_and_fft(
            self.t, self.original_signal, self.analysis_result
        )
        self.canvas.show_fft_comparison(
            self.fft_comparison.harmonic_numbers,
            self.fft_comparison.fourier_magnitudes,
            self.fft_comparison.fft_magnitudes,
        )
        fft_samples, _ = prepare_fft_period_samples(self.t, self.original_signal)
        timing = compare_performance(
            [fft_samples.size], harmonics=min(self.MAX_HARMONICS, fft_samples.size // 2 - 1)
        )[0]
        self.controls.set_fft_timing(timing.fourier_seconds, timing.fft_seconds)
        self._update_action_state()
        self.status_label.setText("Fourier Series and FFT spectra compared.")

    def show_epicycles(self) -> None:
        """Open or replace the dedicated epicycle animation window."""
        if self.t is None or self.original_signal is None:
            QMessageBox.warning(
                self,
                "No signal generated",
                "Generate or finish drawing a signal before opening epicycles.",
            )
            return
        if self.analysis_result is None:
            self.analysis_result = FourierAnalyzer(
                self.t, self.original_signal, num_harmonics=self.MAX_HARMONICS
            ).analyze()
        self._close_epicycle_window()
        self.epicycle_window = EpicycleWindow(
            self.analysis_result.a0,
            self.analysis_result.a,
            self.analysis_result.b,
            self.analysis_result.omega0,
            self.controls.num_harmonics,
            self.controls.current_signal_label.text().replace("Current signal: ", ""),
            self,
        )
        self.epicycle_window.show()
        self.status_label.setText("Epicycle visualization opened.")

    def _close_epicycle_window(self) -> None:
        """Stop and release the dedicated epicycle window, if open."""
        if self.epicycle_window is not None:
            self.epicycle_window.close()
            self.epicycle_window = None

    def reset(self) -> None:
        """Clear current signal state, metrics, and plots."""
        self._clear_active_signal()
        self.drawn_t = None
        self.drawn_x = None
        self.analysis_result = None
        self.reconstructed_signal = None
        self.error = None
        self.convergence_result = None
        self.gibbs_result = None
        self.fft_comparison = None
        self._close_epicycle_window()
        self.controls.reset_metrics()
        self.controls.reset_gibbs()
        self.controls.reset_fft_timing()
        self.controls.set_drawing_active(False)
        self.controls.set_signal_available(False)
        self.controls.set_reconstruction_available(False)
        self.controls.set_current_signal("None")
        self.canvas.show_empty_state()
        self._update_action_state()
        self.status_label.setText("Choose a preset and click Generate.")
