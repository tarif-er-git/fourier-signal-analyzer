"""Interactive 2D reconstruction error and convergence window."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from fourier.curve2d_analysis import CurveFourierResult
from fourier.curve2d_error import CurveConvergenceResult, CurveErrorMetrics


class Curve2DErrorWindow(QMainWindow):
    """Display curve comparison, pointwise error, and harmonic convergence."""

    def __init__(
        self,
        coefficients: CurveFourierResult,
        convergence: CurveConvergenceResult,
        selected: CurveErrorMetrics,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Reconstruction Error and Convergence")
        self.resize(1100, 850)
        self.coefficients = coefficients
        self.convergence = convergence
        self.selected = selected
        self.figure = Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.update_selected(selected)

    def update_selected(self, selected: CurveErrorMetrics) -> None:
        """Refresh all plots and metrics for the currently selected N."""
        self.selected = selected
        self.figure.clear()
        axes = self.figure.subplots(2, 2).ravel()
        comparison_axis, error_axis, convergence_axis, stats_axis = axes
        original = np.column_stack((self.coefficients.x_samples, self.coefficients.y_samples))
        reconstructed = np.column_stack((selected.reconstructed_x, selected.reconstructed_y))
        original_closed = np.vstack((original, original[0]))
        reconstructed_closed = np.vstack((reconstructed, reconstructed[0]))
        comparison_axis.plot(original_closed[:, 0], original_closed[:, 1], color="0.45", label="Original")
        comparison_axis.plot(
            reconstructed_closed[:, 0], reconstructed_closed[:, 1],
            color="tab:orange", linestyle="--", label=f"Reconstructed (N={selected.harmonic_count})",
        )
        segments = np.stack((original, reconstructed), axis=1)
        comparison_axis.add_collection(
            LineCollection(segments, colors="tab:red", alpha=0.18, linewidths=0.5)
        )
        all_points = np.vstack((original, reconstructed))
        center = np.mean(all_points, axis=0)
        span = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1])), 1e-6)
        margin = span * 0.1
        limit = span / 2.0 + margin
        comparison_axis.set_xlim(center[0] - limit, center[0] + limit)
        comparison_axis.set_ylim(center[1] - limit, center[1] + limit)
        comparison_axis.set_aspect("equal", adjustable="box")
        comparison_axis.set_title("Original vs Reconstruction")
        comparison_axis.set_xlabel("X")
        comparison_axis.set_ylabel("Y")
        comparison_axis.grid(True, alpha=0.25)
        comparison_axis.legend(loc="best", fontsize=8)

        error_axis.plot(selected.parameter, selected.errors, color="tab:red")
        error_axis.fill_between(selected.parameter, selected.errors, alpha=0.2, color="tab:red")
        error_axis.set_title("Error Along Curve")
        error_axis.set_xlabel("Curve Parameter t")
        error_axis.set_ylabel("Pointwise Error")
        error_axis.grid(True, alpha=0.25)

        counts = self.convergence.harmonic_counts
        rmse = np.asarray([metrics.rmse for metrics in self.convergence.metrics])
        mean_error = np.asarray([metrics.mean_error for metrics in self.convergence.metrics])
        max_error = np.asarray([metrics.maximum_error for metrics in self.convergence.metrics])
        convergence_axis.plot(counts, rmse, color="tab:blue", label="RMSE")
        convergence_axis.plot(counts, mean_error, color="tab:green", alpha=0.7, label="Mean Error")
        convergence_axis.plot(counts, max_error, color="tab:red", alpha=0.7, label="Max Error")
        convergence_axis.scatter([selected.harmonic_count], [selected.rmse], color="black", zorder=3)
        convergence_axis.set_title("Harmonic Convergence")
        convergence_axis.set_xlabel("Number of Harmonics")
        convergence_axis.set_ylabel("Error")
        convergence_axis.grid(True, alpha=0.25)
        convergence_axis.legend(fontsize=8)

        stats_axis.axis("off")
        stats_axis.text(
            0.02,
            0.95,
            "Current Reconstruction\n\n"
            f"Harmonics: {selected.harmonic_count}\n"
            f"MSE: {selected.mse:.6g}\n"
            f"RMSE: {selected.rmse:.6g}\n"
            f"Mean Error: {selected.mean_error:.6g}\n"
            f"Max Error: {selected.maximum_error:.6g}\n\n"
            f"Largest Error Location:\n"
            f"t = {selected.maximum_parameter:.6g}\n"
            f"x = {self.coefficients.x_samples[selected.maximum_index]:.6g}\n"
            f"y = {self.coefficients.y_samples[selected.maximum_index]:.6g}",
            va="top",
            fontsize=11,
        )
        self.canvas.draw_idle()

    def closeEvent(self, event) -> None:
        """Release the Matplotlib canvas when closing the window."""
        super().closeEvent(event)
