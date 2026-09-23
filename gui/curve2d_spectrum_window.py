"""Interactive harmonic spectrum window for analyzed 2D curves."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fourier.curve2d_analysis import CurveFourierResult
from visualization.curve2d_spectrum import (
    Curve2DSpectrumRow,
    build_spectrum_rows,
    dominant_harmonics,
    sort_spectrum_rows,
)


class Curve2DSpectrumWindow(QMainWindow):
    """Show X/Y magnitudes, phases, combined magnitude, and harmonic details."""

    def __init__(
        self,
        coefficients: CurveFourierResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Fourier Harmonic Spectrum")
        self.resize(1050, 800)
        self.coefficients = coefficients
        self.rows = build_spectrum_rows(coefficients)
        self.display_rows = self.rows
        self.figure = Figure(figsize=(9, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._build_layout()
        self._plot_spectrum()
        self._populate_table()
        self._select_row(0)

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas, 1)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self.sort_selector = QComboBox()
        self.sort_selector.addItem("Harmonic Index", "harmonic")
        self.sort_selector.addItem("Combined Magnitude", "combined_magnitude")
        self.sort_selector.currentIndexChanged.connect(self._sort_changed)
        sort_row.addWidget(self.sort_selector)
        sort_row.addStretch()
        self.dominant_label = QLabel()
        self.dominant_label.setWordWrap(True)
        sort_row.addWidget(self.dominant_label, 1)
        layout.addLayout(sort_row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["k", "|Cx|", "Phase X (rad)", "|Cy|", "Phase Y (rad)", "Combined 2D Magnitude"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._table_selection_changed)
        layout.addWidget(self.table, 1)
        self.selected_label = QLabel("Select a harmonic for details.")
        self.selected_label.setWordWrap(True)
        layout.addWidget(self.selected_label)
        self.setCentralWidget(central)

    def _plot_spectrum(self) -> None:
        self.figure.clear()
        magnitude_axis, phase_axis = self.figure.subplots(2, 1, sharex=True)
        self._magnitude_axis = magnitude_axis
        self._phase_axis = phase_axis
        harmonics = self.coefficients.harmonics
        x_magnitudes = self.coefficients.x_magnitudes
        y_magnitudes = self.coefficients.y_magnitudes
        combined = np.hypot(x_magnitudes, y_magnitudes)
        x_phases = self.coefficients.x_phases
        y_phases = self.coefficients.y_phases
        width = 0.25
        magnitude_axis.bar(harmonics - width, x_magnitudes, width=width, label="X magnitude")
        magnitude_axis.bar(harmonics, y_magnitudes, width=width, label="Y magnitude")
        magnitude_axis.bar(harmonics + width, combined, width=width, label="Combined 2D magnitude")
        magnitude_axis.set_ylabel("Magnitude")
        magnitude_axis.set_title("2D Fourier Harmonic Spectrum")
        magnitude_axis.grid(True, axis="y", alpha=0.25)
        magnitude_axis.legend(loc="best", fontsize=8)
        phase_axis.plot(harmonics, x_phases, "o-", markersize=3, label="X phase")
        phase_axis.plot(harmonics, y_phases, "o-", markersize=3, label="Y phase")
        phase_axis.set_xlabel("Harmonic Index")
        phase_axis.set_ylabel("Phase (radians)")
        phase_axis.grid(True, alpha=0.25)
        phase_axis.legend(loc="best", fontsize=8)
        self._selected_magnitude_line = magnitude_axis.axvline(
            0.0, color="tab:red", alpha=0.7, visible=False
        )
        self._selected_phase_line = phase_axis.axvline(
            0.0, color="tab:red", alpha=0.7, visible=False
        )
        self.canvas.draw_idle()

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for row_index, row in enumerate(self.display_rows):
            self.table.insertRow(row_index)
            values = (
                str(row.harmonic),
                f"{row.x_magnitude:.6g}",
                f"{row.x_phase:.6g}",
                f"{row.y_magnitude:.6g}",
                f"{row.y_phase:.6g}",
                f"{row.combined_magnitude:.6g}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        dominant = dominant_harmonics(self.rows, count=min(3, len(self.rows)))
        self.dominant_label.setText(
            "Largest coefficient magnitudes: "
            + ", ".join(f"k={row.harmonic}" for row in dominant)
        )

    def _sort_changed(self) -> None:
        self.display_rows = sort_spectrum_rows(
            self.rows, self.sort_selector.currentData()
        )
        self._populate_table()
        self._select_row(0)

    def _select_row(self, row_index: int) -> None:
        if self.table.rowCount() > 0:
            self.table.selectRow(max(0, min(row_index, self.table.rowCount() - 1)))

    def _table_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        row = self.display_rows[selected[0].row()]
        self._show_selected(row)

    def _show_selected(self, row: Curve2DSpectrumRow) -> None:
        self._selected_magnitude_line.set_xdata([row.harmonic, row.harmonic])
        self._selected_phase_line.set_xdata([row.harmonic, row.harmonic])
        self._selected_magnitude_line.set_visible(True)
        self._selected_phase_line.set_visible(True)
        self.canvas.draw_idle()
        self.selected_label.setText(
            f"Selected Harmonic: k = {row.harmonic} | "
            f"Cx magnitude = {row.x_magnitude:.6g}, Phase X = {row.x_phase:.6g} rad | "
            f"Cy magnitude = {row.y_magnitude:.6g}, Phase Y = {row.y_phase:.6g} rad | "
            f"Combined Magnitude = {row.combined_magnitude:.6g}"
        )
