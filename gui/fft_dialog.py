"""Pop-up dialog displaying Fourier Series vs FFT magnitude comparison."""

from __future__ import annotations

from typing import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from visualization.spectrum_plot import plot_fourier_vs_fft_spectrum


class FFTComparisonDialog(QDialog):
    """Pop-up window comparing Fourier Series and FFT spectra using stem plots."""

    def __init__(
        self,
        harmonics: Sequence[int],
        fourier_magnitudes: Sequence[float],
        fft_magnitudes: Sequence[float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fourier Series vs FFT Magnitude")
        self.resize(700, 500)
        self.setMinimumSize(700, 500)

        # Use subplots_adjust to reserve enough room for axis labels on all sides
        self.figure = Figure(figsize=(7, 5))
        self.figure.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.13)
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        plot_fourier_vs_fft_spectrum(
            harmonics,
            fourier_magnitudes,
            fft_magnitudes,
            ax=self.ax,
        )
        self.canvas.draw()
