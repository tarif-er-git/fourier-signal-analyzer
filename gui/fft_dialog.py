"""Pop-up dialog displaying Fourier Series vs FFT magnitude comparison."""

from __future__ import annotations

from typing import Sequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from visualization.spectrum_plot import plot_fourier_vs_fft_spectrum


class FFTComparisonDialog(QDialog):
    """Pop-up window comparing Fourier Series and FFT spectra."""

    def __init__(
        self,
        harmonics: Sequence[int],
        fourier_magnitudes: Sequence[float],
        fft_magnitudes: Sequence[float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fourier Series vs FFT Magnitude")
        self.resize(780, 500)

        self.figure = Figure(figsize=(7.8, 5.0))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        plot_fourier_vs_fft_spectrum(
            harmonics,
            fourier_magnitudes,
            fft_magnitudes,
            ax=self.ax,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.figure.tight_layout()
        self.canvas.draw()


