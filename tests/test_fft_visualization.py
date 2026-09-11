"""Non-interactive smoke test for the Fourier-versus-FFT plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualization.spectrum_plot import plot_fourier_vs_fft_spectrum


def test_fourier_vs_fft_plot_returns_figure_and_axes() -> None:
    figure, axes = plot_fourier_vs_fft_spectrum([1, 2], [1.0, 0.1], [1.0, 0.09])

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    plt.close(figure)
