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
    # Padded x-limits keep all stem markers inside the axes
    assert axes.get_xlim() == (0.5, 20.5)
    assert axes.get_ylim() == (0.0, 1.1)
    # X-axis must only show integer tick labels
    ticks = [t for t in axes.get_xticks() if 0.5 <= t <= 20.5]
    assert all(float(t).is_integer() for t in ticks)
    plt.close(figure)

