"""Non-interactive smoke tests for Matplotlib visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualization.error_plot import plot_error, plot_error_vs_harmonics
from visualization.signal_plot import (
    plot_harmonic,
    plot_original_vs_reconstructed,
    plot_signal,
)
from visualization.spectrum_plot import plot_magnitude_spectrum, plot_phase_spectrum


def assert_plot_result(result: tuple[Figure, Axes]) -> None:
    figure, axes = result
    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    plt.close(figure)


def test_signal_plots_return_figures_and_axes() -> None:
    time = np.linspace(0.0, 1.0, 101)
    original = np.sin(2.0 * np.pi * time)
    reconstructed = 0.9 * original

    assert_plot_result(plot_signal(time, original))
    assert_plot_result(plot_original_vs_reconstructed(time, original, reconstructed, 1))
    assert_plot_result(plot_harmonic(time, original, 1))


def test_spectrum_plots_return_figures_and_axes() -> None:
    harmonics = np.arange(1, 4)
    magnitude = np.array([1.0, 0.2, 0.1])
    phase = np.array([0.0, -np.pi / 2.0, np.pi])

    assert_plot_result(plot_magnitude_spectrum(harmonics, magnitude))
    assert_plot_result(plot_phase_spectrum(harmonics, phase))


def test_error_plots_return_figures_and_axes() -> None:
    time = np.linspace(0.0, 1.0, 101)
    error = np.sin(2.0 * np.pi * time) * 0.1

    assert_plot_result(plot_error(time, error))
    assert_plot_result(plot_error_vs_harmonics([1, 2, 3], [0.5, 0.2, 0.1]))


def test_plots_can_draw_on_supplied_axes() -> None:
    figure, axes = plt.subplots()
    result_figure, result_axes = plot_signal(
        [0.0, 1.0], [1.0, 0.0], ax=axes, label="provided"
    )

    assert result_figure is figure
    assert result_axes is axes
    plt.close(figure)
