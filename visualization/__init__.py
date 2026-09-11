"""Matplotlib visualization helpers."""

from .error_plot import plot_error, plot_error_vs_harmonics
from .epicycle import EpicycleFrame, EpicycleVisualizer
from .signal_plot import (
    plot_harmonic,
    plot_original_vs_reconstructed,
    plot_signal,
)
from .spectrum_plot import (
    plot_fourier_vs_fft_spectrum,
    plot_magnitude_spectrum,
    plot_phase_spectrum,
)

__all__ = [
    "plot_error",
    "plot_error_vs_harmonics",
    "EpicycleFrame",
    "EpicycleVisualizer",
    "plot_fourier_vs_fft_spectrum",
    "plot_harmonic",
    "plot_magnitude_spectrum",
    "plot_original_vs_reconstructed",
    "plot_phase_spectrum",
    "plot_signal",
]
