"""Reusable Matplotlib plots for Fourier magnitude and phase spectra."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def _get_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return an existing axes or create a new figure and axes."""
    if ax is None:
        figure, axes = plt.subplots()
        return figure, axes
    return ax.get_figure(), ax


def _validate_spectrum_inputs(
    harmonics: Sequence[int], values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate harmonic indices and corresponding spectral values."""
    harmonic_values = np.asarray(harmonics)
    spectral_values = np.asarray(values, dtype=float)
    if harmonic_values.ndim != 1 or spectral_values.ndim != 1:
        raise ValueError("harmonics and spectral values must be one-dimensional")
    if harmonic_values.shape != spectral_values.shape:
        raise ValueError("harmonics and spectral values must have the same shape")
    if not np.all(np.isfinite(harmonic_values)) or not np.all(
        np.isfinite(spectral_values)
    ):
        raise ValueError("harmonics and spectral values must be finite")
    return harmonic_values, spectral_values


def plot_magnitude_spectrum(
    harmonics: Sequence[int],
    magnitude: Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create a stem plot of harmonic number versus magnitude."""
    harmonic_values, magnitudes = _validate_spectrum_inputs(harmonics, magnitude)
    figure, axes = _get_axes(ax)
    axes.stem(harmonic_values, magnitudes, basefmt=" ")
    axes.set_title("Fourier Magnitude Spectrum")
    axes.set_xlabel("Harmonic Number")
    axes.set_ylabel("Magnitude")
    axes.grid(True, alpha=0.3)
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_phase_spectrum(
    harmonics: Sequence[int],
    phase: Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create a stem plot of harmonic number versus phase in radians."""
    harmonic_values, phases = _validate_spectrum_inputs(harmonics, phase)
    figure, axes = _get_axes(ax)
    axes.stem(harmonic_values, phases, basefmt=" ")
    axes.set_title("Fourier Phase Spectrum")
    axes.set_xlabel("Harmonic Number")
    axes.set_ylabel("Phase (radians)")
    axes.grid(True, alpha=0.3)
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_fourier_vs_fft_spectrum(
    harmonics: Sequence[int],
    fourier_magnitude: Sequence[float],
    fft_magnitude: Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot Fourier Series and FFT magnitudes side-by-side using discrete stem plots."""
    harmonic_values, fourier_values = _validate_spectrum_inputs(
        harmonics, fourier_magnitude
    )
    _, fft_values = _validate_spectrum_inputs(harmonics, fft_magnitude)
    figure, axes = _get_axes(ax)

    # Side-by-side stem offset so both series are clearly distinguishable.
    # Clipping is kept ON (the default) so stems never render outside the axes box.
    offset = 0.15
    axes.stem(
        harmonic_values - offset,
        fourier_values,
        linefmt="tab:blue",
        markerfmt="C0o",
        basefmt=" ",
        label="Fourier Series",
    )
    axes.stem(
        harmonic_values + offset,
        fft_values,
        linefmt="tab:orange",
        markerfmt="C1s",
        basefmt=" ",
        label="FFT",
    )

    axes.axhline(0, color="gray", linewidth=0.8, alpha=0.5)

    # X-axis: add 0.5-unit padding on each side so the first/last stems are
    # fully inside the plot frame (harmonic 1 stem at 0.85 stays within 0.5).
    # Y-axis: 0 to 1.1 linear scale.
    axes.set_xlim(0.5, 20.5)
    axes.set_ylim(0.0, 1.1)

    # Only show integer tick labels (1, 2, 3 …) on the x-axis.
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes.set_title("Fourier Series vs FFT Magnitude")
    axes.set_xlabel("Harmonic Number")
    axes.set_ylabel("Magnitude")
    axes.grid(True, alpha=0.3)
    axes.legend()
    if ax is None:
        figure.tight_layout()
    return figure, axes
