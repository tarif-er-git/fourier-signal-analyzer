"""Reusable Matplotlib plots for time-domain signals."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


def _get_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return an existing axes or create a new figure and axes."""
    if ax is None:
        figure, axes = plt.subplots()
        return figure, axes
    return ax.get_figure(), ax


def _validate_series(t: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate one time series for plotting."""
    time = np.asarray(t, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and signal values must be one-dimensional")
    if time.shape != signal.shape:
        raise ValueError("time and signal values must have the same shape")
    if time.size == 0:
        raise ValueError("time and signal values must not be empty")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("time and signal values must be finite")
    return time, signal


def plot_signal(
    t: Sequence[float],
    signal: Sequence[float],
    *,
    ax: Axes | None = None,
    title: str = "Signal",
    label: str = "Signal",
) -> tuple[Figure, Axes]:
    """Plot one time-domain signal and return its Figure and Axes."""
    time, values = _validate_series(t, signal)
    figure, axes = _get_axes(ax)
    axes.plot(time, values, label=label)
    axes.set_title(title)
    axes.set_xlabel("Time")
    axes.set_ylabel("Amplitude")
    axes.grid(True, alpha=0.3)
    if label:
        axes.legend()
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_original_vs_reconstructed(
    t: Sequence[float],
    original: Sequence[float],
    reconstructed: Sequence[float],
    num_harmonics: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot an original signal and its N-harmonic reconstruction together."""
    time, original_values = _validate_series(t, original)
    reconstructed_time, reconstructed_values = _validate_series(t, reconstructed)
    if reconstructed_time.shape != time.shape:
        raise ValueError("original and reconstructed must have the same shape")
    figure, axes = _get_axes(ax)
    axes.plot(time, original_values, label="Original", linewidth=1.5)
    axes.plot(
        time,
        reconstructed_values,
        label=f"Reconstructed (N={num_harmonics})",
        linestyle="--",
    )
    axes.set_title(f"Original vs Reconstructed Signal (N={num_harmonics})")
    axes.set_xlabel("Time")
    axes.set_ylabel("Amplitude")
    axes.grid(True, alpha=0.3)
    axes.legend()
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_harmonic(
    t: Sequence[float],
    harmonic: Sequence[float],
    harmonic_number: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one individual Fourier harmonic component."""
    time, values = _validate_series(t, harmonic)
    figure, axes = _get_axes(ax)
    axes.plot(time, values, label=f"Harmonic {harmonic_number}")
    axes.set_title(f"Fourier Harmonic {harmonic_number}")
    axes.set_xlabel("Time")
    axes.set_ylabel("Amplitude")
    axes.grid(True, alpha=0.3)
    axes.legend()
    if ax is None:
        figure.tight_layout()
    return figure, axes
