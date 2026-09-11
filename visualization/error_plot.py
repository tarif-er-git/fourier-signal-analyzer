"""Reusable Matplotlib plots for reconstruction errors."""

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
        raise ValueError("time and values must be one-dimensional")
    if time.shape != signal.shape:
        raise ValueError("time and values must have the same shape")
    if time.size == 0:
        raise ValueError("time and values must not be empty")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("time and values must be finite")
    return time, signal


def plot_error(
    t: Sequence[float],
    error: Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a precomputed point-wise reconstruction error signal."""
    time, error_values = _validate_series(t, error)
    figure, axes = _get_axes(ax)
    axes.plot(time, error_values, label="Error")
    axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes.set_title("Reconstruction Error")
    axes.set_xlabel("Time")
    axes.set_ylabel("Error")
    axes.grid(True, alpha=0.3)
    axes.legend()
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_error_vs_harmonics(
    harmonic_counts: Sequence[int],
    mse_values: Sequence[float],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot precomputed MSE values against the number of harmonics."""
    counts = np.asarray(harmonic_counts)
    mse = np.asarray(mse_values, dtype=float)
    if counts.ndim != 1 or mse.ndim != 1:
        raise ValueError("harmonic counts and MSE values must be one-dimensional")
    if counts.shape != mse.shape:
        raise ValueError("harmonic counts and MSE values must have the same shape")
    if counts.size == 0:
        raise ValueError("harmonic counts and MSE values must not be empty")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(mse)):
        raise ValueError("harmonic counts and MSE values must be finite")

    figure, axes = _get_axes(ax)
    axes.plot(counts, mse, marker="o", markersize=3)
    axes.set_title("Reconstruction MSE vs Number of Harmonics")
    axes.set_xlabel("Number of Harmonics (N)")
    axes.set_ylabel("Mean Squared Error")
    axes.grid(True, alpha=0.3)
    if ax is None:
        figure.tight_layout()
    return figure, axes
