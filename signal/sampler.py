"""Utilities for preparing signals on uniform time grids."""

from collections.abc import Sequence

import numpy as np


def create_time_grid(
    start: float, end: float, num_samples: int, endpoint: bool = True
) -> np.ndarray:
    """Create an evenly spaced one-dimensional time grid.

    Parameters
    ----------
    start, end
        Start and end times. ``end`` must be greater than ``start``.
    num_samples
        Number of samples to create. At least two samples are required.
    endpoint
        If true, include ``end`` as the final sample, matching
        :func:`numpy.linspace`.

    Returns
    -------
    numpy.ndarray
        A floating-point array of evenly spaced time values.
    """
    start_value = float(start)
    end_value = float(end)
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        raise ValueError("start and end must be finite")
    if end_value <= start_value:
        raise ValueError("end must be greater than start")
    if isinstance(num_samples, bool) or num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if not isinstance(num_samples, (int, np.integer)):
        raise TypeError("num_samples must be an integer")

    return np.linspace(start_value, end_value, int(num_samples), endpoint=endpoint)


def prepare_signal_points(
    t_original: Sequence[float], x_original: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, sort, and de-duplicate arbitrary signal points.

    Duplicate time values are replaced by one time value whose signal value
    is their arithmetic mean. This gives mouse-drawn data a deterministic
    representation before interpolation.

    Parameters
    ----------
    t_original, x_original
        One-dimensional finite arrays containing time and signal samples.
        At least two samples are required.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Strictly increasing time values and the corresponding signal values.
    """
    time = np.asarray(t_original, dtype=float)
    values = np.asarray(x_original, dtype=float)
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and signal inputs must be one-dimensional")
    if time.size != values.size:
        raise ValueError("time and signal inputs must have the same length")
    if time.size < 2:
        raise ValueError("at least two signal points are required")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("time and signal inputs must contain only finite values")

    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    sorted_values = values[order]
    unique_time, group_indices = np.unique(sorted_time, return_inverse=True)
    if unique_time.size == sorted_time.size:
        return unique_time, sorted_values

    value_sums = np.zeros(unique_time.size, dtype=float)
    value_counts = np.zeros(unique_time.size, dtype=int)
    np.add.at(value_sums, group_indices, sorted_values)
    np.add.at(value_counts, group_indices, 1)
    return unique_time, value_sums / value_counts


def resample_signal(
    t_original: Sequence[float],
    x_original: Sequence[float],
    t_uniform: Sequence[float],
) -> np.ndarray:
    """Interpolate signal samples onto a target time grid.

    The original points are sorted and duplicate times are averaged by
    :func:`prepare_signal_points`. Linear interpolation is used between
    points. Outside the original time range, the nearest endpoint value is
    held constant, which is NumPy's default ``interp`` behavior.

    Parameters
    ----------
    t_original, x_original
        One-dimensional finite source time and signal samples.
    t_uniform
        One-dimensional finite target times. They need not be sorted.

    Returns
    -------
    numpy.ndarray
        Interpolated signal values with the same shape as ``t_uniform``.
    """
    target_time = np.asarray(t_uniform, dtype=float)
    if target_time.ndim != 1:
        raise ValueError("t_uniform must be one-dimensional")
    if not np.all(np.isfinite(target_time)):
        raise ValueError("t_uniform must contain only finite values")

    source_time, source_values = prepare_signal_points(t_original, x_original)
    return np.interp(target_time, source_time, source_values)
