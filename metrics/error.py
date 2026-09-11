"""Numerical error metrics for comparing sampled signals."""

from collections.abc import Sequence

import numpy as np


def _validate_signal_pair(
    original: Sequence[float], reconstructed: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated floating-point arrays with matching non-empty shapes."""
    original_values = np.asarray(original, dtype=float)
    reconstructed_values = np.asarray(reconstructed, dtype=float)
    if original_values.shape != reconstructed_values.shape:
        raise ValueError("original and reconstructed must have the same shape")
    if original_values.size == 0:
        raise ValueError("signals must not be empty")
    if not np.all(np.isfinite(original_values)):
        raise ValueError("original must contain only finite values")
    if not np.all(np.isfinite(reconstructed_values)):
        raise ValueError("reconstructed must contain only finite values")
    return original_values, reconstructed_values


def calculate_error(
    original: Sequence[float], reconstructed: Sequence[float]
) -> np.ndarray:
    """Calculate the signed point-wise error ``original - reconstructed``."""
    original_values, reconstructed_values = _validate_signal_pair(
        original, reconstructed
    )
    return original_values - reconstructed_values


def calculate_mse(
    original: Sequence[float], reconstructed: Sequence[float]
) -> float:
    """Calculate mean squared error over all signal samples."""
    error = calculate_error(original, reconstructed)
    return float(np.mean(error**2))


def calculate_rmse(
    original: Sequence[float], reconstructed: Sequence[float]
) -> float:
    """Calculate root mean squared error over all signal samples."""
    return float(np.sqrt(calculate_mse(original, reconstructed)))


def calculate_max_error(
    original: Sequence[float], reconstructed: Sequence[float]
) -> float:
    """Calculate the largest absolute point-wise error."""
    error = calculate_error(original, reconstructed)
    return float(np.max(np.abs(error)))


def calculate_mae(
    original: Sequence[float], reconstructed: Sequence[float]
) -> float:
    """Calculate mean absolute error over all signal samples."""
    error = calculate_error(original, reconstructed)
    return float(np.mean(np.abs(error)))
