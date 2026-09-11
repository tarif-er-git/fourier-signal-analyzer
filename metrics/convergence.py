"""Reconstruction accuracy across a range of Fourier harmonic counts."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from fourier.synthesis import FourierSynthesizer
from .error import calculate_max_error, calculate_mse, calculate_rmse


@dataclass(frozen=True)
class ConvergenceResult:
    """Accuracy metrics paired with each requested harmonic count."""

    harmonic_counts: np.ndarray
    mse_values: np.ndarray
    rmse_values: np.ndarray
    max_error_values: np.ndarray


def analyze_convergence(
    original: Sequence[float],
    synthesizer: FourierSynthesizer,
    harmonic_counts: Sequence[int] | None = None,
) -> ConvergenceResult:
    """Calculate MSE, RMSE, and maximum error for several values of N.

    Error generally improves as N increases for suitable signals, but no
    strict point-by-point monotonic decrease is assumed, especially near
    discontinuities.
    """
    original_values = np.asarray(original, dtype=float)
    if original_values.ndim != 1 or original_values.size == 0:
        raise ValueError("original must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(original_values)):
        raise ValueError("original must contain only finite values")
    if original_values.shape != synthesizer.t.shape:
        raise ValueError("original and synthesizer time values must have the same shape")

    if harmonic_counts is None:
        counts = np.arange(1, synthesizer.available_harmonics + 1, dtype=int)
    else:
        counts = np.asarray(harmonic_counts)
        if counts.ndim != 1 or counts.size == 0:
            raise ValueError("harmonic_counts must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(counts)):
            raise ValueError("harmonic_counts must contain finite values")
        if not np.all(np.equal(counts, counts.astype(int))):
            raise ValueError("harmonic_counts must contain integers")
        counts = counts.astype(int)
        if np.any(counts <= 0) or np.any(counts > synthesizer.available_harmonics):
            raise ValueError("harmonic_counts must be within the available harmonics")

    mse_values = np.empty(counts.size, dtype=float)
    rmse_values = np.empty(counts.size, dtype=float)
    max_error_values = np.empty(counts.size, dtype=float)
    for index, count in enumerate(counts):
        reconstructed = synthesizer.reconstruct(int(count))
        mse_values[index] = calculate_mse(original_values, reconstructed)
        rmse_values[index] = calculate_rmse(original_values, reconstructed)
        max_error_values[index] = calculate_max_error(original_values, reconstructed)

    return ConvergenceResult(
        harmonic_counts=counts,
        mse_values=mse_values,
        rmse_values=rmse_values,
        max_error_values=max_error_values,
    )
