"""Pointwise error and harmonic convergence analysis for 2D curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourier.curve2d_analysis import (
    CurveFourierResult,
    pointwise_curve_error,
    reconstruct_curve,
)


@dataclass(frozen=True)
class CurveErrorMetrics:
    """Pointwise geometric errors and aggregate metrics for one harmonic count."""

    parameter: np.ndarray
    errors: np.ndarray
    mse: float
    rmse: float
    mean_error: float
    maximum_error: float
    maximum_index: int
    harmonic_count: int
    reconstructed_x: np.ndarray
    reconstructed_y: np.ndarray

    @property
    def maximum_parameter(self) -> float:
        """Return the normalized parameter at the largest sampled error."""
        return float(self.parameter[self.maximum_index])


@dataclass(frozen=True)
class CurveConvergenceResult:
    """Cached error metrics ordered by harmonic count."""

    harmonic_counts: np.ndarray
    metrics: tuple[CurveErrorMetrics, ...]

    def for_harmonic(self, harmonic_count: int) -> CurveErrorMetrics:
        """Return cached metrics for one harmonic count."""
        matches = np.flatnonzero(self.harmonic_counts == harmonic_count)
        if matches.size == 0:
            raise ValueError("harmonic count is not present in convergence results")
        return self.metrics[int(matches[0])]


def evaluate_curve_error(
    coefficients: CurveFourierResult,
    harmonic_count: int,
) -> CurveErrorMetrics:
    """Reconstruct once and calculate all pointwise and aggregate errors."""
    parameter, reconstructed_x, reconstructed_y = reconstruct_curve(
        coefficients, harmonic_count=harmonic_count
    )
    errors = pointwise_curve_error(
        coefficients.x_samples,
        coefficients.y_samples,
        reconstructed_x,
        reconstructed_y,
    )
    maximum_index = int(np.argmax(errors))
    squared_errors = errors**2
    mse = float(np.mean(squared_errors))
    return CurveErrorMetrics(
        parameter=parameter,
        errors=errors,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mean_error=float(np.mean(errors)),
        maximum_error=float(errors[maximum_index]),
        maximum_index=maximum_index,
        harmonic_count=int(harmonic_count),
        reconstructed_x=reconstructed_x,
        reconstructed_y=reconstructed_y,
    )


def analyze_curve_convergence(
    coefficients: CurveFourierResult,
    *,
    maximum_harmonic: int | None = None,
) -> CurveConvergenceResult:
    """Calculate and cache metrics for N=0 through the selected maximum N."""
    maximum = coefficients.maximum_harmonic if maximum_harmonic is None else maximum_harmonic
    if isinstance(maximum, bool) or not isinstance(maximum, (int, np.integer)):
        raise ValueError("maximum_harmonic must be an integer")
    if maximum < 0 or maximum > coefficients.maximum_harmonic:
        raise ValueError("maximum_harmonic is outside the available range")
    counts = np.arange(int(maximum) + 1, dtype=int)
    metrics = tuple(evaluate_curve_error(coefficients, int(count)) for count in counts)
    return CurveConvergenceResult(counts, metrics)
