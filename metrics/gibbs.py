"""Numerical measurements for Gibbs behavior near jump discontinuities."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

GIBBS_OVERSHOOT_FRACTION = 0.089489872236


@dataclass(frozen=True)
class GibbsResult:
    """Measured Gibbs information for one likely discontinuity."""

    overshoot: float | None
    undershoot: float | None
    max_local_error: float | None
    discontinuity_index: int | None
    discontinuity_location: float | None
    jump_magnitude: float | None
    oscillation_width: float | None
    detected: bool
    message: str


def _validate_signal_inputs(
    t: Sequence[float],
    original: Sequence[float],
    reconstructed: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate time, original, and reconstructed signal arrays."""
    time = np.asarray(t, dtype=float)
    original_values = np.asarray(original, dtype=float)
    reconstructed_values = np.asarray(reconstructed, dtype=float)
    if time.ndim != 1 or original_values.ndim != 1 or reconstructed_values.ndim != 1:
        raise ValueError("t, original, and reconstructed must be one-dimensional")
    if time.size < 3:
        raise ValueError("at least three samples are required")
    if time.shape != original_values.shape or time.shape != reconstructed_values.shape:
        raise ValueError("t, original, and reconstructed must have the same shape")
    if not np.all(np.isfinite(time)):
        raise ValueError("t must contain only finite values")
    if not np.all(np.isfinite(original_values)):
        raise ValueError("original must contain only finite values")
    if not np.all(np.isfinite(reconstructed_values)):
        raise ValueError("reconstructed must contain only finite values")
    if not np.all(np.diff(time) > 0):
        raise ValueError("t must be strictly increasing")
    return time, original_values, reconstructed_values


def detect_discontinuity(
    t: Sequence[float],
    signal: Sequence[float],
    *,
    threshold_factor: float = 8.0,
) -> int | None:
    """Return the left index of a likely jump, or ``None``.

    The largest adjacent change is compared with the median adjacent change.
    This simple detector works well for square-like presets, but steep smooth
    slopes and noisy data can produce false positives or missed jumps. A
    manually supplied index can be passed to :func:`analyze_gibbs` when needed.
    """
    time = np.asarray(t, dtype=float)
    values = np.asarray(signal, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.shape != values.shape:
        raise ValueError("t and signal must be matching one-dimensional arrays")
    if time.size < 3:
        raise ValueError("at least three samples are required")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("t and signal must contain only finite values")
    if not np.all(np.diff(time) > 0):
        raise ValueError("t must be strictly increasing")
    if not np.isfinite(threshold_factor) or threshold_factor <= 0:
        raise ValueError("threshold_factor must be positive and finite")

    changes = np.abs(np.diff(values))
    largest_index = int(np.argmax(changes))
    typical_change = float(np.median(changes))
    threshold = max(typical_change * threshold_factor, 1e-12)
    if changes[largest_index] <= threshold:
        return None
    return largest_index


def theoretical_gibbs_overshoot(jump_magnitude: float) -> float:
    """Return the theoretical 9% Gibbs overshoot reference for a jump."""
    jump = float(jump_magnitude)
    if not np.isfinite(jump) or jump < 0:
        raise ValueError("jump_magnitude must be finite and non-negative")
    return GIBBS_OVERSHOOT_FRACTION * jump


def analyze_gibbs(
    t: Sequence[float],
    original: Sequence[float],
    reconstructed: Sequence[float],
    *,
    discontinuity_index: int | None = None,
    window_fraction: float = 0.05,
    threshold_factor: float = 8.0,
) -> GibbsResult:
    """Measure local overshoot and undershoot near a likely discontinuity.

    ``window_fraction`` is the fraction of the sampled period inspected on
    each side of the candidate jump. The reported local error and oscillation
    width are intentionally local; they do not summarize the whole signal.
    If no significant jump is detected, Gibbs measurements are ``None``.
    """
    time, original_values, reconstructed_values = _validate_signal_inputs(
        t, original, reconstructed
    )
    if not np.isfinite(window_fraction) or not 0 < window_fraction <= 0.5:
        raise ValueError("window_fraction must be in the interval (0, 0.5]")

    if discontinuity_index is None:
        candidate_index = detect_discontinuity(
            time, original_values, threshold_factor=threshold_factor
        )
    else:
        if isinstance(discontinuity_index, bool) or not isinstance(
            discontinuity_index, (int, np.integer)
        ):
            raise TypeError("discontinuity_index must be an integer")
        candidate_index = int(discontinuity_index)
        if not 0 <= candidate_index < time.size - 1:
            raise ValueError("discontinuity_index must refer to an adjacent sample pair")

    if candidate_index is None:
        return GibbsResult(
            overshoot=None,
            undershoot=None,
            max_local_error=None,
            discontinuity_index=None,
            discontinuity_location=None,
            jump_magnitude=None,
            oscillation_width=None,
            detected=False,
            message="No significant discontinuity detected",
        )

    samples_per_side = max(1, int(round(window_fraction * (time.size - 1))))
    start = max(0, candidate_index - samples_per_side)
    end = min(time.size, candidate_index + 1 + samples_per_side)
    left_baseline = original_values[start:candidate_index]
    right_baseline = original_values[candidate_index + 1 : end]
    if left_baseline.size == 0 or right_baseline.size == 0:
        raise ValueError("discontinuity must have samples on both sides")

    left_level = float(np.median(left_baseline))
    right_level = float(np.median(right_baseline))
    jump_magnitude = abs(right_level - left_level)
    local_reconstruction = reconstructed_values[start:end]
    upper_level = max(left_level, right_level)
    lower_level = min(left_level, right_level)
    overshoot = max(0.0, float(np.max(local_reconstruction)) - upper_level)
    undershoot = max(0.0, lower_level - float(np.min(local_reconstruction)))
    local_error = np.abs(original_values[start:end] - local_reconstruction)
    max_local_error = float(np.max(local_error))

    error_threshold = max(0.1 * jump_magnitude, 1e-12)
    error_indices = np.flatnonzero(local_error >= error_threshold)
    if error_indices.size >= 2:
        oscillation_width = float(
            time[start + error_indices[-1]] - time[start + error_indices[0]]
        )
    else:
        oscillation_width = 0.0

    return GibbsResult(
        overshoot=overshoot,
        undershoot=undershoot,
        max_local_error=max_local_error,
        discontinuity_index=candidate_index,
        discontinuity_location=float(
            0.5 * (time[candidate_index] + time[candidate_index + 1])
        ),
        jump_magnitude=jump_magnitude,
        oscillation_width=oscillation_width,
        detected=True,
        message="Discontinuity detected",
    )
