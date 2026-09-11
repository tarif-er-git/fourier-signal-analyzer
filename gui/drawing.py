"""Data conversion helpers for mouse-drawn one-period signals."""

from collections.abc import Sequence

import numpy as np

from signal.preprocessor import preprocess_signal
from signal.sampler import create_time_grid, prepare_signal_points, resample_signal


def pixel_to_signal_coordinates(
    pixel_x: float,
    pixel_y: float,
    width: float,
    height: float,
    *,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[float, float]:
    """Map drawing-area pixels to ``(time, amplitude)`` coordinates.

    The left and right edges represent ``0`` and ``duration``. The vertical
    center represents zero amplitude; the top and bottom represent positive
    and negative ``amplitude`` respectively.
    """
    values = np.asarray([pixel_x, pixel_y, width, height], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("pixel coordinates and drawing dimensions must be finite")
    if width <= 0 or height <= 0 or duration <= 0 or amplitude <= 0:
        raise ValueError("drawing dimensions, duration, and amplitude must be positive")
    time = np.clip(pixel_x / width, 0.0, 1.0) * duration
    signal = (0.5 - np.clip(pixel_y / height, 0.0, 1.0)) * 2.0 * amplitude
    return float(time), float(signal)


def prepare_custom_signal(
    drawn_t: Sequence[float],
    drawn_x: Sequence[float],
    *,
    num_samples: int = 1001,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw mouse points into a uniform, closed one-period signal.

    Raw points must lie within ``[0, duration]``. They are sorted and duplicate
    times are averaged by :func:`signal.sampler.prepare_signal_points`. Linear
    interpolation then produces an endpoint-inclusive uniform grid. Because a
    Fourier period has one shared left/right boundary, the two interpolated
    endpoint values are replaced by their average. This avoids silently
    choosing one boundary or creating a GUI-specific discontinuity.
    """
    time = np.asarray(drawn_t, dtype=float)
    values = np.asarray(drawn_x, dtype=float)
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("drawn time and signal values must be one-dimensional")
    if time.shape != values.shape:
        raise ValueError("drawn time and signal values must have the same shape")
    if time.size < 2:
        raise ValueError("draw at least two points")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("drawn time and signal values must be finite")
    if duration <= 0 or amplitude <= 0:
        raise ValueError("duration and amplitude must be positive")
    if np.any(time < 0.0) or np.any(time > duration):
        raise ValueError("drawn time values must lie within the drawing period")
    if isinstance(num_samples, bool) or not isinstance(
        num_samples, (int, np.integer)
    ) or num_samples < 2:
        raise ValueError("num_samples must be an integer of at least 2")

    sorted_time, sorted_values = prepare_signal_points(time, values)
    if sorted_time.size < 2:
        raise ValueError("draw at least two points with different time values")

    uniform_time = create_time_grid(0.0, duration, int(num_samples), endpoint=True)
    uniform_values = resample_signal(sorted_time, sorted_values, uniform_time)
    boundary_value = 0.5 * (uniform_values[0] + uniform_values[-1])
    uniform_values[0] = boundary_value
    uniform_values[-1] = boundary_value
    return preprocess_signal(uniform_time, uniform_values, periodic=False)
