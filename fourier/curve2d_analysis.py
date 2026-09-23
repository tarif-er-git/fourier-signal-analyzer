"""Fourier analysis utilities for periodic 2D curves."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from signal.curve2d import Curve2D


@dataclass(frozen=True)
class CurveFourierResult:
    """Fourier coefficients and uniformly parameterized samples of a curve."""

    parameter: np.ndarray
    x_samples: np.ndarray
    y_samples: np.ndarray
    harmonics: np.ndarray
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray

    @property
    def x_magnitudes(self) -> np.ndarray:
        """Return magnitudes of the X coefficients."""
        return np.abs(self.x_coefficients)

    @property
    def y_magnitudes(self) -> np.ndarray:
        """Return magnitudes of the Y coefficients."""
        return np.abs(self.y_coefficients)

    @property
    def x_phases(self) -> np.ndarray:
        """Return phases of the X coefficients in radians.

        Coefficients with near-zero magnitude receive phase zero by convention.
        """
        phases = np.angle(self.x_coefficients)
        return np.where(np.isclose(self.x_magnitudes, 0.0, atol=1e-12), 0.0, phases)

    @property
    def y_phases(self) -> np.ndarray:
        """Return phases of the Y coefficients in radians.

        Coefficients with near-zero magnitude receive phase zero by convention.
        """
        phases = np.angle(self.y_coefficients)
        return np.where(np.isclose(self.y_magnitudes, 0.0, atol=1e-12), 0.0, phases)

    @property
    def point_count(self) -> int:
        """Return the number of uniform samples used for analysis."""
        return int(self.parameter.size)

    @property
    def maximum_harmonic(self) -> int:
        """Return the largest absolute stored harmonic index."""
        return int(np.max(np.abs(self.harmonics)))


@dataclass(frozen=True)
class CurveReconstruction:
    """Reconstructed X/Y samples and geometric error against the source."""

    parameter: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mse: float
    rmse: float
    maximum_error: float
    harmonic_count: int


def parameterize_curve(
    curve_or_points: Curve2D | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cleaned points, normalized closed arc length, and total length.

    The returned parameter has one value per stored vertex and excludes the
    duplicated endpoint. The closing segment from the last vertex to the first
    is included in the total length.
    """
    points = _curve_points(curve_or_points)
    _validate_points(points)
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(np.sum(lengths))
    if not np.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("curve must have positive total arc length")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    parameter = cumulative / total_length
    return points, parameter, total_length


def resample_curve(
    curve_or_points: Curve2D | Sequence[Sequence[float]],
    *,
    num_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a closed curve uniformly by normalized arc length."""
    if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)):
        raise ValueError("num_samples must be an integer")
    if num_samples < 4:
        raise ValueError("num_samples must be at least 4")
    points, parameter, _ = parameterize_curve(curve_or_points)
    uniform_parameter = np.arange(int(num_samples), dtype=float) / int(num_samples)
    periodic_parameter = np.concatenate((parameter, [1.0]))
    periodic_x = np.concatenate((points[:, 0], [points[0, 0]]))
    periodic_y = np.concatenate((points[:, 1], [points[0, 1]]))
    x_samples = np.interp(uniform_parameter, periodic_parameter, periodic_x)
    y_samples = np.interp(uniform_parameter, periodic_parameter, periodic_y)
    return uniform_parameter, x_samples, y_samples


def analyze_curve(
    curve_or_points: Curve2D | Sequence[Sequence[float]],
    *,
    num_samples: int = 512,
    maximum_harmonic: int | None = None,
) -> CurveFourierResult:
    """Calculate signed complex Fourier coefficients for both curve coordinates."""
    parameter, x_samples, y_samples = resample_curve(
        curve_or_points, num_samples=num_samples
    )
    sample_count = parameter.size
    x_fft = np.fft.fft(x_samples) / sample_count
    y_fft = np.fft.fft(y_samples) / sample_count
    fft_harmonics = np.fft.fftfreq(sample_count, d=1.0 / sample_count).astype(int)
    if maximum_harmonic is None:
        maximum_harmonic = sample_count // 2
    if isinstance(maximum_harmonic, bool) or not isinstance(
        maximum_harmonic, (int, np.integer)
    ):
        raise ValueError("maximum_harmonic must be an integer")
    if maximum_harmonic < 0 or maximum_harmonic > sample_count // 2:
        raise ValueError("maximum_harmonic must be between 0 and num_samples // 2")
    selected = np.abs(fft_harmonics) <= int(maximum_harmonic)
    order = np.argsort(fft_harmonics[selected])
    return CurveFourierResult(
        parameter=parameter,
        x_samples=x_samples,
        y_samples=y_samples,
        harmonics=fft_harmonics[selected][order],
        x_coefficients=x_fft[selected][order],
        y_coefficients=y_fft[selected][order],
    )


def reconstruct_curve(
    coefficients: CurveFourierResult,
    *,
    harmonic_count: int,
    parameter: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a curve with DC plus symmetric harmonics ``-N`` through ``N``."""
    if isinstance(harmonic_count, bool) or not isinstance(
        harmonic_count, (int, np.integer)
    ):
        raise ValueError("harmonic_count must be an integer")
    if harmonic_count < 0 or harmonic_count > coefficients.maximum_harmonic:
        raise ValueError("harmonic_count is outside the available harmonic range")
    if parameter is None:
        parameter_values = coefficients.parameter
    else:
        parameter_values = np.asarray(parameter, dtype=float)
        if parameter_values.ndim != 1 or not np.all(np.isfinite(parameter_values)):
            raise ValueError("parameter must be a finite one-dimensional array")
    selected = np.abs(coefficients.harmonics) <= int(harmonic_count)
    phase = np.exp(
        2j * np.pi * parameter_values[:, np.newaxis] * coefficients.harmonics[selected]
    )
    x_values = np.real(phase @ coefficients.x_coefficients[selected])
    y_values = np.real(phase @ coefficients.y_coefficients[selected])
    return parameter_values.copy(), x_values, y_values


def calculate_curve_error(
    coefficients: CurveFourierResult,
    *,
    harmonic_count: int,
) -> CurveReconstruction:
    """Reconstruct a curve and calculate MSE, RMSE, and maximum distance error."""
    parameter, x_values, y_values = reconstruct_curve(
        coefficients, harmonic_count=harmonic_count
    )
    errors = pointwise_curve_error(
        coefficients.x_samples,
        coefficients.y_samples,
        x_values,
        y_values,
    )
    squared_distance = errors**2
    mse = float(np.mean(squared_distance))
    return CurveReconstruction(
        parameter=parameter,
        x=x_values,
        y=y_values,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        maximum_error=float(np.sqrt(np.max(squared_distance))),
        harmonic_count=int(harmonic_count),
    )


def pointwise_curve_error(
    original_x: Sequence[float],
    original_y: Sequence[float],
    reconstructed_x: Sequence[float],
    reconstructed_y: Sequence[float],
) -> np.ndarray:
    """Return Euclidean error for corresponding 2D curve samples."""
    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (original_x, original_y, reconstructed_x, reconstructed_y)
    )
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("curve coordinates must be one-dimensional")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape == arrays[3].shape):
        raise ValueError("curve coordinate arrays must have matching shapes")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("curve coordinates must be finite")
    return np.hypot(arrays[0] - arrays[2], arrays[1] - arrays[3])


def _curve_points(
    curve_or_points: Curve2D | Sequence[Sequence[float]],
) -> np.ndarray:
    """Extract a copy of cleaned points from a model or point sequence."""
    if isinstance(curve_or_points, Curve2D):
        return curve_or_points.cleaned_points.copy()
    return Curve2D.from_points(curve_or_points).cleaned_points


def _validate_points(points: np.ndarray) -> None:
    """Validate the minimum geometry required for a closed curve."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("curve points must have shape (n, 2)")
    if points.shape[0] < 3:
        raise ValueError("a closed curve requires at least three points")
    if not np.all(np.isfinite(points)):
        raise ValueError("curve points must be finite")
