"""Tests for Fourier analysis of closed 2D curves."""

import numpy as np
import pytest

from fourier.curve2d_analysis import (
    analyze_curve,
    calculate_curve_error,
    parameterize_curve,
    reconstruct_curve,
    resample_curve,
)
from signal.curve2d import Curve2D


def circle_points(count: int = 512) -> np.ndarray:
    parameter = np.arange(count, dtype=float) / count
    return np.column_stack((np.cos(2 * np.pi * parameter), np.sin(2 * np.pi * parameter)))


def test_arc_length_parameterization_includes_closing_segment() -> None:
    points, parameter, length = parameterize_curve(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
    )

    np.testing.assert_array_equal(points, [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(parameter, [0.0, 3.0 / 12.0, 7.0 / 12.0])
    assert length == pytest.approx(12.0)


def test_resampling_uses_uniform_endpoint_excluded_parameter() -> None:
    parameter, x_values, y_values = resample_curve(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        num_samples=8,
    )

    np.testing.assert_allclose(parameter, np.arange(8) / 8)
    assert parameter[-1] < 1.0
    np.testing.assert_allclose([x_values[0], y_values[0]], [0.0, 0.0])
    np.testing.assert_allclose([x_values[4], y_values[4]], [2.0, 2.0])


def test_circle_has_expected_first_harmonic_and_reconstructs() -> None:
    result = analyze_curve(circle_points(), num_samples=512, maximum_harmonic=4)

    index_positive = np.flatnonzero(result.harmonics == 1)[0]
    index_negative = np.flatnonzero(result.harmonics == -1)[0]
    assert result.x_coefficients[index_positive] == pytest.approx(0.5, abs=2e-4)
    assert result.y_coefficients[index_positive] == pytest.approx(-0.5j, abs=2e-4)
    assert result.x_coefficients[index_negative] == pytest.approx(0.5, abs=2e-4)
    assert result.y_coefficients[index_negative] == pytest.approx(0.5j, abs=2e-4)

    reconstruction = calculate_curve_error(result, harmonic_count=1)
    assert reconstruction.rmse < 2e-3
    assert reconstruction.maximum_error < 3e-3


def test_ellipse_preserves_dc_and_reconstructs_with_first_harmonic() -> None:
    parameter = np.arange(512, dtype=float) / 512
    points = np.column_stack((2.0 * np.cos(2 * np.pi * parameter) + 3.0,
                              np.sin(2 * np.pi * parameter) - 2.0))
    result = analyze_curve(points, num_samples=512, maximum_harmonic=16)

    dc = np.flatnonzero(result.harmonics == 0)[0]
    assert result.x_coefficients[dc] == pytest.approx(3.0, abs=2e-3)
    assert result.y_coefficients[dc] == pytest.approx(-2.0, abs=2e-3)
    reconstruction = calculate_curve_error(result, harmonic_count=8)
    assert reconstruction.rmse < 0.01


def test_simple_harmonic_curve_has_expected_signed_coefficients() -> None:
    parameter = np.arange(512, dtype=float) / 512
    points = np.column_stack((1.0 + np.cos(2 * np.pi * parameter),
                              -0.5 + np.sin(2 * np.pi * parameter)))
    result = analyze_curve(points, num_samples=512, maximum_harmonic=3)
    positive = np.flatnonzero(result.harmonics == 1)[0]
    dc = np.flatnonzero(result.harmonics == 0)[0]

    assert result.x_coefficients[dc] == pytest.approx(1.0, abs=2e-3)
    assert result.y_coefficients[dc] == pytest.approx(-0.5, abs=2e-3)
    assert result.x_coefficients[positive] == pytest.approx(0.5, abs=2e-3)
    assert result.y_coefficients[positive] == pytest.approx(-0.5j, abs=2e-3)


def test_reconstruction_harmonic_selection_and_error_metrics() -> None:
    result = analyze_curve(circle_points(), num_samples=128, maximum_harmonic=8)

    _, x_dc, y_dc = reconstruct_curve(result, harmonic_count=0)
    assert np.allclose(x_dc, result.x_coefficients[result.harmonics == 0][0].real)
    assert np.allclose(y_dc, result.y_coefficients[result.harmonics == 0][0].real)
    with pytest.raises(ValueError, match="outside"):
        reconstruct_curve(result, harmonic_count=9)
    error = calculate_curve_error(result, harmonic_count=1)
    assert error.mse == pytest.approx(error.rmse**2)
    assert error.maximum_error >= 0.0


def test_reconstruction_supports_zero_one_and_multiple_harmonics() -> None:
    result = analyze_curve(circle_points(), num_samples=128, maximum_harmonic=8)
    original_x_coefficients = result.x_coefficients.copy()
    original_y_coefficients = result.y_coefficients.copy()

    zero = calculate_curve_error(result, harmonic_count=0)
    one = calculate_curve_error(result, harmonic_count=1)
    several = calculate_curve_error(result, harmonic_count=5)

    assert zero.x.shape == (128,)
    assert one.y.shape == (128,)
    assert several.x.shape == (128,)
    assert one.rmse < zero.rmse
    assert several.rmse <= one.rmse + 1e-12
    np.testing.assert_array_equal(result.x_coefficients, original_x_coefficients)
    np.testing.assert_array_equal(result.y_coefficients, original_y_coefficients)


def test_invalid_and_degenerate_curves_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least three"):
        analyze_curve([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="at least three"):
        analyze_curve([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="num_samples"):
        analyze_curve(circle_points(), num_samples=3)
    with pytest.raises(ValueError, match="maximum_harmonic"):
        analyze_curve(circle_points(), maximum_harmonic=True)


def test_curve_model_input_is_supported() -> None:
    curve = Curve2D.from_points(circle_points(32))
    result = analyze_curve(curve, num_samples=64, maximum_harmonic=4)

    assert result.point_count == 64
    assert result.harmonics[0] == -4
