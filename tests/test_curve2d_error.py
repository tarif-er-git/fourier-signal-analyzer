"""Tests for 2D pointwise error and harmonic convergence."""

import numpy as np
import pytest

from fourier.curve2d_analysis import analyze_curve
from fourier.curve2d_error import (
    analyze_curve_convergence,
    evaluate_curve_error,
    pointwise_curve_error,
)


def circle_points(count: int = 128) -> np.ndarray:
    parameter = np.arange(count, dtype=float) / count
    return np.column_stack((np.cos(2 * np.pi * parameter), np.sin(2 * np.pi * parameter)))


def test_identical_curves_have_zero_pointwise_error() -> None:
    values = np.array([0.0, 1.0, -1.0])

    errors = pointwise_curve_error(values, values, values, values)

    np.testing.assert_allclose(errors, 0.0)


def test_translation_has_expected_error_metrics() -> None:
    original_x = np.zeros(4)
    original_y = np.zeros(4)
    reconstructed_x = np.full(4, 3.0)
    reconstructed_y = np.full(4, 4.0)

    errors = pointwise_curve_error(original_x, original_y, reconstructed_x, reconstructed_y)

    np.testing.assert_allclose(errors, 5.0)
    assert np.mean(errors**2) == pytest.approx(25.0)
    assert np.sqrt(np.mean(errors**2)) == pytest.approx(5.0)
    assert np.mean(errors) == pytest.approx(5.0)
    assert np.max(errors) == pytest.approx(5.0)


def test_known_maximum_error_location() -> None:
    errors = pointwise_curve_error(
        np.zeros(3), np.zeros(3), np.array([0.0, 2.0, 1.0]), np.zeros(3)
    )

    assert int(np.argmax(errors)) == 1
    assert errors[1] == pytest.approx(2.0)


def test_convergence_is_ordered_and_reuses_coefficients() -> None:
    coefficients = analyze_curve(circle_points(), num_samples=128, maximum_harmonic=8)
    x_before = coefficients.x_coefficients.copy()

    convergence = analyze_curve_convergence(coefficients, maximum_harmonic=5)

    np.testing.assert_array_equal(convergence.harmonic_counts, np.arange(6))
    assert convergence.for_harmonic(0).harmonic_count == 0
    assert convergence.for_harmonic(5).harmonic_count == 5
    np.testing.assert_array_equal(coefficients.x_coefficients, x_before)


def test_convergence_contains_mean_and_max_error() -> None:
    coefficients = analyze_curve(circle_points(), num_samples=64, maximum_harmonic=4)
    metrics = evaluate_curve_error(coefficients, 1)

    assert metrics.errors.shape == (64,)
    assert metrics.mean_error >= 0.0
    assert metrics.maximum_error >= metrics.mean_error
    assert metrics.maximum_index == int(np.argmax(metrics.errors))


def test_invalid_error_and_convergence_inputs_are_rejected() -> None:
    coefficients = analyze_curve(circle_points(), num_samples=32, maximum_harmonic=4)
    with pytest.raises(ValueError, match="outside"):
        evaluate_curve_error(coefficients, 5)
    with pytest.raises(ValueError, match="outside"):
        analyze_curve_convergence(coefficients, maximum_harmonic=5)
    with pytest.raises(ValueError, match="matching"):
        pointwise_curve_error(np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))
