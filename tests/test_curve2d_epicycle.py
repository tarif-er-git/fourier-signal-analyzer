"""Tests for 2D Fourier epicycle geometry."""

import numpy as np
import pytest

from fourier.curve2d_analysis import analyze_curve, reconstruct_curve
from visualization.curve2d_epicycle import Curve2DEpicycle


def coefficient_result() :
    parameter = np.arange(32, dtype=float) / 32.0
    harmonics = np.array([-2, -1, 0, 1, 2])
    x_coefficients = np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=complex)
    y_coefficients = np.array([0.0, 0.0, -0.5, 0.0, 0.0], dtype=complex)
    return type("Result", (), {
        "parameter": parameter,
        "x_samples": np.zeros(32),
        "y_samples": np.zeros(32),
        "harmonics": harmonics,
        "x_coefficients": x_coefficients,
        "y_coefficients": y_coefficients,
    })()


def test_dc_only_frame_stays_at_average_position() -> None:
    model = Curve2DEpicycle(coefficient_result())

    frame = model.frame(0.25, 0)

    assert frame.endpoint == pytest.approx(1.0 - 0.5j)
    assert frame.centers.shape == (1, 2)
    assert frame.endpoints.shape == (1, 2)


def test_single_and_multiple_harmonics_change_chain_size() -> None:
    model = Curve2DEpicycle(coefficient_result())

    assert model.frame(0.0, 1).endpoints.shape == (3, 2)
    assert model.frame(0.0, 2).endpoints.shape == (5, 2)
    np.testing.assert_array_equal(model.selected_harmonics(1), [-1, 0, 1])


def test_endpoint_is_periodic_and_matches_reconstruction() -> None:
    parameter = np.arange(256, dtype=float) / 256.0
    points = np.column_stack((np.cos(2 * np.pi * parameter), np.sin(2 * np.pi * parameter)))
    coefficients = analyze_curve(points, num_samples=256, maximum_harmonic=8)
    model = Curve2DEpicycle(coefficients)

    for value in (0.0, 0.25, 0.5, 0.75):
        endpoint = model.endpoint(value, 1)
        _, x_values, y_values = reconstruct_curve(
            coefficients, harmonic_count=1, parameter=[value]
        )
        assert endpoint.real == pytest.approx(x_values[0], abs=1e-10)
        assert endpoint.imag == pytest.approx(y_values[0], abs=1e-10)
    assert model.endpoint(0.0, 1) == pytest.approx(model.endpoint(1.0, 1))


def test_positive_and_negative_harmonics_are_used() -> None:
    model = Curve2DEpicycle(coefficient_result())
    frame = model.frame(0.125, 2)

    assert frame.radii.shape == (5,)
    assert np.all(np.isfinite(frame.angles))
    assert frame.endpoint == pytest.approx(
        model.endpoint(0.125, 2)
    )


def test_invalid_harmonic_counts_and_empty_coefficients_raise() -> None:
    model = Curve2DEpicycle(coefficient_result())
    with pytest.raises(ValueError):
        model.frame(0.0, 3)
    with pytest.raises(ValueError):
        model.frame(0.0, -1)
    with pytest.raises(ValueError):
        model.frame(1.1, 1)
    empty = type("EmptyResult", (), {
        "harmonics": np.array([], dtype=int),
        "x_coefficients": np.array([], dtype=complex),
        "y_coefficients": np.array([], dtype=complex),
    })()
    with pytest.raises(ValueError):
        Curve2DEpicycle(empty)
