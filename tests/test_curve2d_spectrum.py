"""Tests for 2D harmonic spectrum display data."""

import numpy as np
import pytest

from fourier.curve2d_analysis import CurveFourierResult, analyze_curve
from visualization.curve2d_spectrum import (
    build_spectrum_rows,
    dominant_harmonics,
    sort_spectrum_rows,
)


def synthetic_result() -> CurveFourierResult:
    return CurveFourierResult(
        parameter=np.array([0.0, 0.5]),
        x_samples=np.zeros(2),
        y_samples=np.zeros(2),
        harmonics=np.array([-1, 0, 1]),
        x_coefficients=np.array([1j, 2.0, -1j]),
        y_coefficients=np.array([2.0, -1.0, 0.5j]),
    )


def test_spectrum_rows_calculate_magnitude_phase_and_combined_values() -> None:
    rows = build_spectrum_rows(synthetic_result())
    negative = rows[0]
    dc = rows[1]

    assert negative.harmonic == -1
    assert negative.x_magnitude == pytest.approx(1.0)
    assert negative.x_phase == pytest.approx(np.pi / 2.0)
    assert negative.y_magnitude == pytest.approx(2.0)
    assert negative.y_phase == pytest.approx(0.0)
    assert negative.combined_magnitude == pytest.approx(np.sqrt(5.0))
    assert dc.harmonic == 0
    assert dc.x_magnitude == pytest.approx(2.0)
    assert dc.y_magnitude == pytest.approx(1.0)


def test_spectrum_sorting_and_dominant_harmonics() -> None:
    rows = build_spectrum_rows(synthetic_result())

    assert [row.harmonic for row in sort_spectrum_rows(rows)] == [-1, 0, 1]
    magnitude_order = sort_spectrum_rows(rows, "combined_magnitude")
    assert magnitude_order[0].harmonic == -1
    assert [row.harmonic for row in dominant_harmonics(rows, count=2)] == [-1, 0]


def test_circle_and_ellipse_have_coordinate_specific_magnitudes() -> None:
    parameter = np.arange(256, dtype=float) / 256.0
    circle = np.column_stack((np.cos(2 * np.pi * parameter), np.sin(2 * np.pi * parameter)))
    ellipse = np.column_stack((2.0 * np.cos(2 * np.pi * parameter), np.sin(2 * np.pi * parameter)))
    circle_rows = build_spectrum_rows(analyze_curve(circle, maximum_harmonic=4))
    ellipse_rows = build_spectrum_rows(analyze_curve(ellipse, maximum_harmonic=4))
    circle_positive = next(row for row in circle_rows if row.harmonic == 1)
    ellipse_positive = next(row for row in ellipse_rows if row.harmonic == 1)

    assert ellipse_positive.x_magnitude > circle_positive.x_magnitude
    assert ellipse_positive.y_magnitude > 0.0
    assert circle_positive.y_magnitude > 0.0


def test_empty_or_invalid_spectrum_data_is_rejected() -> None:
    empty = CurveFourierResult(
        parameter=np.array([]), x_samples=np.array([]), y_samples=np.array([]),
        harmonics=np.array([], dtype=int), x_coefficients=np.array([], dtype=complex),
        y_coefficients=np.array([], dtype=complex),
    )
    with pytest.raises(ValueError, match="empty"):
        build_spectrum_rows(empty)
    with pytest.raises(ValueError, match="sort_by"):
        sort_spectrum_rows(build_spectrum_rows(synthetic_result()), "invalid")
    with pytest.raises(ValueError, match="positive"):
        dominant_harmonics(build_spectrum_rows(synthetic_result()), count=0)
