"""Tests for 2D closed-curve data cleanup and normalization."""

import numpy as np
import pytest

from signal.curve2d import Curve2D


def test_empty_curve_is_invalid() -> None:
    curve = Curve2D.from_points([])

    assert curve.is_valid is False
    assert curve.point_count == 0
    assert curve.bounding_box is None
    assert curve.closed_points.shape == (0, 2)


def test_one_or_two_points_are_insufficient() -> None:
    assert Curve2D.from_points([[0.0, 0.0]]).is_valid is False
    assert Curve2D.from_points([[0.0, 0.0], [1.0, 0.0]]).is_valid is False


def test_consecutive_duplicates_are_removed() -> None:
    curve = Curve2D.from_points(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )

    np.testing.assert_allclose(
        curve.cleaned_points, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    assert curve.point_count == 3


def test_valid_curve_closes_logically_without_duplicate_storage() -> None:
    curve = Curve2D.from_points(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )

    assert curve.is_valid is True
    assert curve.is_closed is True
    assert curve.cleaned_points.shape == (4, 2)
    assert curve.closed_points.shape == (5, 2)
    np.testing.assert_allclose(curve.closed_points[0], curve.closed_points[-1])


def test_bounding_box_and_normalization() -> None:
    curve = Curve2D.from_points(
        [[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]]
    )

    assert curve.bounding_box == (2.0, 6.0, 4.0, 8.0)
    np.testing.assert_allclose(curve.normalized_points.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(curve.normalized_points.max(axis=0), [1.0, 1.0])


def test_raw_points_are_preserved() -> None:
    raw = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    curve = Curve2D.from_points(raw)

    np.testing.assert_array_equal(curve.raw_points, raw)
    assert curve.raw_points is not curve.cleaned_points


def test_nonfinite_points_are_removed_from_processed_curve() -> None:
    curve = Curve2D.from_points(
        [[0.0, 0.0], [np.nan, 2.0], [1.0, 0.0], [np.inf, 4.0], [0.0, 1.0]]
    )

    assert curve.is_valid is True
    assert np.all(np.isfinite(curve.cleaned_points))
    assert curve.point_count == 3


def test_malformed_point_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        Curve2D.from_points([[0.0, 1.0, 2.0]])
