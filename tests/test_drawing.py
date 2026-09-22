"""Tests for mouse-drawing coordinate and sampling helpers."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from gui.drawing import (
    normalize_drawn_coordinates,
    pixel_to_signal_coordinates,
    prepare_custom_signal,
    process_freehand_points,
)


def test_pixel_conversion_maps_edges_and_center() -> None:
    assert pixel_to_signal_coordinates(0.0, 50.0, 100.0, 100.0) == (0.0, 0.0)
    assert pixel_to_signal_coordinates(100.0, 0.0, 100.0, 100.0) == (1.0, 1.0)
    assert pixel_to_signal_coordinates(100.0, 100.0, 100.0, 100.0) == (1.0, -1.0)


def test_raw_points_are_sorted_deduplicated_and_uniformly_sampled() -> None:
    drawn_t = np.array([0.8, 0.2, 0.2, 0.5])
    drawn_x = np.array([-0.5, 0.2, 0.4, 0.0])

    time, values = prepare_custom_signal(drawn_t, drawn_x, num_samples=101)

    assert time.shape == (101,)
    assert values.shape == (101,)
    np.testing.assert_allclose(np.diff(time), time[1] - time[0])
    assert values[0] == pytest.approx(values[-1])
    assert np.all(np.isfinite(values))


def test_custom_signal_can_enter_fourier_analyzer() -> None:
    drawn_t = np.linspace(0.0, 1.0, 21)
    drawn_x = np.sin(2.0 * np.pi * drawn_t)
    time, values = prepare_custom_signal(drawn_t, drawn_x, num_samples=101)

    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()

    assert analyzer.period == pytest.approx(1.0)
    assert analyzer.a.shape == (5,)
    assert analyzer.b.shape == (5,)


def test_insufficient_or_duplicate_only_points_raise() -> None:
    with pytest.raises(ValueError, match="at least two"):
        prepare_custom_signal([0.5], [0.0])
    with pytest.raises(ValueError, match="different time"):
        prepare_custom_signal([0.5, 0.5], [0.0, 1.0])


def test_nonfinite_points_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        prepare_custom_signal([0.0, np.nan], [0.0, 1.0])


def test_freehand_processing_reverses_direction_and_uniformly_samples() -> None:
    raw_points = [[100.0, 50.0], [50.0, 25.0], [0.0, 50.0]]

    time, values = process_freehand_points(
        raw_points, 100.0, 100.0, num_samples=11
    )

    np.testing.assert_allclose(time, np.linspace(0.0, 1.0, 11))
    np.testing.assert_allclose(values[[0, 5, -1]], [0.0, 0.5, 0.0])


def test_freehand_processing_averages_duplicate_x_values() -> None:
    time, values = process_freehand_points(
        [[0.0, 50.0], [50.0, 25.0], [50.0, 75.0], [100.0, 50.0]],
        100.0,
        100.0,
        num_samples=3,
    )

    np.testing.assert_allclose(time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(values, [0.0, 0.0, 0.0])


def test_freehand_processing_rejects_short_and_invalid_strokes() -> None:
    with pytest.raises(ValueError, match="larger portion"):
        process_freehand_points([[10.0, 50.0], [20.0, 50.0]], 100.0, 100.0)
    with pytest.raises(ValueError, match="finite"):
        normalize_drawn_coordinates([[0.0, np.inf], [100.0, 50.0]], 100.0, 100.0)
    with pytest.raises(ValueError, match="at least two"):
        process_freehand_points([[0.0, 50.0]], 100.0, 100.0)
