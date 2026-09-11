"""Tests for the basic periodic signal generators."""

import numpy as np
import pytest

from presets.signals import PRESET_SIGNALS, get_preset_signal
from signal.generator import sawtooth_wave, sine_wave, square_wave, triangle_wave
from signal.preprocessor import (
    normalize_signal,
    preprocess_signal,
    validate_signal_arrays,
)
from signal.sampler import create_time_grid, prepare_signal_points, resample_signal


@pytest.mark.parametrize(
    "generator",
    [sine_wave, square_wave, triangle_wave, sawtooth_wave],
)
def test_generators_preserve_time_array_shape(generator) -> None:
    time = np.linspace(0.0, 2.0, 101)

    values = generator(time, amplitude=2.5, frequency=1.5)

    assert isinstance(values, np.ndarray)
    assert values.shape == time.shape


@pytest.mark.parametrize(
    "generator",
    [sine_wave, square_wave, triangle_wave, sawtooth_wave],
)
def test_generators_stay_within_amplitude_range(generator) -> None:
    time = np.linspace(0.0, 4.0, 4001)
    amplitude = 2.5

    values = generator(time, amplitude=amplitude, frequency=1.0)

    assert np.all(values <= amplitude)
    assert np.all(values >= -amplitude)


def test_sine_wave_has_expected_basic_values() -> None:
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    values = sine_wave(time, amplitude=2.0, frequency=1.0)

    np.testing.assert_allclose(values, [0.0, 2.0, 0.0, -2.0, 0.0], atol=1e-12)


def test_preset_lookup_returns_all_four_generators() -> None:
    time = np.linspace(0.0, 1.0, 10)

    assert set(PRESET_SIGNALS) == {"sine", "square", "triangle", "sawtooth"}
    for name in PRESET_SIGNALS:
        values = get_preset_signal(name)(time, 1.0, 1.0)
        assert values.shape == time.shape


def test_invalid_signal_parameters_are_rejected() -> None:
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        sine_wave(time, amplitude=-1.0)
    with pytest.raises(ValueError):
        sine_wave(time, frequency=0.0)


def test_create_time_grid_is_uniform_and_includes_boundaries() -> None:
    time = create_time_grid(0.0, 1.0, 5)

    np.testing.assert_allclose(time, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(np.diff(time), 0.25)


def test_resample_signal_interpolates_and_holds_endpoints() -> None:
    original_time = np.array([0.0, 1.0, 2.0])
    original_values = np.array([0.0, 2.0, 0.0])
    target_time = np.array([-1.0, 0.5, 1.5, 3.0])

    values = resample_signal(original_time, original_values, target_time)

    np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 0.0])


def test_resample_signal_sorts_points_and_averages_duplicate_times() -> None:
    time, values = prepare_signal_points(
        [2.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 3.0]
    )

    np.testing.assert_allclose(time, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(values, [0.0, 2.0, 2.0])


def test_mismatched_time_and_signal_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_signal_arrays([0.0, 1.0], [2.0])


def test_nonfinite_values_can_be_rejected_or_removed() -> None:
    time = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError):
        preprocess_signal(time, values)

    clean_time, clean_values = preprocess_signal(time, values, nonfinite="remove")
    np.testing.assert_allclose(clean_time, [0.0, 2.0])
    np.testing.assert_allclose(clean_values, [1.0, 3.0])


def test_normalization_scales_peak_without_removing_mean() -> None:
    values = np.array([2.0, 4.0, 6.0])

    normalized = normalize_signal(values, target_amplitude=1.0)

    np.testing.assert_allclose(normalized, [1 / 3, 2 / 3, 1.0])
    assert np.mean(normalized) == pytest.approx(np.mean(values) / 6.0)


def test_preprocessing_preserves_mean_by_default() -> None:
    time = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, 3.0, 5.0])

    _, processed = preprocess_signal(time, values)

    np.testing.assert_allclose(processed, values)
    assert np.mean(processed) == pytest.approx(np.mean(values))


def test_periodic_preprocessing_removes_duplicate_matching_endpoint() -> None:
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    values = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

    processed_time, processed_values = preprocess_signal(
        time, values, periodic=True
    )

    np.testing.assert_allclose(processed_time, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(processed_values, [0.0, 1.0, 0.0, -1.0])
