"""Tests for FFT comparison utilities."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.fft_comparison import (
    compare_fourier_and_fft,
    compare_performance,
    fft_transform,
    prepare_fft_period_samples,
    single_sided_spectrum,
)
from signal.generator import square_wave


def test_fft_identifies_sine_fundamental() -> None:
    count = 1000
    time = np.linspace(0.0, 1.0, count, endpoint=False)
    values = np.sin(2.0 * np.pi * time)
    spectrum = single_sided_spectrum(values, sampling_frequency=count)

    dominant = int(np.argmax(spectrum.magnitudes[1:]) + 1)

    assert dominant == 1
    assert spectrum.frequencies[dominant] == pytest.approx(1.0)
    assert spectrum.magnitudes[dominant] == pytest.approx(1.0, abs=1e-10)


def test_fft_identifies_cosine_fundamental() -> None:
    count = 1000
    time = np.linspace(0.0, 1.0, count, endpoint=False)
    spectrum = single_sided_spectrum(np.cos(2.0 * np.pi * time), count)

    assert np.argmax(spectrum.magnitudes[1:]) + 1 == 1
    assert spectrum.magnitudes[1] == pytest.approx(1.0, abs=1e-10)


def test_square_wave_has_strong_odd_and_weak_even_harmonics() -> None:
    count = 1000
    time = np.linspace(0.0, 1.0, count, endpoint=False)
    spectrum = single_sided_spectrum(square_wave(time), count)

    assert spectrum.magnitudes[1] > 0.9
    assert spectrum.magnitudes[3] > 0.25
    assert spectrum.magnitudes[2] < 0.01
    assert spectrum.magnitudes[4] < 0.01


def test_frequency_bins_use_sampling_frequency() -> None:
    result = fft_transform(np.zeros(8), sampling_frequency=8.0)

    np.testing.assert_allclose(result.frequencies, [0, 1, 2, 3, -4, -3, -2, -1])
    assert result.period == pytest.approx(1.0)


def test_fourier_and_fft_magnitudes_are_consistent() -> None:
    count = 1000
    time = np.linspace(0.0, 1.0, count, endpoint=False)
    values = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()

    comparison = compare_fourier_and_fft(time, values, analyzer)

    assert comparison.harmonic_numbers[0] == 1
    assert comparison.fourier_magnitudes[0] == pytest.approx(1.0, abs=1e-8)
    assert comparison.fft_magnitudes[0] == pytest.approx(1.0, abs=1e-10)
    assert comparison.absolute_difference[0] < 1e-8


def test_endpoint_inclusive_samples_drop_duplicate_for_fft() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.sin(2.0 * np.pi * time)

    fft_values, period = prepare_fft_period_samples(time, values)

    assert fft_values.size == 1000
    assert period == pytest.approx(1.0)


def test_performance_comparison_returns_timings() -> None:
    results = compare_performance([16, 32], harmonics=3)

    assert [result.sample_count for result in results] == [16, 32]
    assert all(result.fourier_seconds >= 0 for result in results)
    assert all(result.fft_seconds >= 0 for result in results)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: fft_transform([], 1.0),
        lambda: fft_transform([1.0, np.nan], 1.0),
        lambda: fft_transform([1.0, 2.0], 0.0),
        lambda: single_sided_spectrum([1.0, 2.0], -1.0),
        lambda: prepare_fft_period_samples([0.0, 0.5, 1.0], [1.0, 2.0]),
    ],
)
def test_invalid_fft_inputs_raise_value_error(operation) -> None:
    with pytest.raises(ValueError):
        operation()
