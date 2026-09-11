"""Tests for Fourier Series magnitude and phase spectra."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.spectrum import FourierSpectrum
from signal.generator import square_wave


def test_pure_cosine_has_unit_magnitude_and_zero_phase() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.cos(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()
    spectrum = FourierSpectrum(analyzer.a0, analyzer.a, analyzer.b)

    assert spectrum.magnitude()[0] == pytest.approx(1.0, abs=1e-8)
    assert spectrum.phase()[0] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_array_equal(spectrum.harmonics, [1, 2, 3, 4, 5])


def test_pure_sine_has_negative_quarter_cycle_phase() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()
    spectrum = FourierSpectrum(analyzer.a0, analyzer.a, analyzer.b)

    assert spectrum.magnitude()[0] == pytest.approx(1.0, abs=1e-8)
    assert spectrum.phase()[0] == pytest.approx(-np.pi / 2.0, abs=1e-8)


def test_constant_signal_has_dc_value_and_no_harmonic_magnitude() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    constant = 3.5
    analyzer = FourierAnalyzer(
        time, np.full(time.shape, constant), num_harmonics=5
    ).analyze()
    spectrum = FourierSpectrum(analyzer.a0, analyzer.a, analyzer.b)

    assert spectrum.dc_value == pytest.approx(constant, abs=1e-10)
    assert np.all(spectrum.magnitude() < 1e-10)
    assert np.all(spectrum.phase() == 0.0)


def test_square_wave_has_decreasing_odd_harmonics_and_small_even_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 2001)
    values = square_wave(time, amplitude=1.0, frequency=1.0)
    analyzer = FourierAnalyzer(time, values, num_harmonics=9).analyze()
    spectrum = FourierSpectrum(analyzer.a0, analyzer.a, analyzer.b)
    magnitudes = spectrum.magnitude()

    assert magnitudes[0] > 0.9
    assert magnitudes[2] > magnitudes[4] > magnitudes[6]
    assert np.all(magnitudes[1::2] < 2e-3)


def test_zero_coefficient_pair_has_zero_magnitude_and_finite_zero_phase() -> None:
    spectrum = FourierSpectrum(0.0, [0.0, 1.0], [0.0, 0.0])

    assert spectrum.magnitude()[0] == 0.0
    assert spectrum.phase()[0] == 0.0
    assert np.all(np.isfinite(spectrum.phase()))


def test_spectrum_returns_combined_information() -> None:
    spectrum = FourierSpectrum(2.0, [1.0], [0.0])

    result = spectrum.as_dict()

    assert set(result) == {"harmonic", "magnitude", "phase"}
    np.testing.assert_array_equal(result["harmonic"], [1])
    np.testing.assert_allclose(result["magnitude"], [1.0])


@pytest.mark.parametrize(
    "a0, a, b",
    [
        (0.0, [1.0], [0.0, 1.0]),
        (np.nan, [1.0], [0.0]),
        (0.0, [np.inf], [0.0]),
        (0.0, [1.0], [np.nan]),
        (0.0, [[1.0]], [[0.0]]),
    ],
)
def test_invalid_spectrum_inputs_raise_value_error(a0, a, b) -> None:
    with pytest.raises(ValueError):
        FourierSpectrum(a0, a, b)
