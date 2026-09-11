"""Tests for numerical Fourier Series coefficient analysis."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.synthesis import FourierSynthesizer
from signal.generator import square_wave


def test_pure_sine_has_unit_first_sine_coefficient() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.sin(2.0 * np.pi * time)

    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()

    assert analyzer.a0 == pytest.approx(0.0, abs=1e-10)
    assert analyzer.a[0] == pytest.approx(0.0, abs=1e-10)
    assert analyzer.b[0] == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.abs(analyzer.a[1:]) < 1e-8)
    assert np.all(np.abs(analyzer.b[1:]) < 1e-8)


def test_pure_cosine_has_unit_first_cosine_coefficient() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.cos(2.0 * np.pi * time)

    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()

    assert analyzer.a0 == pytest.approx(0.0, abs=1e-10)
    assert analyzer.a[0] == pytest.approx(1.0, abs=1e-8)
    assert analyzer.b[0] == pytest.approx(0.0, abs=1e-10)


def test_constant_signal_is_represented_by_dc_component() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    constant = 3.5
    values = np.full(time.shape, constant)

    analyzer = FourierAnalyzer(time, values, num_harmonics=5).analyze()

    assert analyzer.a0 == pytest.approx(2.0 * constant, abs=1e-10)
    assert np.all(np.abs(analyzer.a) < 1e-10)
    assert np.all(np.abs(analyzer.b) < 1e-10)


def test_symmetric_square_wave_has_strong_odd_and_small_even_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 2001)
    values = square_wave(time, amplitude=1.0, frequency=1.0)

    analyzer = FourierAnalyzer(time, values, num_harmonics=7).analyze()

    assert analyzer.a0 == pytest.approx(0.0, abs=2e-3)
    assert abs(analyzer.b[0]) > 0.9
    assert abs(analyzer.b[2]) > 0.25
    assert np.all(np.abs(analyzer.b[1::2]) < 2e-3)


def test_endpoint_excluded_input_infers_period_from_sample_spacing() -> None:
    time = np.linspace(0.0, 1.0, 1000, endpoint=False)
    values = np.sin(2.0 * np.pi * time)

    analyzer = FourierAnalyzer(time, values, num_harmonics=1).analyze()

    assert analyzer.period == pytest.approx(1.0)
    assert analyzer.b[0] == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    "time, values, harmonics",
    [
        ([0.0, 1.0], [0.0], 3),
        ([0.0, 0.5, 1.0], [0.0, 1.0, 0.0], 0),
        ([0.0, 0.6, 1.0], [0.0, 1.0, 0.0], 3),
        ([0.0, 0.5, 1.0], [0.0, np.nan, 0.0], 3),
    ],
)
def test_invalid_inputs_raise_value_error(time, values, harmonics) -> None:
    with pytest.raises((TypeError, ValueError)):
        FourierAnalyzer(time, values, harmonics)


def test_pure_sine_reconstructs_with_one_harmonic() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=3).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)

    reconstructed = synthesizer.reconstruct(1)

    np.testing.assert_allclose(reconstructed, values, atol=1e-8)


def test_pure_sine_individual_first_harmonic_matches_original() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=3).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)

    np.testing.assert_allclose(synthesizer.harmonic(1), values, atol=1e-8)


def test_pure_cosine_reconstructs_with_one_harmonic() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    values = np.cos(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, values, num_harmonics=3).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)

    np.testing.assert_allclose(synthesizer.reconstruct(1), values, atol=1e-8)


def test_constant_signal_reconstructs_with_zero_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 101)
    values = np.full(time.shape, 4.25)
    analyzer = FourierAnalyzer(time, values, num_harmonics=3).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)

    np.testing.assert_allclose(synthesizer.reconstruct(0), values, atol=1e-10)
    np.testing.assert_allclose(synthesizer.dc_component(), values, atol=1e-10)


def test_harmonics_and_components_have_expected_shapes() -> None:
    time = np.linspace(0.0, 1.0, 101)
    analyzer = FourierAnalyzer(time, np.sin(2.0 * np.pi * time), 4).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)

    first_harmonic = synthesizer.harmonic(1)
    components = synthesizer.get_harmonics(3)

    assert first_harmonic.shape == time.shape
    assert components.shape == (3, time.size)
    np.testing.assert_allclose(components[0], first_harmonic)


def test_square_wave_approximation_improves_away_from_discontinuities() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    values = square_wave(time, amplitude=1.0, frequency=1.0)
    analyzer = FourierAnalyzer(time, values, num_harmonics=50).analyze()
    synthesizer = FourierSynthesizer(time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0)
    away_from_edges = (time > 0.05) & (time < 0.45) | (time > 0.55) & (time < 0.95)

    errors = []
    for harmonic_count in (1, 3, 10, 50):
        reconstructed = synthesizer.reconstruct(harmonic_count)
        errors.append(np.mean((reconstructed[away_from_edges] - values[away_from_edges]) ** 2))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]


def test_synthesizer_rejects_unavailable_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 101)
    synthesizer = FourierSynthesizer(time, 0.0, [1.0], [0.0], 2.0 * np.pi)

    with pytest.raises(ValueError):
        synthesizer.reconstruct(2)
    with pytest.raises(ValueError):
        synthesizer.harmonic(2)


def test_synthesizer_rejects_negative_non_integer_and_oversized_n() -> None:
    time = np.linspace(0.0, 1.0, 101)
    synthesizer = FourierSynthesizer(
        time, 0.0, [1.0, 0.0], [0.0, 0.0], 2.0 * np.pi
    )

    with pytest.raises(ValueError):
        synthesizer.reconstruct(-1)
    with pytest.raises(TypeError):
        synthesizer.reconstruct(1.5)
    with pytest.raises(ValueError):
        synthesizer.reconstruct(3)