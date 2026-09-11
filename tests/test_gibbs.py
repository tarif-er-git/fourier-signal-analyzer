"""Tests for Gibbs detection and convergence analysis."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.synthesis import FourierSynthesizer
from metrics.convergence import analyze_convergence
from metrics.gibbs import (
    analyze_gibbs,
    detect_discontinuity,
    theoretical_gibbs_overshoot,
)
from signal.generator import square_wave


def test_smooth_sine_has_no_significant_discontinuity() -> None:
    time = np.linspace(0.0, 1.0, 2001)
    original = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=20).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    result = analyze_gibbs(time, original, synthesizer.reconstruct(20))

    assert result.detected is False
    assert result.overshoot is None
    assert result.message == "No significant discontinuity detected"


def test_square_wave_has_measurable_gibbs_behavior() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    original = square_wave(time, amplitude=1.0, frequency=1.0)
    analyzer = FourierAnalyzer(time, original, num_harmonics=50).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    result = analyze_gibbs(time, original, synthesizer.reconstruct(50))

    assert result.detected is True
    assert result.discontinuity_location == pytest.approx(0.5, abs=2e-3)
    assert result.overshoot is not None and result.overshoot > 0.05
    assert result.undershoot is not None and result.undershoot > 0.05
    assert result.max_local_error is not None and result.max_local_error > 0.5


def test_gibbs_oscillation_region_becomes_more_concentrated() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    original = square_wave(time, amplitude=1.0, frequency=1.0)
    analyzer = FourierAnalyzer(time, original, num_harmonics=50).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    low_n = analyze_gibbs(time, original, synthesizer.reconstruct(5))
    high_n = analyze_gibbs(time, original, synthesizer.reconstruct(50))

    assert low_n.oscillation_width is not None
    assert high_n.oscillation_width is not None
    assert high_n.oscillation_width < low_n.oscillation_width


def test_theoretical_gibbs_reference_uses_jump_magnitude() -> None:
    assert theoretical_gibbs_overshoot(2.0) == pytest.approx(0.178979744472)


def test_convergence_returns_metrics_for_requested_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    original = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=10).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    result = analyze_convergence(original, synthesizer, [1, 5, 10])

    np.testing.assert_array_equal(result.harmonic_counts, [1, 5, 10])
    assert result.mse_values.shape == (3,)
    assert result.rmse_values.shape == (3,)
    assert result.max_error_values.shape == (3,)
    assert np.all(result.mse_values < 1e-12)


def test_invalid_gibbs_inputs_raise_clear_errors() -> None:
    time = np.linspace(0.0, 1.0, 10)
    signal = np.zeros(10)
    with pytest.raises(ValueError):
        analyze_gibbs(time[:-1], signal, signal)
    with pytest.raises(ValueError):
        analyze_gibbs(time, signal, signal, discontinuity_index=10)
    with pytest.raises(ValueError):
        analyze_gibbs(time, signal, signal, window_fraction=0.6)
    with pytest.raises(ValueError):
        detect_discontinuity(time, signal, threshold_factor=0.0)


def test_invalid_convergence_inputs_raise_clear_errors() -> None:
    time = np.linspace(0.0, 1.0, 101)
    original = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=5).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    with pytest.raises(ValueError):
        analyze_convergence(original, synthesizer, [0, 2])
    with pytest.raises(ValueError):
        analyze_convergence(original, synthesizer, [1.5])
