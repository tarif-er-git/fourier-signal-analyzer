"""Tests for reconstruction error and accuracy metrics."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.synthesis import FourierSynthesizer
from metrics.error import (
    calculate_error,
    calculate_mae,
    calculate_max_error,
    calculate_mse,
    calculate_rmse,
)


def test_identical_signals_have_zero_error_metrics() -> None:
    signal = np.array([-1.0, 0.0, 2.0, 4.0])

    error = calculate_error(signal, signal)

    np.testing.assert_array_equal(error, np.zeros(signal.shape))
    assert calculate_mse(signal, signal) == 0.0
    assert calculate_rmse(signal, signal) == 0.0
    assert calculate_mae(signal, signal) == 0.0
    assert calculate_max_error(signal, signal) == 0.0


def test_known_difference_has_correct_metrics() -> None:
    original = np.array([1.0, 2.0, 3.0])
    reconstructed = np.array([0.0, 1.0, 2.0])

    np.testing.assert_array_equal(calculate_error(original, reconstructed), [1.0, 1.0, 1.0])
    assert calculate_mse(original, reconstructed) == pytest.approx(1.0)
    assert calculate_rmse(original, reconstructed) == pytest.approx(1.0)
    assert calculate_mae(original, reconstructed) == pytest.approx(1.0)
    assert calculate_max_error(original, reconstructed) == pytest.approx(1.0)


def test_error_uses_original_minus_reconstructed_sign() -> None:
    original = np.array([1.0, -2.0])
    reconstructed = np.array([3.0, -1.0])

    np.testing.assert_array_equal(calculate_error(original, reconstructed), [-2.0, -1.0])


def test_metrics_reject_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_mse([1.0, 2.0], [[1.0, 2.0]])


def test_metrics_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_error([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        calculate_rmse([1.0, 2.0], [1.0, np.inf])


def test_sine_reconstruction_has_small_error() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    original = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=5).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    one_harmonic_rmse = calculate_rmse(original, synthesizer.reconstruct(1))
    five_harmonic_rmse = calculate_rmse(original, synthesizer.reconstruct(5))

    assert one_harmonic_rmse < 1e-8
    assert five_harmonic_rmse < 1e-8
    assert five_harmonic_rmse <= one_harmonic_rmse + 1e-15
