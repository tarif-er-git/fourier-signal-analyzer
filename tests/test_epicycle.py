"""Tests for Fourier epicycle geometry."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.synthesis import FourierSynthesizer
from visualization.epicycle import EpicycleVisualizer


def test_single_cosine_vector_matches_cosine_signal() -> None:
    time = np.array([0.0, 0.25, 0.5, 0.75])
    visualizer = EpicycleVisualizer(0.0, [1.0], [0.0], 2.0 * np.pi)

    values = np.array([visualizer.reconstruct(value, 1) for value in time])

    np.testing.assert_allclose(values, np.cos(2.0 * np.pi * time), atol=1e-12)
    frame = visualizer.frame(0.0, 1)
    assert frame.radii[0] == pytest.approx(1.0)
    assert frame.angles[0] == pytest.approx(0.0)


def test_single_sine_vector_matches_existing_synthesizer() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    original = np.sin(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=1).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )
    visualizer = EpicycleVisualizer(
        analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    epicycle_values = np.array(
        [visualizer.reconstruct(value, 1) for value in time[::100]]
    )

    np.testing.assert_allclose(
        epicycle_values, synthesizer.reconstruct(1)[::100], atol=1e-8
    )


def test_multiple_vector_chain_matches_synthesis() -> None:
    time = np.linspace(0.0, 1.0, 101)
    a0 = 0.4
    cosine_coefficients = np.array([1.0, 0.2, -0.1])
    sine_coefficients = np.array([0.3, -0.4, 0.15])
    omega0 = 2.0 * np.pi
    visualizer = EpicycleVisualizer(
        a0, cosine_coefficients, sine_coefficients, omega0
    )
    synthesizer = FourierSynthesizer(
        time, a0, cosine_coefficients, sine_coefficients, omega0
    )

    epicycle_values = np.array(
        [visualizer.reconstruct(value, 3) for value in time]
    )

    np.testing.assert_allclose(epicycle_values, synthesizer.reconstruct(3), atol=1e-12)


def test_requested_harmonic_count_limits_chain() -> None:
    visualizer = EpicycleVisualizer(0.0, [1.0, 2.0, 3.0, 4.0, 5.0], np.zeros(5), 1.0)

    assert visualizer.frame(0.0, 1).endpoints.shape == (1, 2)
    assert visualizer.frame(0.0, 3).endpoints.shape == (3, 2)
    assert visualizer.frame(0.0, 5).endpoints.shape == (5, 2)


@pytest.mark.parametrize(
    "a0, a, b, omega0",
    [
        (0.0, [1.0], [0.0, 1.0], 1.0),
        (np.nan, [1.0], [0.0], 1.0),
        (0.0, [np.inf], [0.0], 1.0),
        (0.0, [1.0], [0.0], 0.0),
    ],
)
def test_invalid_epicycle_inputs_raise_value_error(a0, a, b, omega0) -> None:
    with pytest.raises(ValueError):
        EpicycleVisualizer(a0, a, b, omega0)


def test_invalid_harmonic_counts_raise() -> None:
    visualizer = EpicycleVisualizer(0.0, [1.0], [0.0], 1.0)

    with pytest.raises(ValueError):
        visualizer.frame(0.0, -1)
    with pytest.raises(TypeError):
        visualizer.frame(0.0, 1.5)
    with pytest.raises(ValueError):
        visualizer.frame(0.0, 2)
