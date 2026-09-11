"""Additional convergence validation coverage."""

import numpy as np
import pytest

from fourier.analysis import FourierAnalyzer
from fourier.synthesis import FourierSynthesizer
from metrics.convergence import analyze_convergence


def test_default_convergence_uses_available_harmonics() -> None:
    time = np.linspace(0.0, 1.0, 101)
    original = np.cos(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=4).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    result = analyze_convergence(original, synthesizer)

    np.testing.assert_array_equal(result.harmonic_counts, [1, 2, 3, 4])


def test_convergence_rejects_mismatched_original_length() -> None:
    time = np.linspace(0.0, 1.0, 101)
    original = np.cos(2.0 * np.pi * time)
    analyzer = FourierAnalyzer(time, original, num_harmonics=4).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )

    with pytest.raises(ValueError, match="same shape"):
        analyze_convergence(original[:-1], synthesizer)
