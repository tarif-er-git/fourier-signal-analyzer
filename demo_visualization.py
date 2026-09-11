"""Demonstrate the current signal-to-Fourier visualization pipeline."""

import matplotlib.pyplot as plt
import numpy as np

from fourier.analysis import FourierAnalyzer
from fourier.spectrum import FourierSpectrum
from fourier.synthesis import FourierSynthesizer
from metrics.error import (
    calculate_error,
    calculate_mae,
    calculate_max_error,
    calculate_mse,
    calculate_rmse,
)
from presets.signals import get_preset_signal
from visualization.error_plot import plot_error, plot_error_vs_harmonics
from visualization.signal_plot import plot_original_vs_reconstructed
from visualization.spectrum_plot import plot_magnitude_spectrum, plot_phase_spectrum


def main() -> None:
    """Generate a square wave, analyze it, reconstruct it, and plot results."""
    time = np.linspace(0.0, 1.0, 2001)
    signal_generator = get_preset_signal("square")
    original = signal_generator(time, amplitude=1.0, frequency=1.0)

    analyzer = FourierAnalyzer(time, original, num_harmonics=20).analyze()
    synthesizer = FourierSynthesizer(
        time, analyzer.a0, analyzer.a, analyzer.b, analyzer.omega0
    )
    reconstruction_harmonics = 10
    reconstructed = synthesizer.reconstruct(reconstruction_harmonics)
    spectrum = FourierSpectrum(analyzer.a0, analyzer.a, analyzer.b)
    error = calculate_error(original, reconstructed)

    harmonic_counts = np.arange(1, reconstruction_harmonics + 1)
    mse_values = np.array(
        [calculate_mse(original, synthesizer.reconstruct(count)) for count in harmonic_counts]
    )

    plot_original_vs_reconstructed(
        time, original, reconstructed, reconstruction_harmonics
    )
    plot_magnitude_spectrum(spectrum.harmonics, spectrum.magnitude())
    plot_phase_spectrum(spectrum.harmonics, spectrum.phase())
    plot_error(time, error)
    plot_error_vs_harmonics(harmonic_counts, mse_values)

    print(f"MSE: {calculate_mse(original, reconstructed):.6f}")
    print(f"RMSE: {calculate_rmse(original, reconstructed):.6f}")
    print(f"MAE: {calculate_mae(original, reconstructed):.6f}")
    print(f"Maximum error: {calculate_max_error(original, reconstructed):.6f}")
    plt.show()


if __name__ == "__main__":
    main()
