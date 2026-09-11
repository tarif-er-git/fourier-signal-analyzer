"""Educational comparison between direct Fourier Series and NumPy FFT.

``FourierAnalyzer`` computes trigonometric coefficients ``a_n`` and ``b_n``
by numerical integration. NumPy's FFT computes complex DFT coefficients
``X[k]``. They describe related harmonic information for uniformly sampled
periodic signals, but ``X[k]`` is not itself an ``a_n`` or ``b_n`` coefficient.
This module keeps both representations explicit and converts FFT bins to a
single-sided physical amplitude spectrum for comparison.
"""

from dataclasses import dataclass
from collections.abc import Sequence
import time

import numpy as np

from .analysis import FourierAnalyzer
from .spectrum import FourierSpectrum


@dataclass(frozen=True)
class FFTResult:
    """Full complex DFT result and its physical frequency bins."""

    frequencies: np.ndarray
    coefficients: np.ndarray
    sampling_frequency: float
    period: float


@dataclass(frozen=True)
class SingleSidedSpectrum:
    """Amplitude-scaled non-negative-frequency FFT spectrum."""

    frequencies: np.ndarray
    magnitudes: np.ndarray


@dataclass(frozen=True)
class FourierFFTComparison:
    """Harmonic-by-harmonic Fourier Series and FFT magnitudes."""

    harmonic_numbers: np.ndarray
    frequencies: np.ndarray
    fourier_magnitudes: np.ndarray
    fft_magnitudes: np.ndarray
    absolute_difference: np.ndarray


@dataclass(frozen=True)
class TimingResult:
    """Measured direct-analysis and FFT durations for one sample length."""

    sample_count: int
    harmonics: int
    fourier_seconds: float
    fft_seconds: float


def _validate_signal(x: Sequence[float]) -> np.ndarray:
    """Validate a non-empty finite one-dimensional real signal."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")
    return values


def _validate_sampling_frequency(sampling_frequency: float) -> float:
    """Validate a positive sampling frequency."""
    value = float(sampling_frequency)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sampling_frequency must be positive and finite")
    return value


def _validate_period(period: float) -> float:
    """Validate a positive physical period."""
    value = float(period)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("period must be positive and finite")
    return value


def fft_transform(x: Sequence[float], sampling_frequency: float) -> FFTResult:
    """Calculate the full DFT using NumPy's standard unnormalized convention.

    ``np.fft.fft`` returns ``X[k] = sum(x[j] exp(-2 pi i j k / N))`` and
    ``np.fft.fftfreq`` returns bins in cycles per unit time. The raw
    coefficients are intentionally returned unchanged; use
    :func:`single_sided_spectrum` for amplitude-scaled magnitudes.
    """
    values = _validate_signal(x)
    sample_rate = _validate_sampling_frequency(sampling_frequency)
    return FFTResult(
        frequencies=np.fft.fftfreq(values.size, d=1.0 / sample_rate),
        coefficients=np.fft.fft(values),
        sampling_frequency=sample_rate,
        period=values.size / sample_rate,
    )


def single_sided_spectrum(
    x: Sequence[float], sampling_frequency: float
) -> SingleSidedSpectrum:
    """Return the physical one-sided amplitude spectrum for a real signal.

    The FFT is divided by the sample count. Positive-frequency bins are then
    doubled, except DC and the Nyquist bin when the sample count is even. For
    one period sampled without a duplicate endpoint, bins map directly to
    ``k / T`` and therefore to harmonics ``k * f0``.
    """
    values = _validate_signal(x)
    sample_rate = _validate_sampling_frequency(sampling_frequency)
    coefficients = np.fft.rfft(values)
    magnitudes = np.abs(coefficients) / values.size
    if values.size > 1:
        last_index = magnitudes.size - 1 if values.size % 2 == 0 else magnitudes.size
        magnitudes[1:last_index] *= 2.0
    return SingleSidedSpectrum(
        frequencies=np.fft.rfftfreq(values.size, d=1.0 / sample_rate),
        magnitudes=magnitudes,
    )


def prepare_fft_period_samples(
    t: Sequence[float], x: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Remove a repeated endpoint and return FFT samples plus physical period.

    The project commonly represents one period as ``[0, T]``. Since ``t=0``
    and ``t=T`` are the same periodic point, the final sample is dropped for
    the FFT when the endpoints match. For endpoint-exclusive input, all
    samples are retained and the period is inferred from sample spacing.
    """
    time_values = np.asarray(t, dtype=float)
    signal_values = _validate_signal(x)
    if time_values.ndim != 1 or time_values.shape != signal_values.shape:
        raise ValueError("t and x must be matching one-dimensional arrays")
    if time_values.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(time_values)) or not np.all(np.diff(time_values) > 0):
        raise ValueError("t must be finite and strictly increasing")
    spacing = float(np.median(np.diff(time_values)))
    if not np.allclose(np.diff(time_values), spacing, rtol=1e-5, atol=1e-12):
        raise ValueError("t must be uniformly sampled")
    if np.isclose(signal_values[0], signal_values[-1]) and time_values.size >= 3:
        return signal_values[:-1], float(time_values[-1] - time_values[0])
    return signal_values, float(time_values[-1] - time_values[0] + spacing)


def compare_fourier_and_fft(
    t: Sequence[float],
    x: Sequence[float],
    analyzer: FourierAnalyzer | None = None,
) -> FourierFFTComparison:
    """Compare direct Fourier magnitudes with normalized FFT magnitudes.

    The Fourier analyzer is limited to the available FFT harmonics when it is
    supplied. If omitted, it is created for all positive rFFT harmonics. The
    direct Fourier magnitude is ``sqrt(a_n**2 + b_n**2)``; the FFT magnitude
    uses the single-sided amplitude scaling from :func:`single_sided_spectrum`.
    """
    time_values = np.asarray(t, dtype=float)
    signal_values = _validate_signal(x)
    fft_values, period = prepare_fft_period_samples(time_values, signal_values)
    sample_rate = fft_values.size / period
    fft_spectrum = single_sided_spectrum(fft_values, sample_rate)
    available = fft_spectrum.magnitudes.size - 1
    if available < 1:
        raise ValueError("at least two positive-frequency bins are required")
    if analyzer is None:
        analyzer = FourierAnalyzer(time_values, signal_values, num_harmonics=available)
        analyzer.analyze()
    if analyzer.a is None or analyzer.b is None:
        analyzer.analyze()
    count = min(analyzer.a.size, available)
    fourier_magnitudes = FourierSpectrum(
        analyzer.a0, analyzer.a[:count], analyzer.b[:count]
    ).magnitude()
    harmonic_numbers = np.arange(1, count + 1, dtype=int)
    fft_magnitudes = fft_spectrum.magnitudes[harmonic_numbers]
    return FourierFFTComparison(
        harmonic_numbers=harmonic_numbers,
        frequencies=fft_spectrum.frequencies[harmonic_numbers],
        fourier_magnitudes=fourier_magnitudes,
        fft_magnitudes=fft_magnitudes,
        absolute_difference=np.abs(fourier_magnitudes - fft_magnitudes),
    )


def compare_performance(
    sample_counts: Sequence[int] = (64, 128, 256, 512, 1024),
    *,
    harmonics: int | None = None,
) -> tuple[TimingResult, ...]:
    """Time direct numerical coefficients versus NumPy FFT.

    The direct side analyzes ``min(harmonics, N/2 - 1)`` harmonics, while the
    FFT side computes the complete transform. This is an intentionally simple
    educational comparison, not a benchmark harness; machine, Python, and
    NumPy implementation details affect measured values. Direct calculation
    is expected to scale approximately as O(N^2), while FFT scales as
    O(N log N) for comparable full-spectrum work.
    """
    counts = np.asarray(sample_counts)
    if counts.ndim != 1 or counts.size == 0 or not np.all(np.isfinite(counts)):
        raise ValueError("sample_counts must be a non-empty finite sequence")
    if not np.all(np.equal(counts, counts.astype(int))) or np.any(counts < 4):
        raise ValueError("sample_counts must contain integers of at least 4")
    results: list[TimingResult] = []
    for raw_count in counts.astype(int):
        count = int(raw_count)
        time_values = np.linspace(0.0, 1.0, count, endpoint=False)
        signal_values = np.sin(2.0 * np.pi * time_values)
        harmonic_count = harmonics or max(1, count // 2 - 1)
        harmonic_count = min(harmonic_count, count // 2 - 1)
        start = time.perf_counter()
        FourierAnalyzer(time_values, signal_values, harmonic_count).analyze()
        fourier_seconds = time.perf_counter() - start
        start = time.perf_counter()
        np.fft.fft(signal_values)
        fft_seconds = time.perf_counter() - start
        results.append(
            TimingResult(count, harmonic_count, fourier_seconds, fft_seconds)
        )
    return tuple(results)
