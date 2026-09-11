"""NumPy-based generators for simple periodic signals."""

import numpy as np


def _validate_parameters(amplitude: float, frequency: float) -> None:
    """Validate parameters shared by all waveform generators."""
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    if frequency <= 0:
        raise ValueError("frequency must be positive")


def _as_time_array(t: np.ndarray) -> np.ndarray:
    """Convert a time input to a floating-point NumPy array."""
    return np.asarray(t, dtype=float)


def sine_wave(
    t: np.ndarray, amplitude: float = 1.0, frequency: float = 1.0
) -> np.ndarray:
    """Generate a sine wave, x(t) = A sin(2 pi f t).

    Parameters
    ----------
    t
        Time samples at which to evaluate the signal.
    amplitude
        Peak signal amplitude A. Must be non-negative.
    frequency
        Frequency f in cycles per unit time. Must be positive.

    Returns
    -------
    numpy.ndarray
        Sine-wave values with the same shape as ``t``.
    """
    _validate_parameters(amplitude, frequency)
    time = _as_time_array(t)
    return amplitude * np.sin(2 * np.pi * frequency * time)


def square_wave(
    t: np.ndarray, amplitude: float = 1.0, frequency: float = 1.0
) -> np.ndarray:
    """Generate a symmetric periodic square wave with values -A and +A.

    The positive half-cycle is defined by ``sin(2 pi f t) >= 0``. The
    value at an exact zero crossing is therefore +A.

    Parameters
    ----------
    t
        Time samples at which to evaluate the signal.
    amplitude
        Peak signal amplitude A. Must be non-negative.
    frequency
        Frequency f in cycles per unit time. Must be positive.

    Returns
    -------
    numpy.ndarray
        Square-wave values with the same shape as ``t``.
    """
    _validate_parameters(amplitude, frequency)
    time = _as_time_array(t)
    phase = np.sin(2 * np.pi * frequency * time)
    return np.where(phase >= 0, amplitude, -amplitude)


def triangle_wave(
    t: np.ndarray, amplitude: float = 1.0, frequency: float = 1.0
) -> np.ndarray:
    """Generate a symmetric triangle wave between -A and +A.

    With phase ``p = (f t) mod 1``, this uses
    ``x(t) = A (1 - 4 |p - 1/2|)``. The wave starts at -A, reaches +A
    halfway through each period, and repeats every ``1 / f`` time units.

    Parameters
    ----------
    t
        Time samples at which to evaluate the signal.
    amplitude
        Peak signal amplitude A. Must be non-negative.
    frequency
        Frequency f in cycles per unit time. Must be positive.

    Returns
    -------
    numpy.ndarray
        Triangle-wave values with the same shape as ``t``.
    """
    _validate_parameters(amplitude, frequency)
    time = _as_time_array(t)
    phase = np.mod(frequency * time, 1.0)
    return amplitude * (1.0 - 4.0 * np.abs(phase - 0.5))


def sawtooth_wave(
    t: np.ndarray, amplitude: float = 1.0, frequency: float = 1.0
) -> np.ndarray:
    """Generate a rising sawtooth wave between -A and values below +A.

    With phase ``p = (f t) mod 1``, this uses ``x(t) = A (2 p - 1)``.
    The value jumps from just below +A back to -A at each period boundary.

    Parameters
    ----------
    t
        Time samples at which to evaluate the signal.
    amplitude
        Peak signal amplitude A. Must be non-negative.
    frequency
        Frequency f in cycles per unit time. Must be positive.

    Returns
    -------
    numpy.ndarray
        Sawtooth values with the same shape as ``t``.
    """
    _validate_parameters(amplitude, frequency)
    time = _as_time_array(t)
    phase = np.mod(frequency * time, 1.0)
    return amplitude * (2.0 * phase - 1.0)
