"""Named preset signals used by the application."""

from collections.abc import Callable

import numpy as np

from signal.generator import sawtooth_wave, sine_wave, square_wave, triangle_wave

SignalGenerator = Callable[[np.ndarray, float, float], np.ndarray]

PRESET_SIGNALS: dict[str, SignalGenerator] = {
    "sine": sine_wave,
    "square": square_wave,
    "triangle": triangle_wave,
    "sawtooth": sawtooth_wave,
}


def get_preset_signal(name: str) -> SignalGenerator:
    """Return a waveform generator by its preset name.

    Parameters
    ----------
    name
        One of ``sine``, ``square``, ``triangle``, or ``sawtooth``.

    Returns
    -------
    collections.abc.Callable
        The corresponding generator from :mod:`signal.generator`.

    Raises
    ------
    ValueError
        If ``name`` is not a supported preset.
    """
    try:
        return PRESET_SIGNALS[name.lower()]
    except KeyError as error:
        available_names = ", ".join(PRESET_SIGNALS)
        raise ValueError(
            f"unknown preset {name!r}; choose one of: {available_names}"
        ) from error
