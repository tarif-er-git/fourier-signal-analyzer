"""Validation and optional preprocessing for sampled one-dimensional signals."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .sampler import prepare_signal_points

NonFinitePolicy = Literal["raise", "remove"]


def validate_signal_arrays(
    t: Sequence[float], x: Sequence[float], *, require_finite: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Validate time and signal arrays and return floating-point copies.

    Parameters
    ----------
    t, x
        One-dimensional time and signal samples.
    require_finite
        If true, reject NaN and infinity values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Validated arrays with matching lengths.
    """
    time = np.asarray(t, dtype=float)
    values = np.asarray(x, dtype=float)
    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and signal inputs must be one-dimensional")
    if time.size == 0:
        raise ValueError("time and signal inputs must not be empty")
    if time.size != values.size:
        raise ValueError("time and signal inputs must have the same length")
    if require_finite and (
        not np.all(np.isfinite(time)) or not np.all(np.isfinite(values))
    ):
        raise ValueError("time and signal inputs must contain only finite values")
    return time.copy(), values.copy()


def remove_nonfinite_samples(
    t: Sequence[float], x: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove samples where either time or signal value is non-finite.

    At least one finite sample must remain. A later interpolation step may
    require at least two distinct time values.
    """
    time, values = validate_signal_arrays(t, x, require_finite=False)
    finite_mask = np.isfinite(time) & np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError("no finite signal samples remain")
    return time[finite_mask], values[finite_mask]


def normalize_signal(x: Sequence[float], target_amplitude: float = 1.0) -> np.ndarray:
    """Scale a signal so its largest absolute value equals a target value.

    A zero signal is returned unchanged because it has no meaningful scale.
    This operation scales the mean along with the signal; it does not remove
    the DC component.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal input must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal input must contain only finite values")
    target = float(target_amplitude)
    if not np.isfinite(target) or target < 0:
        raise ValueError("target_amplitude must be finite and non-negative")

    peak = np.max(np.abs(values))
    if peak == 0 or target == peak:
        return values.copy()
    return values * (target / peak)


def preprocess_signal(
    t: Sequence[float],
    x: Sequence[float],
    *,
    normalize: bool = False,
    target_amplitude: float = 1.0,
    remove_mean: bool = False,
    nonfinite: NonFinitePolicy = "raise",
    periodic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and optionally clean a sampled signal for later analysis.

    By default, the function preserves the signal mean and rejects non-finite
    values. Set ``remove_mean=True`` only when explicitly needed. If
    ``periodic=True`` and the first and last samples have matching values,
    the final duplicate boundary sample is removed.

    Parameters
    ----------
    t, x
        Time and signal samples. They may be unsorted and may contain
        duplicate time values.
    normalize
        Scale the signal peak to ``target_amplitude`` when true.
    target_amplitude
        Peak magnitude used by optional normalization.
    remove_mean
        Subtract the signal mean when true. Defaults to false to preserve DC.
    nonfinite
        ``"raise"`` rejects NaN/inf; ``"remove"`` drops affected samples.
    periodic
        Remove a duplicated final periodic boundary when the first and last
        signal values are equal within NumPy's default tolerance.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Cleaned, sorted time and signal arrays.
    """
    if nonfinite not in ("raise", "remove"):
        raise ValueError("nonfinite must be 'raise' or 'remove'")

    if nonfinite == "remove":
        time, values = remove_nonfinite_samples(t, x)
    else:
        time, values = validate_signal_arrays(t, x)

    time, values = prepare_signal_points(time, values)
    if periodic and time.size > 2 and np.isclose(values[0], values[-1]):
        time = time[:-1]
        values = values[:-1]

    if remove_mean:
        values = values - np.mean(values)
    if normalize:
        values = normalize_signal(values, target_amplitude)
    return time, values
