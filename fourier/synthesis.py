"""Fourier Series reconstruction from trigonometric coefficients."""

from collections.abc import Sequence

import numpy as np


class FourierSynthesizer:
    """Synthesize a sampled signal from Fourier Series coefficients.

    The coefficient convention matches :class:`fourier.analysis.FourierAnalyzer`:
    ``a[n - 1]`` and ``b[n - 1]`` are the cosine and sine coefficients for
    harmonic ``n``, while the constant contribution is ``a0 / 2``.
    """

    def __init__(
        self,
        t: Sequence[float],
        a0: float,
        a: Sequence[float],
        b: Sequence[float],
        omega0: float,
    ) -> None:
        self.t = self._validate_time(t)
        self.a0 = self._validate_scalar(a0, "a0")
        self.a = self._validate_coefficients(a, "a")
        self.b = self._validate_coefficients(b, "b")
        if self.a.size != self.b.size:
            raise ValueError("a and b must contain the same number of coefficients")
        self.omega0 = self._validate_omega0(omega0)

    @staticmethod
    def _validate_time(t: Sequence[float]) -> np.ndarray:
        """Validate the time values used for reconstruction."""
        time = np.asarray(t, dtype=float)
        if time.ndim != 1 or time.size == 0:
            raise ValueError("t must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(time)):
            raise ValueError("t must contain only finite values")
        return time.copy()

    @staticmethod
    def _validate_scalar(value: float, name: str) -> float:
        """Validate a finite scalar coefficient or parameter."""
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        return float(scalar)

    @staticmethod
    def _validate_coefficients(
        coefficients: Sequence[float], name: str
    ) -> np.ndarray:
        """Validate and copy a one-dimensional coefficient array."""
        values = np.asarray(coefficients, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values.copy()

    @staticmethod
    def _validate_omega0(omega0: float) -> float:
        """Validate the positive fundamental angular frequency."""
        value = FourierSynthesizer._validate_scalar(omega0, "omega0")
        if value <= 0:
            raise ValueError("omega0 must be positive")
        return value

    def _validate_harmonic_count(self, num_harmonics: int) -> int:
        """Validate a requested reconstruction harmonic count."""
        if isinstance(num_harmonics, bool) or not isinstance(
            num_harmonics, (int, np.integer)
        ):
            raise TypeError("num_harmonics must be a non-negative integer")
        if num_harmonics < 0:
            raise ValueError("num_harmonics must be a non-negative integer")
        if num_harmonics > self.available_harmonics:
            raise ValueError(
                "requested harmonics exceed the number of available coefficients"
            )
        return int(num_harmonics)

    @property
    def available_harmonics(self) -> int:
        """Number of harmonics available in the supplied coefficient arrays."""
        return self.a.size

    def dc_component(self) -> np.ndarray:
        """Return the constant Fourier Series contribution ``a0 / 2``."""
        return np.full_like(self.t, self.a0 / 2.0, dtype=float)

    def harmonic(self, harmonic_number: int) -> np.ndarray:
        """Return one harmonic component without its DC contribution.

        For harmonic ``n``, the returned values are:

        ``a_n cos(n omega0 t) + b_n sin(n omega0 t)``.
        """
        if isinstance(harmonic_number, bool) or not isinstance(
            harmonic_number, (int, np.integer)
        ):
            raise TypeError("harmonic_number must be a positive integer")
        if harmonic_number <= 0:
            raise ValueError("harmonic_number must be a positive integer")
        if harmonic_number > self.available_harmonics:
            raise ValueError("requested harmonic is not available")

        index = int(harmonic_number) - 1
        angle = harmonic_number * self.omega0 * self.t
        return self.a[index] * np.cos(angle) + self.b[index] * np.sin(angle)

    def get_harmonics(self, num_harmonics: int) -> np.ndarray:
        """Return individual harmonics 1 through N as rows of an array."""
        count = self._validate_harmonic_count(num_harmonics)
        components = np.empty((count, self.t.size), dtype=float)
        for harmonic_number in range(1, count + 1):
            components[harmonic_number - 1] = self.harmonic(harmonic_number)
        return components

    def reconstruct(self, num_harmonics: int) -> np.ndarray:
        """Reconstruct the signal using the DC term and the first N harmonics."""
        count = self._validate_harmonic_count(num_harmonics)
        reconstructed = self.dc_component()
        if count == 0:
            return reconstructed

        angles = self.omega0 * self.t[:, np.newaxis]
        harmonic_numbers = np.arange(1, count + 1, dtype=float)
        angles = angles * harmonic_numbers[np.newaxis, :]
        contributions = (
            np.cos(angles) * self.a[:count]
            + np.sin(angles) * self.b[:count]
        )
        return reconstructed + np.sum(contributions, axis=1)
