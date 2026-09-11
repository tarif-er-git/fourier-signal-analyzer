"""Magnitude and phase information for Fourier Series coefficients."""

from collections.abc import Sequence

import numpy as np


class FourierSpectrum:
    """Convert trigonometric Fourier coefficients to spectral quantities.

    For each harmonic n, the coefficient pair is converted according to:

    ``A_n = sqrt(a_n**2 + b_n**2)``

    ``phi_n = atan2(-b_n, a_n)``

    This matches the reconstruction form ``A_n cos(n omega0 t + phi_n)``.
    The phase of a zero-magnitude harmonic is defined as zero because its
    phase has no physical meaning.
    """

    def __init__(
        self, a0: float, a: Sequence[float], b: Sequence[float]
    ) -> None:
        self.a0 = self._validate_scalar(a0, "a0")
        self.a = self._validate_coefficients(a, "a")
        self.b = self._validate_coefficients(b, "b")
        if self.a.shape != self.b.shape:
            raise ValueError("a and b must have the same shape")

    @staticmethod
    def _validate_scalar(value: float, name: str) -> float:
        """Validate a finite scalar value."""
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        return float(scalar)

    @staticmethod
    def _validate_coefficients(
        coefficients: Sequence[float], name: str
    ) -> np.ndarray:
        """Validate and copy a one-dimensional finite coefficient array."""
        values = np.asarray(coefficients, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values.copy()

    @property
    def dc_value(self) -> float:
        """Return the constant Fourier Series value ``a0 / 2``."""
        return self.a0 / 2.0

    @property
    def harmonics(self) -> np.ndarray:
        """Return harmonic indices ``[1, 2, ..., N]``."""
        return np.arange(1, self.a.size + 1, dtype=int)

    def magnitude(self) -> np.ndarray:
        """Return ``sqrt(a_n**2 + b_n**2)`` for each harmonic."""
        return np.sqrt(self.a**2 + self.b**2)

    def phase(self) -> np.ndarray:
        """Return ``atan2(-b_n, a_n)`` for each harmonic.

        A zero-magnitude harmonic receives phase zero by convention because
        its phase cannot affect the reconstructed signal.
        """
        magnitudes = self.magnitude()
        phases = np.arctan2(-self.b, self.a)
        zero_magnitude = np.isclose(magnitudes, 0.0, atol=1e-12)
        return np.where(zero_magnitude, 0.0, phases)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return harmonic indices, magnitudes, and phases together."""
        return {
            "harmonic": self.harmonics,
            "magnitude": self.magnitude(),
            "phase": self.phase(),
        }
