"""Epicycle geometry for complex 2D Fourier curve coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourier.curve2d_analysis import CurveFourierResult


@dataclass(frozen=True)
class Curve2DEpicycleFrame:
    """One planar epicycle-chain frame."""

    centers: np.ndarray
    endpoints: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    endpoint: complex


class Curve2DEpicycle:
    """Calculate rotating planar vectors from an existing 2D Fourier result."""

    def __init__(self, coefficients: CurveFourierResult) -> None:
        self.harmonics = np.asarray(coefficients.harmonics, dtype=int).copy()
        self.coefficients = (
            np.asarray(coefficients.x_coefficients, dtype=complex)
            + 1j * np.asarray(coefficients.y_coefficients, dtype=complex)
        )
        if self.harmonics.ndim != 1 or self.coefficients.shape != self.harmonics.shape:
            raise ValueError("2D Fourier coefficients must have matching one-dimensional arrays")
        if self.harmonics.size == 0 or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("2D Fourier coefficients must be non-empty and finite")
        if not np.array_equal(self.harmonics, np.sort(self.harmonics)):
            raise ValueError("harmonics must be sorted in ascending order")
        if 0 not in self.harmonics:
            raise ValueError("2D Fourier coefficients must include the DC harmonic")
        self._dc_index = int(np.flatnonzero(self.harmonics == 0)[0])

    @property
    def available_harmonics(self) -> int:
        """Return the largest absolute harmonic available."""
        return int(np.max(np.abs(self.harmonics)))

    def selected_harmonics(self, harmonic_count: int) -> np.ndarray:
        """Return DC plus the symmetric ``-N...+N`` harmonic set."""
        count = self._validate_harmonic_count(harmonic_count)
        return self.harmonics[np.abs(self.harmonics) <= count]

    def _selected_indices(self, harmonic_count: int) -> np.ndarray:
        count = self._validate_harmonic_count(harmonic_count)
        return np.flatnonzero(np.abs(self.harmonics) <= count)

    def _validate_harmonic_count(self, harmonic_count: int) -> int:
        if isinstance(harmonic_count, bool) or not isinstance(
            harmonic_count, (int, np.integer)
        ):
            raise TypeError("harmonic_count must be an integer")
        if harmonic_count < 0 or harmonic_count > self.available_harmonics:
            raise ValueError("harmonic_count is outside the available harmonic range")
        return int(harmonic_count)

    def frame(self, parameter: float, harmonic_count: int) -> Curve2DEpicycleFrame:
        """Return chain geometry at normalized periodic parameter ``[0, 1)``."""
        parameter_value = float(parameter)
        if not np.isfinite(parameter_value):
            raise ValueError("parameter must be finite")
        if not 0.0 <= parameter_value <= 1.0:
            raise ValueError("parameter must be between 0 and 1")
        indices = self._selected_indices(harmonic_count)
        harmonics = self.harmonics[indices]
        terms = self.coefficients[indices]
        rotating_terms = terms * np.exp(2j * np.pi * harmonics * parameter_value)
        endpoint = complex(np.sum(rotating_terms))
        centers_complex = np.concatenate(([0.0 + 0.0j], np.cumsum(rotating_terms)[:-1]))
        endpoints_complex = np.cumsum(rotating_terms)
        centers = np.column_stack((centers_complex.real, centers_complex.imag))
        endpoints = np.column_stack((endpoints_complex.real, endpoints_complex.imag))
        return Curve2DEpicycleFrame(
            centers=centers,
            endpoints=endpoints,
            radii=np.abs(terms),
            angles=np.angle(rotating_terms),
            endpoint=endpoint,
        )

    def endpoint(self, parameter: float, harmonic_count: int) -> complex:
        """Return the final chain endpoint, matching 2D reconstruction."""
        return self.frame(parameter, harmonic_count).endpoint
