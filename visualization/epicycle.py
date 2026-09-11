"""Epicycle geometry for visualizing Fourier Series harmonics.

A Fourier term ``A_n cos(n omega0 t + phi_n)`` becomes a rotating vector
with radius ``A_n`` and angle ``n omega0 t + phi_n``. The horizontal
component of each vector is the corresponding real Fourier harmonic, so the
x-coordinate of the final chain endpoint equals the Fourier reconstruction.
The vertical coordinate is used only to show the rotating-vector geometry.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from fourier.spectrum import FourierSpectrum


@dataclass(frozen=True)
class EpicycleFrame:
    """Vector-chain positions for one animation time."""

    centers: np.ndarray
    endpoints: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    reconstructed_value: float


class EpicycleVisualizer:
    """Calculate epicycle vector positions from Fourier coefficients."""

    def __init__(
        self,
        a0: float,
        a: Sequence[float],
        b: Sequence[float],
        omega0: float,
    ) -> None:
        self.a0 = self._validate_scalar(a0, "a0")
        self.a = self._validate_coefficients(a, "a")
        self.b = self._validate_coefficients(b, "b")
        if self.a.shape != self.b.shape:
            raise ValueError("a and b must have the same shape")
        self.omega0 = self._validate_scalar(omega0, "omega0")
        if self.omega0 <= 0:
            raise ValueError("omega0 must be positive")

        spectrum = FourierSpectrum(self.a0, self.a, self.b)
        self.dc_value = spectrum.dc_value
        self.radii = spectrum.magnitude()
        self.phases = spectrum.phase()
        self.harmonic_numbers = spectrum.harmonics

    @staticmethod
    def _validate_scalar(value: float, name: str) -> float:
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        return float(scalar)

    @staticmethod
    def _validate_coefficients(
        coefficients: Sequence[float], name: str
    ) -> np.ndarray:
        values = np.asarray(coefficients, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values.copy()

    @property
    def available_harmonics(self) -> int:
        """Return the number of available harmonic coefficient pairs."""
        return self.a.size

    def _validate_harmonics(self, num_harmonics: int) -> int:
        if isinstance(num_harmonics, bool) or not isinstance(
            num_harmonics, (int, np.integer)
        ):
            raise TypeError("num_harmonics must be a non-negative integer")
        if num_harmonics < 0 or num_harmonics > self.available_harmonics:
            raise ValueError("num_harmonics is outside the available range")
        return int(num_harmonics)

    def frame(self, time: float, num_harmonics: int) -> EpicycleFrame:
        """Calculate vector-chain centers, endpoints, and scalar reconstruction."""
        animation_time = self._validate_scalar(time, "time")
        count = self._validate_harmonics(num_harmonics)
        radii = self.radii[:count]
        harmonics = self.harmonic_numbers[:count]
        angles = harmonics * self.omega0 * animation_time + self.phases[:count]
        vectors = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

        centers = np.zeros((count, 2), dtype=float)
        endpoints = np.zeros((count, 2), dtype=float)
        current = np.array([self.dc_value, 0.0], dtype=float)
        for index, vector in enumerate(vectors):
            centers[index] = current
            current = current + vector
            endpoints[index] = current
        return EpicycleFrame(
            centers=centers,
            endpoints=endpoints,
            radii=radii.copy(),
            angles=angles,
            reconstructed_value=float(current[0]),
        )

    def reconstruct(self, time: float, num_harmonics: int) -> float:
        """Return the chain endpoint x-coordinate at one time."""
        return self.frame(time, num_harmonics).reconstructed_value

    def make_animation(
        self,
        figure,
        axes,
        *,
        num_harmonics: int,
        frame_count: int = 240,
        interval_ms: int = 30,
        trace_length: int = 160,
    ):
        """Create a Matplotlib ``FuncAnimation`` for this epicycle model.

        The caller owns the returned animation and must retain it while the
        animation is running. Geometry is recalculated once per animation
        frame; coefficient-derived radii and phases are precomputed.
        """
        from matplotlib.animation import FuncAnimation

        count = self._validate_harmonics(num_harmonics)
        if count == 0:
            raise ValueError("epicycle animation requires at least one harmonic")
        if frame_count <= 0 or interval_ms <= 0 or trace_length <= 0:
            raise ValueError("animation sizes and interval must be positive")
        vector_lines = [axes.plot([], [], "o-", linewidth=1.2)[0] for _ in range(count)]
        endpoint_marker, = axes.plot([], [], "o", color="tab:red")
        trace_line, = axes.plot([], [], color="tab:red", alpha=0.7)
        trace_times: list[float] = []
        trace_values: list[float] = []
        period = 2.0 * np.pi / self.omega0

        def update(frame_number: int):
            time = period * frame_number / frame_count
            frame = self.frame(time, count)
            for line, center, endpoint in zip(
                vector_lines, frame.centers, frame.endpoints
            ):
                line.set_data([center[0], endpoint[0]], [center[1], endpoint[1]])
            endpoint_marker.set_data([frame.endpoints[-1, 0]], [frame.endpoints[-1, 1]])
            trace_times.append(time)
            trace_values.append(frame.reconstructed_value)
            trace_line.set_data(trace_times[-trace_length:], trace_values[-trace_length:])
            return (*vector_lines, endpoint_marker, trace_line)

        return FuncAnimation(
            figure,
            update,
            frames=frame_count,
            interval=interval_ms,
            blit=True,
            repeat=True,
        )
