"""Numerical trigonometric Fourier Series coefficient analysis."""

from typing import Literal

import numpy as np

EndpointMode = Literal["auto", "included", "excluded"]


class FourierAnalyzer:
    """Calculate Fourier Series coefficients from one sampled period.

    The input may use either of these conventions:

    * ``[t0, t0 + T]`` with the periodic boundary repeated as the last
      sample. The duplicate is removed before integration.
    * ``[t0, t0 + T)`` without the repeated boundary. The period is inferred
      by adding one uniform sample interval to the covered time span.

    ``endpoint_mode="auto"`` detects the first convention when the first and
    last signal values are approximately equal. The numerical integral uses
    the trapezoidal rule after appending one periodic copy of the first sample
    at ``t0 + T``. This avoids counting a duplicated endpoint as an extra
    independent sample.
    """

    def __init__(
        self,
        t: np.ndarray,
        x: np.ndarray,
        num_harmonics: int,
        *,
        endpoint_mode: EndpointMode = "auto",
    ) -> None:
        self._time, self._values = self._validate_inputs(t, x)
        self._validate_harmonics(num_harmonics)
        if endpoint_mode not in ("auto", "included", "excluded"):
            raise ValueError("endpoint_mode must be 'auto', 'included', or 'excluded'")

        self.num_harmonics = int(num_harmonics)
        self.endpoint_mode = endpoint_mode
        self._sample_spacing = self._validate_uniform_spacing(self._time)
        self._endpoint_included = self._determine_endpoint_mode()
        self._period, self._integration_time, self._integration_values = (
            self._prepare_integration_samples()
        )
        self.omega0 = 2.0 * np.pi / self._period

        self.a0: float | None = None
        self.a: np.ndarray | None = None
        self.b: np.ndarray | None = None

    @staticmethod
    def _validate_inputs(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate the sampled time and signal arrays."""
        time = np.asarray(t, dtype=float)
        values = np.asarray(x, dtype=float)
        if time.ndim != 1 or values.ndim != 1:
            raise ValueError("t and x must be one-dimensional")
        if time.size != values.size:
            raise ValueError("t and x must have the same length")
        if time.size < 2:
            raise ValueError("at least two samples are required")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
            raise ValueError("t and x must contain only finite values")
        if not np.all(np.diff(time) > 0):
            raise ValueError("time values must be strictly increasing")
        return time, values

    @staticmethod
    def _validate_harmonics(num_harmonics: int) -> None:
        """Validate the requested positive harmonic count."""
        if isinstance(num_harmonics, bool) or not isinstance(
            num_harmonics, (int, np.integer)
        ):
            raise TypeError("num_harmonics must be a positive integer")
        if num_harmonics <= 0:
            raise ValueError("num_harmonics must be a positive integer")

    @staticmethod
    def _validate_uniform_spacing(time: np.ndarray) -> float:
        """Return the sample spacing after checking that it is uniform."""
        differences = np.diff(time)
        spacing = float(np.median(differences))
        if spacing <= 0 or not np.allclose(
            differences, spacing, rtol=1e-5, atol=1e-12
        ):
            raise ValueError("time values must be uniformly sampled")
        return spacing

    def _determine_endpoint_mode(self) -> bool:
        """Determine whether the input contains a repeated periodic endpoint."""
        if self.endpoint_mode == "included":
            return True
        if self.endpoint_mode == "excluded":
            return False
        return bool(np.isclose(self._values[0], self._values[-1]))

    def _prepare_integration_samples(
        self,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Prepare unique samples and one periodic endpoint for integration."""
        if self._endpoint_included:
            if self._time.size < 3:
                raise ValueError(
                    "at least three samples are required with an included endpoint"
                )
            period = float(self._time[-1] - self._time[0])
            integration_time = self._time[:-1]
            integration_values = self._values[:-1]
        else:
            period = float(self._time[-1] - self._time[0] + self._sample_spacing)
            integration_time = self._time
            integration_values = self._values

        if period <= 0:
            raise ValueError("period must be positive")
        if integration_time.size < 2:
            raise ValueError("at least two unique period samples are required")

        periodic_time = np.concatenate(
            (integration_time, np.array([integration_time[0] + period]))
        )
        periodic_values = np.concatenate(
            (integration_values, np.array([integration_values[0]]))
        )
        return period, periodic_time, periodic_values

    def _integrate(self, values: np.ndarray) -> float:
        """Approximate one period integral using the trapezoidal rule."""
        return float(np.trapezoid(values, self._integration_time))

    def analyze(self) -> "FourierAnalyzer":
        """Calculate and store ``a0``, cosine coefficients ``a``, and ``b``.

        The returned coefficient arrays use zero-based Python indexing:
        ``a[n - 1]`` and ``b[n - 1]`` contain the coefficients for harmonic
        ``n``. The constant term in a synthesized series is ``a0 / 2``.
        """
        relative_time = self._integration_time - self._integration_time[0]
        periodic_relative_time = relative_time
        signal_values = self._integration_values

        self.a0 = 2.0 / self._period * self._integrate(signal_values)
        self.a = np.zeros(self.num_harmonics, dtype=float)
        self.b = np.zeros(self.num_harmonics, dtype=float)

        for harmonic in range(1, self.num_harmonics + 1):
            angle = harmonic * self.omega0 * periodic_relative_time
            self.a[harmonic - 1] = 2.0 / self._period * self._integrate(
                signal_values * np.cos(angle)
            )
            self.b[harmonic - 1] = 2.0 / self._period * self._integrate(
                signal_values * np.sin(angle)
            )

        return self

    @property
    def period(self) -> float:
        """The inferred period T of the sampled signal."""
        return self._period
