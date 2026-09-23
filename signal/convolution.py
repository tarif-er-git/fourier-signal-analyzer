"""Numerical convolution analysis for the Fourier Craft application.

Mathematical Convention
-----------------------
This module implements the standard continuous-time convolution definition,
approximated numerically using sampled signals:

    y(t) = (x * h)(t) = integral x(tau) h(t - tau) dtau

The three-step process visualized in the application is:

    Step 1 - Fix x(tau):
        The signal x remains fixed on the common tau axis.

    Step 2 - Reverse h:
        h_rev(tau) = h(-tau)
        This is a time-reversal of h about tau = 0.

    Step 3 - Shift the reversed signal by t:
        h_shifted(tau; t) = h(t - tau) = h_rev(tau - t)
        Shifting h_rev to the right by t gives h(t - tau).

    Step 4 - Multiply and integrate:
        y(t) = sum_n  x(tau_n) * h(t - tau_n) * dt

The discrete approximation uses the common sampling interval dt as a
scale factor so that the numerical result approximates the continuous-time
integral (i.e. it is NOT a raw unscaled dot product).

Reference Convolution
---------------------
An independent reference is computed using:

    y_ref = numpy.convolve(x_resampled, h_resampled, mode="full") * dt

The output time axis for "full" convolution of two N-sample signals spans
[t_x_start + t_h_start,  t_x_end + t_h_end]
and contains 2*N - 1 samples.

The animation's frame-by-frame values match y_ref within floating-point
numerical tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvolutionResult:
    """All precomputed arrays needed to drive the convolution animation.

    Attributes
    ----------
    tau
        Common time axis on which both x and h are resampled. Length N.
    x_resampled
        Signal x(tau) evaluated on the common grid.
    h_resampled
        Signal h(tau) evaluated on the common grid (original orientation,
        NOT reversed). Reversal is applied on the fly during animation.
    dt
        Uniform sampling interval Δτ of the common grid.
    output_t
        Output time axis for y(t). Length 2*N - 1, spanning
        [tau[0]+tau[0], tau[-1]+tau[-1]].
    y_reference
        Complete reference convolution y(t) = convolve(x, h, 'full') * dt.
        Length 2*N - 1.
    x_name
        Human-readable label for the x signal.
    h_name
        Human-readable label for the h signal.
    """

    tau: np.ndarray
    x_resampled: np.ndarray
    h_resampled: np.ndarray
    dt: float
    output_t: np.ndarray
    y_reference: np.ndarray
    x_name: str = "x(t)"
    h_name: str = "h(t)"

    @property
    def num_samples(self) -> int:
        """Number of samples on the common tau grid."""
        return int(self.tau.size)

    @property
    def num_frames(self) -> int:
        """Number of animation frames (one per output sample)."""
        return int(self.output_t.size)


# ---------------------------------------------------------------------------
# Built-in convolution-friendly signal generators
# ---------------------------------------------------------------------------

def _rectangular_pulse(tau: np.ndarray, width: float = 0.5) -> np.ndarray:
    """Return 1 for |tau| <= width/2, else 0."""
    return np.where(np.abs(tau) <= width / 2.0, 1.0, 0.0)


def _exponential_decay(tau: np.ndarray, decay: float = 5.0) -> np.ndarray:
    """Causal exponential decay: exp(-decay * tau) for tau >= 0, else 0."""
    return np.where(tau >= 0.0, np.exp(-decay * np.maximum(tau, 0.0)), 0.0)


def _triangle_pulse(tau: np.ndarray, width: float = 0.5) -> np.ndarray:
    """Triangular pulse: peak 1 at tau=0, zero outside [-width/2, width/2]."""
    half = width / 2.0
    return np.maximum(0.0, 1.0 - np.abs(tau) / half)


def _sine_burst(tau: np.ndarray, frequency: float = 3.0, width: float = 0.8) -> np.ndarray:
    """Sine wave windowed by a rectangular pulse of given width."""
    rect = np.where(np.abs(tau) <= width / 2.0, 1.0, 0.0)
    return np.sin(2.0 * np.pi * frequency * tau) * rect


# Public registry of preset signals for both x and h inputs
CONVOLUTION_PRESETS: dict[str, tuple[str, object]] = {
    "Rectangular Pulse": ("Rectangular Pulse", _rectangular_pulse),
    "Triangle Pulse": ("Triangle Pulse", _triangle_pulse),
    "Exponential Decay": ("Exponential Decay", _exponential_decay),
    "Sine Burst": ("Sine Burst", _sine_burst),
    "Sine": ("Sine", lambda tau: np.sin(2.0 * np.pi * tau)),
    "Sawtooth": ("Sawtooth", lambda tau: 2.0 * np.mod(tau + 0.5, 1.0) - 1.0),
}

X_PRESET_NAMES: list[str] = [
    "Rectangular Pulse",
    "Triangle Pulse",
    "Sine Burst",
    "Sine",
]

H_PRESET_NAMES: list[str] = [
    "Rectangular Pulse",
    "Exponential Decay",
    "Triangle Pulse",
    "Sine Burst",
]


# ---------------------------------------------------------------------------
# Convolution analyzer
# ---------------------------------------------------------------------------

class ConvolutionAnalyzer:
    """Prepare and evaluate the convolution of two 1D sampled signals.

    This class keeps pure numerical logic separate from the GUI and
    visualization layers. All methods are stateless and operate on
    :class:`ConvolutionResult` objects returned by :meth:`prepare`.

    Numerical Convention
    --------------------
    Both input signals are resampled onto a common uniform tau grid of
    length N. The output has 2*N - 1 samples (full convolution mode).
    Each output value is::

        y[k] = sum_{n=0}^{N-1}  x[n] * h_shifted[n; k]  * dt

    where ``h_shifted[n; k] = h(output_t[k] - tau[n])``.
    This scale factor ``dt`` makes the sum a numerical approximation of
    the continuous integral rather than a dimensionless correlation.
    """

    NUM_SAMPLES: int = 256
    TAU_RANGE: float = 1.0  # Symmetric tau range [-1, 1]

    @classmethod
    def prepare_from_arrays(
        cls,
        t_x: np.ndarray,
        x: np.ndarray,
        t_h: np.ndarray,
        h: np.ndarray,
        num_samples: int = 256,
        x_name: str = "x(t)",
        h_name: str = "h(t)",
    ) -> ConvolutionResult:
        """Prepare a :class:`ConvolutionResult` from arbitrary sampled signals.

        Both signals are resampled onto a common uniform grid spanning the
        union of their time ranges.

        Parameters
        ----------
        t_x, x
            Time and amplitude arrays for signal x. Must be finite,
            one-dimensional, and have at least 2 samples.
        t_h, h
            Time and amplitude arrays for signal h.
        num_samples
            Number of samples on the common tau grid. Must be >= 4.
        x_name, h_name
            Human-readable names for the two signals.

        Returns
        -------
        ConvolutionResult
            Fully prepared convolution data.

        Raises
        ------
        ValueError
            If any input array is invalid or has too few samples.
        """
        t_x = np.asarray(t_x, dtype=float)
        x = np.asarray(x, dtype=float)
        t_h = np.asarray(t_h, dtype=float)
        h = np.asarray(h, dtype=float)

        cls._validate_signal(t_x, x, "x")
        cls._validate_signal(t_h, h, "h")

        if num_samples < 4:
            raise ValueError("num_samples must be at least 4")

        # Build common tau axis covering the union of both time ranges
        tau_start = min(float(t_x[0]), float(t_h[0]))
        tau_end = max(float(t_x[-1]), float(t_h[-1]))
        if tau_end <= tau_start:
            raise ValueError("Combined time range has zero length")

        tau = np.linspace(tau_start, tau_end, num_samples)
        dt = float(tau[1] - tau[0])

        x_resampled = np.interp(tau, t_x, x, left=0.0, right=0.0)
        h_resampled = np.interp(tau, t_h, h, left=0.0, right=0.0)

        # Reference convolution scaled by dt
        y_ref_raw = np.convolve(x_resampled, h_resampled, mode="full")
        y_reference = y_ref_raw * dt

        # Output time axis: tau[0]+tau[0] ... tau[-1]+tau[-1]
        output_t = np.linspace(
            tau_start * 2.0,
            tau_end * 2.0,
            2 * num_samples - 1,
        )

        return ConvolutionResult(
            tau=tau,
            x_resampled=x_resampled,
            h_resampled=h_resampled,
            dt=dt,
            output_t=output_t,
            y_reference=y_reference,
            x_name=x_name,
            h_name=h_name,
        )

    @classmethod
    def prepare_signals(
        cls,
        x_input: str | tuple[np.ndarray, np.ndarray],
        h_input: str | tuple[np.ndarray, np.ndarray],
        num_samples: int = 256,
        x_name: str | None = None,
        h_name: str | None = None,
    ) -> ConvolutionResult:
        """Prepare convolution from any combination of presets and custom signals.

        Supports all four combinations:
          1. x(t) = Preset, h(t) = Preset
          2. x(t) = Custom, h(t) = Preset
          3. x(t) = Preset, h(t) = Custom
          4. x(t) = Custom, h(t) = Custom

        Parameters
        ----------
        x_input : str or tuple of (t, x)
            Preset name or (time, values) tuple for signal x.
        h_input : str or tuple of (t, h)
            Preset name or (time, values) tuple for signal h.
        num_samples : int
            Number of samples on the common tau grid.
        x_name : str, optional
            Display label for signal x.
        h_name : str, optional
            Display label for signal h.

        Returns
        -------
        ConvolutionResult
        """
        # Process x input
        if isinstance(x_input, str):
            if x_input not in CONVOLUTION_PRESETS:
                raise ValueError(f"Unknown x preset: {x_input!r}")
            t_x = np.linspace(-1.0, 1.0, num_samples)
            label, fn = CONVOLUTION_PRESETS[x_input]
            vals_x = np.asarray(fn(t_x), dtype=float)
            final_x_name = x_name or label
        elif isinstance(x_input, (tuple, list)) and len(x_input) == 2:
            t_x = np.asarray(x_input[0], dtype=float)
            vals_x = np.asarray(x_input[1], dtype=float)
            final_x_name = x_name or "Custom x(t)"
        else:
            raise ValueError(
                f"Invalid x_input: expected preset name or (t, x) tuple, got {type(x_input)}"
            )

        # Process h input
        if isinstance(h_input, str):
            if h_input not in CONVOLUTION_PRESETS:
                raise ValueError(f"Unknown h preset: {h_input!r}")
            t_h = np.linspace(-1.0, 1.0, num_samples)
            label, fn = CONVOLUTION_PRESETS[h_input]
            vals_h = np.asarray(fn(t_h), dtype=float)
            final_h_name = h_name or label
        elif isinstance(h_input, (tuple, list)) and len(h_input) == 2:
            t_h = np.asarray(h_input[0], dtype=float)
            vals_h = np.asarray(h_input[1], dtype=float)
            final_h_name = h_name or "Custom h(t)"
        else:
            raise ValueError(
                f"Invalid h_input: expected preset name or (t, h) tuple, got {type(h_input)}"
            )

        return cls.prepare_from_arrays(
            t_x=t_x,
            x=vals_x,
            t_h=t_h,
            h=vals_h,
            num_samples=num_samples,
            x_name=final_x_name,
            h_name=final_h_name,
        )

    @classmethod
    def prepare_from_presets(
        cls,
        x_preset_name: str,
        h_preset_name: str,
        num_samples: int = 256,
        tau_start: float = -1.0,
        tau_end: float = 1.0,
    ) -> ConvolutionResult:
        """Prepare a :class:`ConvolutionResult` from named preset signals.

        Parameters
        ----------
        x_preset_name
            Name of the x signal preset (see :data:`CONVOLUTION_PRESETS`).
        h_preset_name
            Name of the h signal preset.
        num_samples
            Number of samples on the common grid.
        tau_start, tau_end
            Time range for the common tau axis.

        Returns
        -------
        ConvolutionResult
        """
        if x_preset_name not in CONVOLUTION_PRESETS:
            raise ValueError(f"Unknown x preset: {x_preset_name!r}")
        if h_preset_name not in CONVOLUTION_PRESETS:
            raise ValueError(f"Unknown h preset: {h_preset_name!r}")

        tau = np.linspace(tau_start, tau_end, num_samples)
        x_label, x_fn = CONVOLUTION_PRESETS[x_preset_name]
        h_label, h_fn = CONVOLUTION_PRESETS[h_preset_name]
        x_vals = np.asarray(x_fn(tau), dtype=float)
        h_vals = np.asarray(h_fn(tau), dtype=float)
        dt = float(tau[1] - tau[0])

        y_ref_raw = np.convolve(x_vals, h_vals, mode="full")
        y_reference = y_ref_raw * dt

        output_t = np.linspace(tau_start * 2.0, tau_end * 2.0, 2 * num_samples - 1)

        return ConvolutionResult(
            tau=tau,
            x_resampled=x_vals,
            h_resampled=h_vals,
            dt=dt,
            output_t=output_t,
            y_reference=y_reference,
            x_name=x_label,
            h_name=h_label,
        )

    @staticmethod
    def h_shifted(result: ConvolutionResult, t_shift: float) -> np.ndarray:
        """Return h(t_shift − τ) evaluated at each point of result.tau.

        This implements the two-step animation process:
        1. Reverse h: h_rev(τ) = h(−τ)
        2. Shift by t: h_shifted(τ) = h(t − τ) = h_rev(τ − t)

        The returned values are interpolated from the precomputed h samples
        using np.interp with zero padding outside the defined range.

        Parameters
        ----------
        result
            Prepared convolution data.
        t_shift
            Current shift value t. Corresponds to output_t[k] at frame k.

        Returns
        -------
        numpy.ndarray
            h(t_shift − τ) for each τ in result.tau. Shape matches result.tau.
        """
        # h(t - τ) means: evaluate h at argument (t - τ)
        # result.tau contains τ values; we need h evaluated at (t - τ)
        query_points = t_shift - result.tau
        # Interpolate h from its original (tau, h_resampled) samples
        return np.interp(
            query_points,
            result.tau,
            result.h_resampled,
            left=0.0,
            right=0.0,
        )

    @staticmethod
    def overlap_product(
        result: ConvolutionResult, t_shift: float
    ) -> np.ndarray:
        """Return element-wise product x(τ) * h(t_shift − τ).

        This is the integrand of the convolution at shift t_shift.
        Integrating this over τ (multiplied by Δτ) yields one sample y(t).

        Parameters
        ----------
        result
            Prepared convolution data.
        t_shift
            Current shift t.

        Returns
        -------
        numpy.ndarray
            x(τ) · h(t_shift − τ) for each τ in result.tau.
        """
        h_slid = ConvolutionAnalyzer.h_shifted(result, t_shift)
        return result.x_resampled * h_slid

    @staticmethod
    def convolution_value_at_shift(
        result: ConvolutionResult, t_shift: float
    ) -> float:
        """Calculate y(t_shift) = Σ x(τ) h(t_shift − τ) Δτ directly.

        This is distinct from looking up y_reference[k]; it recomputes
        the value from scratch, demonstrating the actual convolution sum.
        The result should match y_reference to floating-point tolerance.

        Parameters
        ----------
        result
            Prepared convolution data.
        t_shift
            Shift value t at which to evaluate y.

        Returns
        -------
        float
            The convolution value y(t_shift).
        """
        product = ConvolutionAnalyzer.overlap_product(result, t_shift)
        return float(np.sum(product) * result.dt)

    @staticmethod
    def _validate_signal(t: np.ndarray, x: np.ndarray, name: str) -> None:
        """Raise ValueError if the signal arrays are invalid."""
        if t.ndim != 1 or x.ndim != 1:
            raise ValueError(f"Signal {name}: time and values must be 1D")
        if t.size != x.size:
            raise ValueError(f"Signal {name}: time and value arrays must have the same length")
        if t.size < 2:
            raise ValueError(f"Signal {name}: at least 2 samples required")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"Signal {name}: time array contains non-finite values")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"Signal {name}: value array contains non-finite values")
        if np.ptp(t) <= 0.0:
            raise ValueError(f"Signal {name}: time range must be positive")
