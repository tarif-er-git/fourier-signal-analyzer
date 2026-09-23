"""Tests for signal.convolution — ConvolutionAnalyzer and ConvolutionResult.

Tests cover:
  - Rectangular x rectangular → expected trapezoidal/triangular shape
  - Impulse x signal → signal recovered (within numerical tolerance)
  - Commutativity: x * h ≈ h * x
  - Scaling: (a*x) * h = a * (x * h)
  - Reference comparison: frame-by-frame values match y_reference
  - h_shifted correctness: reversals and edge cases
  - Preset loading: all defined preset combinations
  - Edge case validation: bad inputs raise ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from signal.convolution import (
    ConvolutionAnalyzer,
    ConvolutionResult,
    CONVOLUTION_PRESETS,
    X_PRESET_NAMES,
    H_PRESET_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tau(n: int = 128) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n)


def _rect(tau: np.ndarray, width: float = 0.5) -> np.ndarray:
    return np.where(np.abs(tau) <= width / 2.0, 1.0, 0.0)


def _impulse_approx(tau: np.ndarray, center: float = 0.0) -> np.ndarray:
    """A narrow Gaussian approximating a unit impulse, normalized so area ≈ 1."""
    sigma = 0.04
    g = np.exp(-0.5 * ((tau - center) / sigma) ** 2)
    dt = float(tau[1] - tau[0])
    return g / (np.sum(g) * dt)


def _prepare_from_arrays(t, x, t_h, h, n=128) -> ConvolutionResult:
    return ConvolutionAnalyzer.prepare_from_arrays(t, x, t_h, h, num_samples=n)


def _prepare_rect_rect(n: int = 256) -> ConvolutionResult:
    return ConvolutionAnalyzer.prepare_from_presets(
        "Rectangular Pulse", "Rectangular Pulse", num_samples=n
    )


# ---------------------------------------------------------------------------
# 1. ConvolutionResult structure
# ---------------------------------------------------------------------------

class TestConvolutionResult:
    def test_num_samples(self) -> None:
        result = _prepare_rect_rect(n=128)
        assert result.num_samples == 128

    def test_num_frames(self) -> None:
        result = _prepare_rect_rect(n=128)
        assert result.num_frames == 2 * 128 - 1

    def test_output_t_length(self) -> None:
        result = _prepare_rect_rect(n=64)
        assert result.output_t.size == 2 * 64 - 1

    def test_y_reference_length(self) -> None:
        result = _prepare_rect_rect(n=64)
        assert result.y_reference.size == result.output_t.size

    def test_dt_positive(self) -> None:
        result = _prepare_rect_rect()
        assert result.dt > 0.0

    def test_common_tau_uniform(self) -> None:
        result = _prepare_rect_rect(n=256)
        diffs = np.diff(result.tau)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Rectangular * Rectangular → trapezoidal / triangular result
# ---------------------------------------------------------------------------

class TestRectangularConvolution:
    """Convolving two equal-width rectangular pulses → triangle."""

    def test_output_is_triangle_shaped(self) -> None:
        result = _prepare_rect_rect(n=512)
        y = result.y_reference
        # The output should peak somewhere in the middle
        peak_idx = int(np.argmax(y))
        mid = len(y) // 2
        # Peak should be within 10% of center
        assert abs(peak_idx - mid) < len(y) * 0.10

    def test_output_is_non_negative(self) -> None:
        """Rect * Rect is always >= 0."""
        result = _prepare_rect_rect(n=512)
        # Allow small floating-point negatives
        assert float(np.min(result.y_reference)) >= -1e-10

    def test_output_is_zero_at_boundaries(self) -> None:
        """Convolution of two finite-support signals is zero outside their combined support."""
        result = _prepare_rect_rect(n=256)
        y = result.y_reference
        # First and last ~10% of output should be negligible
        margin = len(y) // 10
        assert np.allclose(y[:margin], 0.0, atol=1e-8)
        assert np.allclose(y[-margin:], 0.0, atol=1e-8)

    def test_output_peak_scales_with_dt(self) -> None:
        """Peak of rect * rect = width^2 (for width-1 rect, scaled by dt)."""
        # Both signals are Rectangular Pulse width=0.5 on tau ∈ [-1, 1]
        # Peak ≈ 0.5 * dt (after scaling by dt in the reference)
        result = _prepare_rect_rect(n=512)
        peak = float(np.max(result.y_reference))
        # width * dt: width=0.5, so peak ≈ 0.5
        assert 0.3 < peak < 0.7, f"Unexpected peak: {peak}"


# ---------------------------------------------------------------------------
# 3. Impulse response: x * delta ≈ x
# ---------------------------------------------------------------------------

class TestImpulseResponse:
    def test_impulse_convolution_recovers_signal(self) -> None:
        n = 512
        tau = np.linspace(-1.0, 1.0, n)
        x_vals = np.sin(2.0 * np.pi * tau)
        h_vals = _impulse_approx(tau, center=0.0)
        dt = float(tau[1] - tau[0])

        result = ConvolutionAnalyzer.prepare_from_arrays(
            tau, x_vals, tau, h_vals, num_samples=n
        )
        y = result.y_reference

        # y(t) should approximate x(t) around the center where overlap is complete
        t_eval = np.linspace(-0.5, 0.5, 100)
        y_eval = np.interp(t_eval, result.output_t, y)
        x_eval = np.sin(2.0 * np.pi * t_eval)
        np.testing.assert_allclose(y_eval, x_eval, atol=0.05)


# ---------------------------------------------------------------------------
# 4. Commutativity: x * h ≈ h * x
# ---------------------------------------------------------------------------

class TestCommutativity:
    def test_rect_triangle_commutativity(self) -> None:
        n = 256
        r1 = ConvolutionAnalyzer.prepare_from_presets(
            "Rectangular Pulse", "Triangle Pulse", num_samples=n
        )
        r2 = ConvolutionAnalyzer.prepare_from_presets(
            "Triangle Pulse", "Rectangular Pulse", num_samples=n
        )
        # The outputs should have the same shape and be approximately equal
        assert r1.y_reference.size == r2.y_reference.size
        np.testing.assert_allclose(r1.y_reference, r2.y_reference, atol=1e-10)

    def test_rect_rect_commutativity(self) -> None:
        n = 128
        r1 = _prepare_rect_rect(n)
        tau = r1.tau
        x = r1.x_resampled
        h = r1.h_resampled
        # swap x and h
        r2 = ConvolutionAnalyzer.prepare_from_arrays(tau, h, tau, x, num_samples=n)
        np.testing.assert_allclose(r1.y_reference, r2.y_reference, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Linearity / Scaling: (a*x) * h = a * (x * h)
# ---------------------------------------------------------------------------

class TestScaling:
    @pytest.mark.parametrize("a", [0.5, 2.0, -1.5, 0.0])
    def test_scaling_property(self, a: float) -> None:
        n = 128
        result_base = _prepare_rect_rect(n)
        tau = result_base.tau
        x_scaled = result_base.x_resampled * a
        result_scaled = ConvolutionAnalyzer.prepare_from_arrays(
            tau, x_scaled, tau, result_base.h_resampled, num_samples=n
        )
        expected = result_base.y_reference * a
        np.testing.assert_allclose(result_scaled.y_reference, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. h_shifted correctness
# ---------------------------------------------------------------------------

class TestHShifted:
    def test_h_shifted_at_zero_is_reversed(self) -> None:
        """h(0 - tau) = h(-tau): the reversal should mirror h around tau=0."""
        n = 128
        result = _prepare_rect_rect(n)
        h_at_zero_shift = ConvolutionAnalyzer.h_shifted(result, 0.0)
        # h_resampled is symmetric for rect pulse, so h(-tau) ≈ h(tau)
        np.testing.assert_allclose(
            h_at_zero_shift, result.h_resampled[::-1], atol=1e-10
        )

    def test_h_shifted_outside_range_is_zero(self) -> None:
        """Shifting far outside the support gives zero."""
        result = _prepare_rect_rect(128)
        h_far = ConvolutionAnalyzer.h_shifted(result, 10.0)  # far outside
        assert np.allclose(h_far, 0.0, atol=1e-10)

    def test_h_shifted_shape_matches_tau(self) -> None:
        result = _prepare_rect_rect(64)
        h_s = ConvolutionAnalyzer.h_shifted(result, 0.5)
        assert h_s.shape == result.tau.shape


# ---------------------------------------------------------------------------
# 7. Frame-by-frame matches reference
# ---------------------------------------------------------------------------

class TestFrameVsReference:
    def test_convolution_value_at_shift_matches_reference(self) -> None:
        """convolution_value_at_shift should agree with y_reference within tolerance."""
        result = _prepare_rect_rect(n=128)
        # Check a sample of frames
        frame_indices = [0, 10, 64, 127, result.num_frames - 1]
        for k in frame_indices:
            t_k = float(result.output_t[k])
            computed = ConvolutionAnalyzer.convolution_value_at_shift(result, t_k)
            reference = float(result.y_reference[k])
            assert abs(computed - reference) < 2e-4, (
                f"Frame {k}: computed={computed:.6g}, reference={reference:.6g}"
            )

    def test_all_frames_within_tolerance(self) -> None:
        """Every frame-by-frame computed value matches the reference."""
        result = _prepare_rect_rect(n=64)
        computed = np.array([
            ConvolutionAnalyzer.convolution_value_at_shift(result, float(t))
            for t in result.output_t
        ])
        np.testing.assert_allclose(computed, result.y_reference, atol=2e-4)


# ---------------------------------------------------------------------------
# 8. Preset loading
# ---------------------------------------------------------------------------

class TestPresets:
    @pytest.mark.parametrize("x_name", X_PRESET_NAMES)
    @pytest.mark.parametrize("h_name", H_PRESET_NAMES)
    def test_all_preset_combinations(self, x_name: str, h_name: str) -> None:
        result = ConvolutionAnalyzer.prepare_from_presets(x_name, h_name, num_samples=64)
        assert result.num_frames == 2 * 64 - 1
        assert np.all(np.isfinite(result.y_reference))

    def test_unknown_x_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown x preset"):
            ConvolutionAnalyzer.prepare_from_presets("NonExistentX", "Rectangular Pulse")

    def test_unknown_h_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown h preset"):
            ConvolutionAnalyzer.prepare_from_presets("Rectangular Pulse", "NonExistentH")


# ---------------------------------------------------------------------------
# 9. Edge case validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_x_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(
                np.array([]), np.array([]), tau, np.zeros(64)
            )

    def test_single_sample_raises(self) -> None:
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(
                np.array([0.0]), np.array([1.0]),
                np.linspace(-1, 1, 64), np.zeros(64)
            )

    def test_nonfinite_time_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        t_bad = tau.copy()
        t_bad[5] = np.nan
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(t_bad, np.zeros(64), tau, np.zeros(64))

    def test_nonfinite_values_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        x_bad = np.zeros(64)
        x_bad[10] = np.inf
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(tau, x_bad, tau, np.zeros(64))

    def test_mismatched_lengths_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(tau, np.zeros(32), tau, np.zeros(64))

    def test_too_few_samples_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        with pytest.raises(ValueError, match="num_samples"):
            ConvolutionAnalyzer.prepare_from_arrays(tau, np.zeros(64), tau, np.zeros(64), num_samples=2)

    def test_zero_time_range_raises(self) -> None:
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(
                np.array([0.5, 0.5]), np.array([1.0, 1.0]),
                np.linspace(-1, 1, 64), np.zeros(64)
            )

    def test_2d_array_raises(self) -> None:
        tau = np.linspace(-1, 1, 64)
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_from_arrays(
                tau.reshape(8, 8), np.zeros(64), tau, np.zeros(64)
            )


# ---------------------------------------------------------------------------
# 10. Overlap product
# ---------------------------------------------------------------------------

class TestOverlapProduct:
    def test_overlap_product_shape(self) -> None:
        result = _prepare_rect_rect(64)
        prod = ConvolutionAnalyzer.overlap_product(result, 0.0)
        assert prod.shape == result.tau.shape

    def test_overlap_product_integral_matches_reference(self) -> None:
        result = _prepare_rect_rect(256)
        k = result.num_frames // 2
        t_k = float(result.output_t[k])
        prod = ConvolutionAnalyzer.overlap_product(result, t_k)
        y_direct = float(np.sum(prod) * result.dt)
        y_ref = float(result.y_reference[k])
        assert abs(y_direct - y_ref) < 1e-3

    def test_overlap_zero_when_no_support(self) -> None:
        """When h(t-tau) is shifted completely outside x, overlap should be zero."""
        result = _prepare_rect_rect(128)
        prod = ConvolutionAnalyzer.overlap_product(result, 10.0)  # far out of range
        assert np.allclose(prod, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. Flexible Signal Inputs: Presets & Custom Signals (All 4 Combinations)
# ---------------------------------------------------------------------------

class TestConvolutionFlexibleInputs:
    """Test all four input combinations for ConvolutionAnalyzer.prepare_signals:
      Case 1: x=Preset, h=Preset
      Case 2: x=Custom, h=Preset
      Case 3: x=Preset, h=Custom
      Case 4: x=Custom, h=Custom
    """

    def test_case1_preset_preset(self) -> None:
        result = ConvolutionAnalyzer.prepare_signals(
            x_input="Rectangular Pulse",
            h_input="Triangle Pulse",
            num_samples=128,
        )
        assert isinstance(result, ConvolutionResult)
        assert result.num_samples == 128
        assert result.x_name == "Rectangular Pulse"
        assert result.h_name == "Triangle Pulse"
        assert len(result.y_reference) == 2 * 128 - 1
        assert np.all(np.isfinite(result.y_reference))

    def test_case2_custom_preset(self) -> None:
        t_custom = np.linspace(-1.0, 1.0, 128)
        x_custom = np.sin(np.pi * t_custom)

        result = ConvolutionAnalyzer.prepare_signals(
            x_input=(t_custom, x_custom),
            h_input="Exponential Decay",
            num_samples=128,
        )
        assert isinstance(result, ConvolutionResult)
        assert result.x_name == "Custom x(t)"
        assert result.h_name == "Exponential Decay"
        assert result.num_samples == 128
        assert np.all(np.isfinite(result.y_reference))

    def test_case3_preset_custom(self) -> None:
        t_custom = np.linspace(-1.0, 1.0, 128)
        h_custom = np.where(np.abs(t_custom) <= 0.3, 1.0, 0.0)

        result = ConvolutionAnalyzer.prepare_signals(
            x_input="Sine Burst",
            h_input=(t_custom, h_custom),
            num_samples=128,
        )
        assert isinstance(result, ConvolutionResult)
        assert result.x_name == "Sine Burst"
        assert result.h_name == "Custom h(t)"
        assert result.num_samples == 128
        assert np.all(np.isfinite(result.y_reference))

    def test_case4_custom_custom(self) -> None:
        t_x = np.linspace(-1.0, 1.0, 128)
        x_custom = np.exp(-t_x**2)
        t_h = np.linspace(-1.0, 1.0, 128)
        h_custom = np.sin(2 * np.pi * t_h)

        result = ConvolutionAnalyzer.prepare_signals(
            x_input=(t_x, x_custom),
            h_input=(t_h, h_custom),
            num_samples=128,
            x_name="My Custom x",
            h_name="My Custom h",
        )
        assert isinstance(result, ConvolutionResult)
        assert result.x_name == "My Custom x"
        assert result.h_name == "My Custom h"
        assert result.num_samples == 128
        assert np.all(np.isfinite(result.y_reference))

    def test_signal_independence_state(self) -> None:
        """Modifying x array after passing it does not alter independent result arrays."""
        t = np.linspace(-1.0, 1.0, 64)
        x = np.ones(64)
        h = np.ones(64) * 0.5

        result = ConvolutionAnalyzer.prepare_signals(
            x_input=(t, x),
            h_input=(t, h),
            num_samples=64,
        )
        # Modify original x
        x[:] = 999.0
        # Result arrays should remain uncorrupted
        assert not np.allclose(result.x_resampled, 999.0)
        assert np.allclose(result.h_resampled, 0.5)

    def test_invalid_input_types_raise(self) -> None:
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_signals(
                x_input=12345,  # type: ignore
                h_input="Rectangular Pulse",
            )
        with pytest.raises(ValueError):
            ConvolutionAnalyzer.prepare_signals(
                x_input="Rectangular Pulse",
                h_input={"invalid": "dict"},  # type: ignore
            )
