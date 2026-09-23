"""Comprehensive integration, regression, and mathematical validation tests."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import fourier
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "io_signal_io", Path(__file__).resolve().parents[1] / "io" / "signal_io.py"
)
io_signal_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io_signal_io)
import metrics
import signal
from fourier.curve2d_analysis import (
    CurveFourierResult,
    analyze_curve,
    calculate_curve_error,
    reconstruct_curve,
)
from gui.main_window import MainWindow
from signal.curve2d import Curve2D
from signal_io import load_curve, load_signal, save_curve, save_signal


@pytest.fixture(scope="module")
def application():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.quit()


def test_compatibility_re_exports() -> None:
    """Verify all expected 1D and 2D symbols are accessible from package roots."""
    assert hasattr(io_signal_io, "save_signal")
    assert hasattr(io_signal_io, "load_signal")
    assert hasattr(io_signal_io, "save_curve")
    assert hasattr(io_signal_io, "load_curve")
    assert hasattr(io_signal_io, "export_curve_csv")
    assert hasattr(io_signal_io, "export_curve_reconstruction")
    assert hasattr(io_signal_io, "export_curve_coefficients")
    assert hasattr(io_signal_io, "LoadedCurve")

    assert hasattr(signal, "Curve2D")
    assert hasattr(fourier, "CurveFourierResult")
    assert hasattr(fourier, "analyze_curve")
    assert hasattr(fourier, "reconstruct_curve")
    assert hasattr(metrics, "GibbsResult")
    assert hasattr(metrics, "analyze_gibbs")


def test_cross_mode_file_loading_helpful_errors(tmp_path) -> None:
    """Verify helpful messages when loading 1D file as 2D curve or vice versa."""
    signal_path = tmp_path / "test_signal.json"
    curve_path = tmp_path / "test_curve.json"

    t = np.linspace(0.0, 1.0, 101)
    sig = np.sin(2.0 * np.pi * t)
    save_signal(signal_path, t, sig)

    curve = Curve2D.from_points([[0, 0], [1, 0], [1, 1], [0, 1]])
    save_curve(curve_path, curve)

    # Attempt to load 1D signal with 2D curve loader:
    with pytest.raises(ValueError) as curve_err:
        load_curve(signal_path)
    assert "This file contains a 1D signal. Please use File -> Load Signal instead." in str(curve_err.value)

    # Attempt to load 2D curve with 1D signal loader:
    with pytest.raises(ValueError) as signal_err:
        load_signal(curve_path)
    assert "This file contains a 2D closed curve. Please use File -> Load 2D Curve instead." in str(signal_err.value)


def test_1d_harmonic_slider_uses_cached_analysis(application, monkeypatch) -> None:
    """Verify moving the 1D harmonic slider reuses cached coefficients and convergence."""
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.generate_signal()
    window.controls.harmonic_slider.setValue(2)

    # First reconstruction computes analysis and caches synthesizer
    window.reconstruct_signal()
    assert window.analysis_result is not None
    assert window.synthesizer is not None
    assert window.convergence_result is not None
    first_analysis = window.analysis_result
    first_convergence = window.convergence_result

    # Monkeypatch FourierAnalyzer.analyze to detect any unexpected re-run
    analyze_called = False
    original_analyze = fourier.analysis.FourierAnalyzer.analyze

    def mock_analyze(self):
        nonlocal analyze_called
        analyze_called = True
        return original_analyze(self)

    monkeypatch.setattr(fourier.analysis.FourierAnalyzer, "analyze", mock_analyze)

    # Now move harmonic slider:
    window.controls.harmonic_slider.setValue(10)

    # The slider move must use the cached synthesizer without recalculating analysis
    assert analyze_called is False
    assert window.analysis_result is first_analysis
    assert window.convergence_result is first_convergence
    assert "10" in window.status_label.text()
    window.close()


def test_mode_transitions_and_reset_disable_unrelated_controls(application) -> None:
    """Verify switching between 1D and 2D cleanly manages button enablement."""
    window = MainWindow()

    # Initially in fresh state:
    assert window.controls.reconstruct_button.isEnabled() is False
    assert window.controls.analyze_curve_button.isEnabled() is False

    # 1. Generate 1D signal
    window.generate_signal()
    assert window.controls.reconstruct_button.isEnabled() is True
    assert window.controls.analyze_curve_button.isEnabled() is False

    # 2. Transition to 2D Curve drawing
    window.start_curve_drawing()
    assert window.t is None
    assert window.original_signal is None
    assert window.controls.reconstruct_button.isEnabled() is False
    assert window.controls.finish_curve_button.isEnabled() is True

    # 3. Finish 2D curve
    candidate = Curve2D.from_points([[0, 0], [1, 0], [1, 1], [0, 1]])
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()

    assert window.controls.analyze_curve_button.isEnabled() is True
    # Crucial: 1D reconstruct must remain disabled in 2D mode!
    assert window.controls.reconstruct_button.isEnabled() is False

    # 4. Analyze 2D curve
    window.analyze_2d_curve()
    assert window.controls.curve_harmonic_slider.isEnabled() is True
    assert window.controls.curve_epicycle_button.isEnabled() is True
    assert window.controls.curve_spectrum_button.isEnabled() is True
    assert window.controls.curve_error_button.isEnabled() is True
    assert window.controls.reconstruct_button.isEnabled() is False

    # 5. Clear / Reset
    window.reset()
    assert window.current_curve is None
    assert window.curve_analysis is None
    assert window.controls.reconstruct_button.isEnabled() is False
    assert window.controls.analyze_curve_button.isEnabled() is False
    assert window.controls.curve_harmonic_slider.isEnabled() is False
    assert window.controls.curve_epicycle_button.isEnabled() is False
    assert window.controls.curve_spectrum_button.isEnabled() is False
    assert window.controls.curve_error_button.isEnabled() is False
    window.close()


def test_2d_reconstruction_at_n0_n1_and_moderate() -> None:
    """Verify mathematical properties of 2D reconstruction at N=0, N=1, and moderate N."""
    parameter = np.arange(256, dtype=float) / 256.0
    circle = np.column_stack((np.cos(2 * np.pi * parameter) + 2.0, np.sin(2 * np.pi * parameter) - 1.0))
    coefficients = analyze_curve(circle, num_samples=256, maximum_harmonic=16)

    # N = 0: DC component only
    _, x0, y0 = reconstruct_curve(coefficients, harmonic_count=0)
    assert np.allclose(x0, 2.0, atol=1e-3)
    assert np.allclose(y0, -1.0, atol=1e-3)

    # N = 1: Should capture full unit circle with centroid
    recon_n1 = calculate_curve_error(coefficients, harmonic_count=1)
    assert recon_n1.rmse < 1e-3
    assert recon_n1.maximum_error < 2e-3

    # Moderate N = 8: Should maintain high precision
    recon_n8 = calculate_curve_error(coefficients, harmonic_count=8)
    assert recon_n8.rmse < 1e-3


def test_synthetic_circle_and_ellipse_validation() -> None:
    """Verify synthetic shapes produce mathematically expected Fourier coefficients."""
    param = np.linspace(0.0, 1.0, 512, endpoint=False)

    # 1. Circle: x(t) = cos(2*pi*t), y(t) = sin(2*pi*t)
    circle_pts = np.column_stack((np.cos(2 * np.pi * param), np.sin(2 * np.pi * param)))
    res_circle = analyze_curve(circle_pts, num_samples=512, maximum_harmonic=4)

    dc_idx = np.flatnonzero(res_circle.harmonics == 0)[0]
    pos1_idx = np.flatnonzero(res_circle.harmonics == 1)[0]
    neg1_idx = np.flatnonzero(res_circle.harmonics == -1)[0]

    assert abs(res_circle.x_coefficients[dc_idx]) < 1e-10
    assert abs(res_circle.y_coefficients[dc_idx]) < 1e-10
    # X1 = 0.5, X-1 = 0.5, Y1 = -0.5j, Y-1 = 0.5j
    assert res_circle.x_coefficients[pos1_idx] == pytest.approx(0.5, abs=1e-3)
    assert res_circle.x_coefficients[neg1_idx] == pytest.approx(0.5, abs=1e-3)
    assert res_circle.y_coefficients[pos1_idx] == pytest.approx(-0.5j, abs=1e-3)
    assert res_circle.y_coefficients[neg1_idx] == pytest.approx(0.5j, abs=1e-3)

    # 2. Ellipse: x(t) = 2*cos(2*pi*t), y(t) = sin(2*pi*t)
    # Under arc-length parameterization, the fundamental is dominant and reconstruction converges
    ellipse_pts = np.column_stack((2.0 * np.cos(2 * np.pi * param), np.sin(2 * np.pi * param)))
    res_ellipse = analyze_curve(ellipse_pts, num_samples=512, maximum_harmonic=8)
    pos1_ellipse = np.flatnonzero(res_ellipse.harmonics == 1)[0]
    assert abs(res_ellipse.x_coefficients[pos1_ellipse]) > 0.9  # Dominant X fundamental
    assert abs(res_ellipse.y_coefficients[pos1_ellipse]) == pytest.approx(0.5, abs=0.05)
    recon_ellipse = calculate_curve_error(res_ellipse, harmonic_count=8)
    assert recon_ellipse.rmse < 0.01


def test_main_window_close_event_cleans_up_windows(application) -> None:
    """Verify closing MainWindow stops all timers and closes subwindows."""
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.generate_signal()
    window.show_epicycles()
    assert window.epicycle_window is not None

    # Simulate window close event
    window.close()
    assert window.epicycle_window is None
    assert window.curve2d_epicycle_window is None
    assert window.curve2d_spectrum_window is None
    assert window.curve2d_error_window is None


def test_phase_zero_for_near_zero_magnitude() -> None:
    """Verify phase calculations return 0.0 for zero-magnitude coefficients."""
    harmonics = np.array([-1, 0, 1])
    x_coeffs = np.array([0.0, 1.0, 0.0], dtype=complex)
    y_coeffs = np.array([0.0, 0.0, 0.0], dtype=complex)
    result = CurveFourierResult(
        parameter=np.linspace(0, 1, 10, endpoint=False),
        x_samples=np.zeros(10),
        y_samples=np.zeros(10),
        harmonics=harmonics,
        x_coefficients=x_coeffs,
        y_coefficients=y_coeffs,
    )
    assert result.x_phases[0] == 0.0
    assert result.y_phases[0] == 0.0
    assert result.y_phases[1] == 0.0
