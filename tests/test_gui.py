"""Headless smoke tests for the Step 9 PySide6 GUI workflow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.main_window import MainWindow
from signal.curve2d import Curve2D


@pytest.fixture(scope="module")
def application():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.quit()


def _mouse_event(x: float, y: float) -> QMouseEvent:
    """Create a left-button event at a canvas-local widget position."""
    pos = QPointF(x, y)
    return QMouseEvent(
        QMouseEvent.Type.MouseMove,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_generate_sine_and_reconstruct_with_one_harmonic(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.controls.harmonic_slider.setValue(1)

    window.generate_signal()
    assert window.original_signal is not None
    assert window.original_signal.shape == (1001,)

    window.reconstruct_signal()
    assert window.reconstructed_signal is not None
    assert window.error is not None
    assert window.controls.metric_labels["RMSE"].text() != "--"
    window.close()


def test_custom_signal_becomes_active_reconstruction_input(application) -> None:
    window = MainWindow()
    window.start_custom_drawing()
    window.canvas._drawn_t = [0.0, 1.0]
    window.canvas._drawn_x = [0.75, 0.75]

    window.finish_custom_drawing()
    window.reconstruct_signal()

    assert window.active_signal_type == "custom"
    assert window.controls.current_signal_label.text() == "Current signal: Custom"
    assert window.active_signal_data is not None
    assert window.current_signal_data is not None
    np.testing.assert_allclose(window.original_signal, 0.75)
    assert window.analysis_result is not None
    assert window.analysis_result.a0 == pytest.approx(1.5)
    np.testing.assert_allclose(window.reconstructed_signal, 0.75, atol=1e-10)
    window.close()


def test_custom_drawing_starts_with_full_zero_baseline(application) -> None:
    window = MainWindow()

    window.start_custom_drawing()

    np.testing.assert_allclose(window.canvas._drawn_t, np.linspace(0.0, 1.0, 1001))
    np.testing.assert_allclose(window.canvas._drawn_x, 0.0)
    window.close()


def test_custom_drawing_accumulates_multiple_strokes(application) -> None:
    window = MainWindow()
    window.start_custom_drawing()

    window.canvas._stroke_t = [0.1, 0.3]
    window.canvas._stroke_x = [1.0, 1.0]
    window.canvas._update_drawn_signal()
    first_stroke = np.asarray(window.canvas._drawn_x, dtype=float).copy()

    window.canvas._stroke_t = [0.7, 0.9]
    window.canvas._stroke_x = [-1.0, -1.0]
    window.canvas._update_drawn_signal()
    second_stroke = np.asarray(window.canvas._drawn_x, dtype=float)

    assert np.max(first_stroke) == pytest.approx(1.0)
    assert np.min(second_stroke) == pytest.approx(-1.0)
    assert second_stroke[200] == pytest.approx(first_stroke[200])
    window.close()


def test_2d_curve_mode_finalizes_and_clears_curve(application) -> None:
    window = MainWindow()

    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()

    assert window.current_curve is candidate
    assert candidate.is_valid
    assert window.controls.finish_curve_button.isEnabled() is False

    window.clear_curve()

    assert window.current_curve is None
    assert window.controls.finish_curve_button.isEnabled() is True
    window.close()


def test_2d_curve_can_be_analyzed_from_control_panel(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    window.controls.analyze_curve_button.click()

    assert window.curve_analysis is not None
    assert "2D Curve Analysis" in window.controls.curve_analysis_label.text()
    assert "DC X:" in window.controls.curve_analysis_label.text()
    window.close()


def test_2d_harmonic_slider_updates_overlay_metrics(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    window.controls.analyze_curve_button.click()

    assert window.controls.curve_harmonic_slider.isEnabled()
    window.controls.curve_harmonic_slider.setValue(3)

    assert window.curve_reconstruction is not None
    assert window.curve_reconstruction.harmonic_count == 3
    assert "Harmonics Used: 3" in window.controls.curve_stats_label.text()
    window.close()


def test_2d_epicycle_view_uses_analyzed_curve(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    window.controls.analyze_curve_button.click()
    window.controls.curve_epicycle_button.click()

    assert window.curve2d_epicycle_window is not None
    window.curve2d_epicycle_window.pause()
    window.curve2d_epicycle_window.reset()
    window.curve2d_epicycle_window.close()
    window.curve2d_epicycle_window = None
    window.close()


def test_2d_harmonic_spectrum_uses_analyzed_curve(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    window.controls.analyze_curve_button.click()
    window.controls.curve_spectrum_button.click()

    assert window.curve2d_spectrum_window is not None
    assert window.curve2d_spectrum_window.table.rowCount() > 0
    window.curve2d_spectrum_window.close()
    window.curve2d_spectrum_window = None
    window.close()


def test_2d_error_convergence_view_uses_cached_analysis(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    window.controls.analyze_curve_button.click()
    window.controls.curve_error_button.click()

    assert window.curve2d_error_window is not None
    cached_convergence = window.curve_convergence
    window.controls.curve_harmonic_slider.setValue(4)
    assert window.curve_reconstruction.harmonic_count == 4
    assert window.curve_convergence is cached_convergence
    window.curve2d_error_window.close()
    window.curve2d_error_window = None
    window.close()


def test_2d_curve_save_load_and_exports(application, monkeypatch, tmp_path) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    candidate = Curve2D.from_points(
        [[-0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, -0.5]]
    )
    window.canvas._curve = candidate
    window._curve_candidate_ready(candidate)
    window.finish_curve_drawing()
    curve_path = tmp_path / "curve.json"
    curve_csv = tmp_path / "curve.csv"
    reconstruction_csv = tmp_path / "reconstruction.csv"
    coefficients_csv = tmp_path / "coefficients.csv"
    save_paths = iter(
        [str(curve_path), str(curve_csv), str(reconstruction_csv), str(coefficients_csv)]
    )
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (next(save_paths), "file"),
    )
    window.save_curve_file()
    window.export_curve_csv_file()
    window.controls.analyze_curve_button.click()
    window.controls.curve_harmonic_slider.setValue(3)
    window.export_curve_reconstruction_file()
    window.export_curve_coefficients_file()
    assert all(path.exists() for path in (curve_path, curve_csv, reconstruction_csv, coefficients_csv))

    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(curve_path), "JSON"),
    )
    window.load_curve_file()
    assert window.current_curve is not None
    assert window.curve_analysis is None
    assert window.controls.analyze_curve_button.isEnabled()
    assert not window.controls.curve_harmonic_slider.isEnabled()
    window.close()


def test_2d_mouse_coordinates_follow_rendered_axis_bounds(application) -> None:
    window = MainWindow()
    window.start_curve_drawing()
    canvas = window.canvas
    canvas.draw()
    axis = canvas.axes[0]

    left_bottom = axis.transData.transform((-1.0, -1.0))
    widget_point = (
        left_bottom[0] / canvas.devicePixelRatioF(),
        (canvas.figure.bbox.height - left_bottom[1]) / canvas.devicePixelRatioF(),
    )
    assert canvas._event_to_curve_point(
        _mouse_event(widget_point[0], widget_point[1])
    ) == pytest.approx((-1.0, -1.0))

    outside = axis.bbox.x1 + 20.0
    outside_widget_x = outside / canvas.devicePixelRatioF()
    assert canvas._event_to_curve_point(
        _mouse_event(outside_widget_x, widget_point[1])
    ) is None
    window.close()


def test_nonconstant_custom_signal_is_not_replaced_by_sine(application) -> None:
    window = MainWindow()
    window.start_custom_drawing()
    window.canvas._drawn_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    window.canvas._drawn_x = [0.0, 1.0, 0.0, -1.0, 0.0]

    window.finish_custom_drawing()
    custom_signal = window.original_signal.copy()
    window.reconstruct_signal()

    assert window.active_signal_type == "custom"
    assert not np.allclose(custom_signal, np.sin(2.0 * np.pi * window.t))
    assert not np.allclose(window.reconstructed_signal, np.sin(2.0 * np.pi * window.t))
    window.close()


def test_custom_finish_and_reconstruct_buttons_use_custom_state(application) -> None:
    window = MainWindow()
    window.controls.draw_button.click()
    window.canvas._drawn_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    window.canvas._drawn_x = [0.0, 1.0, 0.0, -1.0, 0.0]

    window.controls.finish_drawing_button.click()
    window.controls.reconstruct_button.click()

    assert window.active_signal_type == "custom"
    assert window.controls.current_signal_label.text() == "Current signal: Custom"
    assert window.reconstructed_signal is not None
    assert not np.allclose(window.reconstructed_signal, np.sin(2.0 * np.pi * window.t))
    window.close()


def test_generate_square_and_reconstruct_with_ten_harmonics(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(1)
    window.controls.harmonic_slider.setValue(10)

    window.generate_signal()
    window.reconstruct_signal()

    assert window.analysis_result is not None
    assert window.analysis_result.num_harmonics == 50
    assert window.reconstructed_signal is not None
    assert np.all(np.isfinite(window.reconstructed_signal))
    assert window.controls.metric_labels["MSE"].text() != "--"
    assert window.convergence_result is not None
    assert window.gibbs_result is not None
    assert window.controls.gibbs_labels["Overshoot"].text() != "--"
    window.close()


def test_harmonic_slider_live_updates_active_reconstruction(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(1)
    window.generate_signal()
    window.controls.harmonic_slider.setValue(1)
    window.reconstruct_signal()
    first_reconstruction = window.reconstructed_signal.copy()

    window.controls.harmonic_slider.setValue(10)

    assert window.reconstructed_signal is not None
    assert not np.allclose(window.reconstructed_signal, first_reconstruction)
    assert "10" in window.status_label.text()
    window.close()


def test_smooth_signal_reports_no_gibbs_discontinuity(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.generate_signal()
    window.reconstruct_signal()

    assert window.gibbs_result is not None
    assert window.gibbs_result.detected is False
    assert "No significant discontinuity" in window.controls.gibbs_status.text()
    assert window.controls.gibbs_labels["Overshoot"].text() == "--"
    window.close()


def test_fft_comparison_updates_spectrum_and_timing(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.generate_signal()

    window.compare_with_fft()

    assert window.fft_comparison is not None
    assert window.fft_comparison.harmonic_numbers[0] == 1
    assert window.controls.fft_timing_labels["Fourier"].text().endswith("ms")
    assert window.controls.fft_timing_labels["FFT"].text().endswith("ms")
    window.close()


def test_epicycle_window_opens_and_can_be_replaced(application) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(0)
    window.generate_signal()
    window.controls.harmonic_slider.setValue(3)

    window.show_epicycles()

    assert window.epicycle_window is not None
    assert window.epicycle_window.harmonic_slider.value() == 3
    window.epicycle_window.harmonic_slider.setValue(5)
    window.epicycle_window.pause()
    window.epicycle_window.reset()
    window.show_epicycles()
    assert window.epicycle_window is not None
    window.reset()
    assert window.epicycle_window is None
    window.close()


def test_save_load_and_exports_use_current_application_state(
    application, monkeypatch, tmp_path
) -> None:
    window = MainWindow()
    window.controls.signal_selector.setCurrentIndex(1)
    window.generate_signal()
    window.reconstruct_signal()
    signal_path = tmp_path / "signal.json"
    reconstruction_path = tmp_path / "reconstruction.csv"
    spectrum_path = tmp_path / "spectrum.csv"
    error_path = tmp_path / "error.csv"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(signal_path), "JSON"),
    )
    window.save_signal_file()
    assert signal_path.exists()

    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(signal_path), "JSON"),
    )
    window.load_signal_file()
    assert window.original_signal is not None
    assert window.analysis_result is None
    assert window.reconstructed_signal is None

    destinations = iter(
        [str(reconstruction_path), str(spectrum_path), str(error_path)]
    )
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (next(destinations), "CSV"),
    )
    window.reconstruct_signal()
    window.export_reconstruction_file()
    window.export_spectrum_file()
    window.export_error_file()
    assert reconstruction_path.exists()
    assert spectrum_path.exists()
    assert error_path.exists()

    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(report_path), "JSON"),
    )
    window.export_report_file()
    assert report_path.exists()
    window.close()


def test_reconstruct_before_generate_shows_user_message(application, monkeypatch) -> None:
    window = MainWindow()
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: messages.append(message),
    )

    window.reconstruct_signal()

    assert messages == [
        "Generate or finish drawing a signal before reconstructing it."
    ]
    window.close()
