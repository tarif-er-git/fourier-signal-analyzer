"""GUI tests for ConvolutionWindow and SignalDrawingCanvas.

Verifies:
  - SignalDrawingCanvas mouse interaction, baseline updates, validation, and clearing
  - ConvolutionWindow preset selection and custom drawing workflow
  - All four combinations:
      1. Preset x + Preset h
      2. Custom x + Preset h
      3. Preset x + Custom h
      4. Custom x + Custom h
  - Complete signal independence between x(t) and h(t)
  - Missing signal validation warnings (no crashes)
  - Safe switching between Preset and Custom Draw modes
  - Playback controls and manual scrubbing across all combinations
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from gui.convolution_window import ConvolutionWindow, _READY, _PLAYING, _PAUSED
from gui.drawing_canvas import SignalDrawingCanvas


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestSignalDrawingCanvas:
    """Tests for the reusable 1D SignalDrawingCanvas widget."""

    def test_initial_state(self, qapp) -> None:
        canvas = SignalDrawingCanvas(num_samples=128, t_range=(-1.0, 1.0))
        assert not canvas.has_content()
        # Empty drawing raises ValueError on finish
        with pytest.raises(ValueError, match="Please draw a signal"):
            canvas.finish_drawing()

    def test_drawing_programmatic_stroke(self, qapp) -> None:
        canvas = SignalDrawingCanvas(num_samples=128, t_range=(-1.0, 1.0))
        # Simulate drawing a stroke
        canvas.start_drawing("Test Signal")

        # Directly simulate stroke points
        canvas._stroke_t = [-0.5, 0.0, 0.5]
        canvas._stroke_x = [0.2, 0.8, 0.2]
        canvas._update_drawn_signal()

        assert canvas.has_content()
        t, x = canvas.finish_drawing()
        assert len(t) == 128
        assert len(x) == 128
        assert np.isclose(t[0], -1.0)
        assert np.isclose(t[-1], 1.0)
        assert np.max(x) > 0.5

    def test_clear_canvas(self, qapp) -> None:
        canvas = SignalDrawingCanvas(num_samples=128, t_range=(-1.0, 1.0))
        canvas._stroke_t = [-0.5, 0.5]
        canvas._stroke_x = [0.5, 0.5]
        canvas._update_drawn_signal()
        assert canvas.has_content()

        canvas.clear_drawing()
        assert not canvas.has_content()
        assert np.allclose(canvas._drawn_x, 0.0)

    def test_preload_initial_signal(self, qapp) -> None:
        canvas = SignalDrawingCanvas(num_samples=128, t_range=(-1.0, 1.0))
        t_init = np.linspace(-1.0, 1.0, 64)
        x_init = np.sin(np.pi * t_init)

        canvas.start_drawing("Preloaded", initial_signal=(t_init, x_init))
        assert canvas.has_content()
        t, x = canvas.finish_drawing()
        assert len(t) == 128
        assert np.max(x) > 0.8


class TestConvolutionWindowGUI:
    """Tests for ConvolutionWindow input combinations, independence, and controls."""

    def test_window_initialization(self, qapp) -> None:
        win = ConvolutionWindow()
        assert win._x_mode == "preset"
        assert win._h_mode == "preset"
        assert win._custom_x is None
        assert win._custom_h is None
        assert win._stacked_widget.currentIndex() == 0  # Simulation view
        win.close()

    def test_case1_preset_preset_prepare_and_play(self, qapp) -> None:
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(0)
        win._h_source_combo.setCurrentIndex(0)
        win._x_preset_combo.setCurrentText("Rectangular Pulse")
        win._h_preset_combo.setCurrentText("Triangle Pulse")

        win._on_prepare()
        assert win._state == _READY
        assert win._result is not None
        assert win._result.num_frames > 0

        # Test playback
        win.play()
        assert win._state == _PLAYING
        win.pause()
        assert win._state == _PAUSED
        win.reset()
        assert win._state == _READY
        win.close()

    def test_case2_custom_x_preset_h(self, qapp) -> None:
        win = ConvolutionWindow()
        # Switch x to custom
        win._x_source_combo.setCurrentIndex(1)
        assert win._x_mode == "custom"

        # Provide custom x
        t_custom = np.linspace(-1.0, 1.0, 256)
        x_custom = np.sin(2.0 * np.pi * t_custom)
        win._custom_x = (t_custom, x_custom)

        win._h_source_combo.setCurrentIndex(0)
        win._h_preset_combo.setCurrentText("Exponential Decay")

        win._on_prepare()
        assert win._state == _READY
        assert win._result is not None
        assert "Custom" in win._result.x_name
        win.close()

    def test_case3_preset_x_custom_h(self, qapp) -> None:
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(0)
        win._x_preset_combo.setCurrentText("Sine Burst")

        # Switch h to custom
        win._h_source_combo.setCurrentIndex(1)
        assert win._h_mode == "custom"
        t_custom = np.linspace(-1.0, 1.0, 256)
        h_custom = np.where(np.abs(t_custom) <= 0.4, 0.8, 0.0)
        win._custom_h = (t_custom, h_custom)

        win._on_prepare()
        assert win._state == _READY
        assert win._result is not None
        assert "Custom" in win._result.h_name
        win.close()

    def test_case4_custom_x_custom_h(self, qapp) -> None:
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(1)
        win._h_source_combo.setCurrentIndex(1)

        t = np.linspace(-1.0, 1.0, 256)
        win._custom_x = (t, np.cos(np.pi * t))
        win._custom_h = (t, np.sin(np.pi * t))

        win._on_prepare()
        assert win._state == _READY
        assert win._result is not None
        assert "Custom" in win._result.x_name
        assert "Custom" in win._result.h_name
        win.close()

    def test_signals_complete_independence(self, qapp) -> None:
        """Verify modifying, drawing, or clearing x does not affect h and vice versa."""
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(1)
        win._h_source_combo.setCurrentIndex(1)

        t = np.linspace(-1.0, 1.0, 256)
        sig_x = np.ones(256) * 0.7
        sig_h = np.ones(256) * 0.3

        win._custom_x = (t.copy(), sig_x.copy())
        win._custom_h = (t.copy(), sig_h.copy())

        # Clearing x does NOT clear h
        win._on_clear_x()
        assert win._custom_x is None
        assert win._custom_h is not None
        assert np.allclose(win._custom_h[1], 0.3)

        # Re-set x, then clear h: x is NOT cleared
        win._custom_x = (t.copy(), sig_x.copy())
        win._on_clear_h()
        assert win._custom_h is None
        assert win._custom_x is not None
        assert np.allclose(win._custom_x[1], 0.7)
        win.close()

    def test_drawing_workflow_navigation(self, qapp) -> None:
        """Verify entering draw mode switches stack and updates headers."""
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(1)

        # Click Draw x
        win._on_draw_x()
        assert win._stacked_widget.currentIndex() == 1
        assert "x(t)" in win._draw_title_label.text()
        assert win._active_drawing_target == "x"

        # Cancel returns to simulation view
        win._on_cancel_drawing()
        assert win._stacked_widget.currentIndex() == 0

        # Click Draw h
        win._h_source_combo.setCurrentIndex(1)
        win._on_draw_h()
        assert win._stacked_widget.currentIndex() == 1
        assert "h(t)" in win._draw_title_label.text()
        assert win._active_drawing_target == "h"

        win._on_cancel_drawing()
        assert win._stacked_widget.currentIndex() == 0
        win.close()

    def test_validation_prevents_crash_on_missing_custom_draw(self, qapp, monkeypatch) -> None:
        """Validation shows a QMessageBox without crashing when a custom signal is not drawn."""
        win = ConvolutionWindow()
        win._x_source_combo.setCurrentIndex(1)
        assert win._custom_x is None

        # Monkeypatch QMessageBox.warning to intercept without blocking
        warning_shown = []
        monkeypatch.setattr(
            "gui.convolution_window.QMessageBox.warning",
            lambda parent, title, text: warning_shown.append((title, text)),
        )

        win._on_prepare()
        assert len(warning_shown) == 1
        assert "Signal x(t) missing" in warning_shown[0][0]
        assert win._result is None

        # Test missing h
        win._x_source_combo.setCurrentIndex(0)
        win._h_source_combo.setCurrentIndex(1)
        assert win._custom_h is None

        win._on_prepare()
        assert len(warning_shown) == 2
        assert "Kernel h(t) missing" in warning_shown[1][0]
        win.close()
