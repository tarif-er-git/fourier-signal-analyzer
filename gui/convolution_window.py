"""Self-contained PySide6 window for the convolution simulation animation.

Architecture
------------
This window follows the same pattern as Curve2DEpicycleWindow:
  - A QMainWindow with an embedded FigureCanvasQTAgg
  - A QTimer drives per-frame updates (not matplotlib.animation.FuncAnimation)
  - All mathematical logic is delegated to signal.convolution.ConvolutionAnalyzer
  - All drawing logic is delegated to visualization.convolution_plot

The window is self-contained: users pick x(t) and h(t) from built-in presets
inside the window, without needing a signal active in the main window.

Animation State Machine
-----------------------
IDLE    -> no signals prepared yet
READY   -> signals prepared, at frame 0
PLAYING -> timer running, frames advancing
PAUSED  -> timer stopped, frame held
DONE    -> reached last frame, full output shown
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from signal.convolution import (
    ConvolutionAnalyzer,
    ConvolutionResult,
    X_PRESET_NAMES,
    H_PRESET_NAMES,
)
from visualization.convolution_plot import (
    setup_convolution_figure,
    update_convolution_frame,
    draw_final_result,
)

# Animation state constants
_IDLE = "idle"
_READY = "ready"
_PLAYING = "playing"
_PAUSED = "paused"
_DONE = "done"

# Default QTimer interval in ms (≈ 33 fps)
_DEFAULT_FRAME_INTERVAL_MS = 30
# Number of samples on the common tau grid
_NUM_SAMPLES = 256


class ConvolutionWindow(QMainWindow):
    """Interactive window that visualizes the convolution process step by step.

    The animation demonstrates:
      1. x(τ)    — fixed signal (top panel, never changes)
      2. h(t−τ) — h reversed then shifted right by t (middle panel)
      3. y(t)   — overlap integral progressively built (bottom panel)

    Controls
    --------
    Play / Pause / Reset — animation playback
    Speed slider — adjusts QTimer interval (slow ↔ fast)
    Shift slider — manually scrub through frames (updates middle+bottom panels)
    Prepare Convolution — (re)generates signals and resets to frame 0
    """

    FRAME_INTERVAL_MS = _DEFAULT_FRAME_INTERVAL_MS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convolution Simulation")
        self.resize(1000, 760)

        # Core state
        self._state: str = _IDLE
        self._result: ConvolutionResult | None = None
        self._frame_index: int = 0
        self._output_t_so_far: np.ndarray = np.array([], dtype=float)
        self._y_so_far: np.ndarray = np.array([], dtype=float)
        self._handles: dict[str, object] = {}

        # QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)

        self._build_layout()
        self._set_state(_IDLE)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Construct the window layout: controls on the left, plots on the right."""
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # --- Left control panel ---
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(260)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)

        # Signal selection group
        signal_group = QGroupBox("Signal Selection")
        sg_layout = QFormLayout(signal_group)

        self._x_selector = QComboBox()
        for name in X_PRESET_NAMES:
            self._x_selector.addItem(name)
        sg_layout.addRow("Signal  x(t):", self._x_selector)

        self._h_selector = QComboBox()
        for name in H_PRESET_NAMES:
            self._h_selector.addItem(name)
        sg_layout.addRow("Kernel  h(t):", self._h_selector)

        ctrl_layout.addWidget(signal_group)

        # Prepare button
        self._prepare_button = QPushButton("Prepare Convolution")
        self._prepare_button.setToolTip(
            "Resample both signals onto a common grid\n"
            "and compute the reference convolution."
        )
        self._prepare_button.clicked.connect(self._on_prepare)
        ctrl_layout.addWidget(self._prepare_button)

        # Playback group
        pb_group = QGroupBox("Playback")
        pb_layout = QVBoxLayout(pb_group)

        btn_row = QHBoxLayout()
        self._play_button = QPushButton("Play")
        self._pause_button = QPushButton("Pause")
        self._reset_button = QPushButton("Reset")
        self._play_button.clicked.connect(self.play)
        self._pause_button.clicked.connect(self.pause)
        self._reset_button.clicked.connect(self.reset)
        btn_row.addWidget(self._play_button)
        btn_row.addWidget(self._pause_button)
        btn_row.addWidget(self._reset_button)
        pb_layout.addLayout(btn_row)

        # Speed slider
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 10)
        self._speed_slider.setValue(5)
        self._speed_slider.setToolTip("Animation speed (1=slow, 10=fast)")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        self._speed_label = QLabel("5")
        self._speed_label.setMinimumWidth(20)
        speed_row.addWidget(self._speed_slider, 1)
        speed_row.addWidget(self._speed_label)
        pb_layout.addLayout(speed_row)

        ctrl_layout.addWidget(pb_group)

        # Manual shift group
        shift_group = QGroupBox("Manual Shift  t")
        sh_layout = QVBoxLayout(shift_group)
        self._shift_slider = QSlider(Qt.Orientation.Horizontal)
        self._shift_slider.setRange(0, 100)
        self._shift_slider.setValue(0)
        self._shift_slider.setToolTip(
            "Drag to manually position h(t−τ) for a specific shift value t."
        )
        self._shift_slider.valueChanged.connect(self._on_shift_slider_changed)
        sh_layout.addWidget(self._shift_slider)
        self._shift_value_label = QLabel("t = --")
        self._shift_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sh_layout.addWidget(self._shift_value_label)
        ctrl_layout.addWidget(shift_group)

        # Info / status group
        info_group = QGroupBox("Current Frame")
        info_layout = QVBoxLayout(info_group)
        self._frame_label = QLabel("Frame: --")
        self._conv_value_label = QLabel("y(t) = --")
        self._conv_value_label.setWordWrap(True)
        info_layout.addWidget(self._frame_label)
        info_layout.addWidget(self._conv_value_label)
        ctrl_layout.addWidget(info_group)

        # Educational note
        note_group = QGroupBox("Convention")
        note_layout = QVBoxLayout(note_group)
        note_text = QLabel(
            "x(τ) is fixed.\n\n"
            "h(τ) is reversed → h(−τ)\n"
            "then shifted → h(t−τ)\n\n"
            "At each shift t:\n"
            "  y(t) ≈ Σ x(τ)·h(t−τ)·Δτ\n\n"
            "Green region = overlap\n"
            "integrated → one y(t) value."
        )
        note_text.setWordWrap(True)
        note_text.setStyleSheet("font-size: 10px; color: #555;")
        note_layout.addWidget(note_text)
        ctrl_layout.addWidget(note_group)

        ctrl_layout.addStretch()
        root.addWidget(ctrl_panel, 0)

        # --- Right: Matplotlib canvas ---
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 8), constrained_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=False)
        canvas_layout.addWidget(self.canvas)

        root.addWidget(canvas_widget, 1)
        self.setCentralWidget(central)

        # Show a placeholder in the plots
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Signal preparation
    # ------------------------------------------------------------------

    def _on_prepare(self) -> None:
        """Generate signals from selected presets and prepare convolution."""
        x_name = self._x_selector.currentText()
        h_name = self._h_selector.currentText()
        try:
            result = ConvolutionAnalyzer.prepare_from_presets(
                x_preset_name=x_name,
                h_preset_name=h_name,
                num_samples=_NUM_SAMPLES,
                tau_start=-1.0,
                tau_end=1.0,
            )
        except (ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Preparation failed", str(exc))
            return

        self._timer.stop()
        self._result = result
        self._frame_index = 0
        self._output_t_so_far = np.array([], dtype=float)
        self._y_so_far = np.array([], dtype=float)
        self._handles = {}

        # Set up plots
        self._handles = setup_convolution_figure(
            self.figure,
            list(self.axes),
            result.tau,
            result.x_resampled,
            result.h_resampled,
            result.output_t,
            x_name=result.x_name,
            h_name=result.h_name,
        )

        # Configure shift slider range
        total = result.num_frames
        self._shift_slider.setRange(0, total - 1)
        self._shift_slider.setValue(0)
        self._update_shift_label(0)

        self._set_state(_READY)
        self._draw_frame(0)

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume animation."""
        if self._state == _IDLE:
            return
        if self._state == _DONE:
            # Replay from beginning
            self._frame_index = 0
            self._output_t_so_far = np.array([], dtype=float)
            self._y_so_far = np.array([], dtype=float)
            self._draw_frame(0)
        self._set_state(_PLAYING)
        self._timer.start()

    def pause(self) -> None:
        """Pause animation."""
        if self._state == _PLAYING:
            self._timer.stop()
            self._set_state(_PAUSED)

    def reset(self) -> None:
        """Stop animation and return to frame 0."""
        self._timer.stop()
        if self._result is None:
            self._set_state(_IDLE)
            return
        self._frame_index = 0
        self._output_t_so_far = np.array([], dtype=float)
        self._y_so_far = np.array([], dtype=float)
        self._shift_slider.blockSignals(True)
        self._shift_slider.setValue(0)
        self._shift_slider.blockSignals(False)
        self._set_state(_READY)
        self._draw_frame(0)

    def _on_speed_changed(self, value: int) -> None:
        """Adjust QTimer interval based on speed slider."""
        self._speed_label.setText(str(value))
        # Interval: speed 1 → 120 ms (slow), speed 10 → 12 ms (fast)
        interval = max(12, 132 - value * 12)
        self._timer.setInterval(interval)

    def _on_shift_slider_changed(self, slider_value: int) -> None:
        """Manual scrub: pause animation and jump to the selected frame."""
        if self._result is None or self._state == _IDLE:
            return
        if self._state == _PLAYING:
            self._timer.stop()
            self._set_state(_PAUSED)
        frame = int(np.clip(slider_value, 0, self._result.num_frames - 1))
        # Rebuild output buffer up to this frame
        self._frame_index = frame
        if frame == 0:
            self._output_t_so_far = np.array([], dtype=float)
            self._y_so_far = np.array([], dtype=float)
        else:
            result = self._result
            self._output_t_so_far = result.output_t[:frame + 1]
            self._y_so_far = result.y_reference[:frame + 1]
        self._update_shift_label(frame)
        self._draw_frame(frame)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _advance_frame(self) -> None:
        """Called by QTimer each tick. Advance by one frame."""
        if self._result is None:
            self._timer.stop()
            return
        result = self._result
        frame = self._frame_index

        if frame >= result.num_frames:
            self._timer.stop()
            self._set_state(_DONE)
            draw_final_result(
                list(self.axes),
                self._handles,
                result.output_t,
                result.y_reference,
                self.canvas,
            )
            return

        self._draw_frame(frame)

        # Append current point to progressive output
        t_k = result.output_t[frame]
        y_k = result.y_reference[frame]
        self._output_t_so_far = np.append(self._output_t_so_far, t_k)
        self._y_so_far = np.append(self._y_so_far, y_k)

        self._frame_index += 1

        # Sync the shift slider (suppress its valueChanged signal)
        self._shift_slider.blockSignals(True)
        self._shift_slider.setValue(frame)
        self._shift_slider.blockSignals(False)
        self._update_shift_label(frame)

    def _draw_frame(self, frame_index: int) -> None:
        """Render a single animation frame without advancing the counter."""
        if self._result is None or not self._handles:
            return
        result = self._result
        frame_index = int(np.clip(frame_index, 0, result.num_frames - 1))
        t_k = float(result.output_t[frame_index])
        y_k = float(result.y_reference[frame_index])
        h_shifted = ConvolutionAnalyzer.h_shifted(result, t_k)

        update_convolution_frame(
            axes=list(self.axes),
            handles=self._handles,
            tau=result.tau,
            h_shifted_values=h_shifted,
            x_resampled=result.x_resampled,
            output_t_so_far=self._output_t_so_far,
            y_so_far=self._y_so_far,
            current_t=t_k,
            current_y=y_k,
            canvas=self.canvas,
        )

        # Update info labels
        self._frame_label.setText(
            f"Frame: {frame_index + 1} / {result.num_frames}"
        )
        self._conv_value_label.setText(
            f"t = {t_k:.4g}\ny(t) = {y_k:.4g}"
        )

    def _update_shift_label(self, frame_index: int) -> None:
        """Update the text label showing the current shift value."""
        if self._result is None:
            self._shift_value_label.setText("t = --")
            return
        t_k = float(self._result.output_t[int(frame_index)])
        self._shift_value_label.setText(f"t = {t_k:.4g}")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        """Update internal state and refresh button enable/disable states."""
        self._state = state
        self._play_button.setEnabled(state in (_READY, _PAUSED, _DONE))
        self._pause_button.setEnabled(state == _PLAYING)
        self._reset_button.setEnabled(state in (_READY, _PLAYING, _PAUSED, _DONE))
        self._shift_slider.setEnabled(state in (_READY, _PAUSED, _DONE))

    # ------------------------------------------------------------------
    # Placeholder
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Show instructional placeholder text before signals are prepared."""
        for ax in self.axes:
            ax.clear()
            ax.set_visible(True)
        self.figure.set_constrained_layout(False)
        titles = [
            "Fixed signal   x(τ)  — select a preset and click Prepare",
            "Reversed & shifted kernel   h(t−τ)",
            "Convolution output   y(t) = x(t) ∗ h(t)",
        ]
        ylabels = ["Amplitude", "Amplitude", "y(t)"]
        for ax, title, ylabel in zip(self.axes, titles, ylabels):
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.axhline(0.0, color="0.7", linewidth=0.8)
        self.axes[2].set_xlabel("t")
        self.figure.tight_layout(h_pad=1.2)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop the QTimer before the window is destroyed."""
        self._timer.stop()
        super().closeEvent(event)
