"""Self-contained PySide6 window for the convolution simulation animation.

Architecture
------------
This window follows the same pattern as Curve2DEpicycleWindow:
  - A QMainWindow with an embedded FigureCanvasQTAgg
  - A QTimer drives per-frame updates (not matplotlib.animation.FuncAnimation)
  - All mathematical logic is delegated to signal.convolution.ConvolutionAnalyzer
  - All drawing logic is delegated to visualization.convolution_plot
  - Custom 1D signal drawing is delegated to gui.drawing_canvas.SignalDrawingCanvas

The window supports dual input methods for both inputs:
  1. Signal x(t): Preset selector OR Custom 1D drawing
  2. Signal h(t): Preset selector OR Custom 1D drawing

All four input combinations are supported with completely independent signal states:
  - Preset x(t) + Preset h(t)
  - Custom x(t) + Preset h(t)
  - Preset x(t) + Custom h(t)
  - Custom x(t) + Custom h(t)

Animation State Machine
-----------------------
IDLE    -> no signals prepared yet / waiting for prepare
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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.drawing_canvas import SignalDrawingCanvas
from signal.convolution import (
    ConvolutionAnalyzer,
    ConvolutionResult,
    CONVOLUTION_PRESETS,
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

    Both signals x(t) and h(t) can independently be selected from built-in
    presets or drawn manually using the embedded 1D signal drawing canvas.
    """

    FRAME_INTERVAL_MS = _DEFAULT_FRAME_INTERVAL_MS

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convolution Simulation")
        self.resize(1040, 780)

        # Core state
        self._state: str = _IDLE
        self._result: ConvolutionResult | None = None
        self._frame_index: int = 0
        self._output_t_so_far: np.ndarray = np.array([], dtype=float)
        self._y_so_far: np.ndarray = np.array([], dtype=float)
        self._handles: dict[str, object] = {}

        # Signal source selection and independent custom signal storage
        self._x_mode: str = "preset"  # "preset" or "custom"
        self._h_mode: str = "preset"  # "preset" or "custom"
        self._custom_x: tuple[np.ndarray, np.ndarray] | None = None
        self._custom_h: tuple[np.ndarray, np.ndarray] | None = None
        self._active_drawing_target: str = "x"  # "x" or "h"

        # QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)

        self._build_layout()
        self._set_state(_IDLE)
        self._update_previews()

    # ------------------------------------------------------------------
    # Layout Construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Construct the window layout: controls on left, stacked views on right."""
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ==============================================================
        # --- Left control panel in Scroll Area ---
        # ==============================================================
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setFixedWidth(295)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        ctrl_panel = QWidget()
        ctrl_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(4, 4, 8, 4)
        ctrl_layout.setSpacing(10)

        # --- Signal Selection Group ---
        signal_group = QGroupBox("Signal Selection")
        signal_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sg_layout = QVBoxLayout(signal_group)
        sg_layout.setContentsMargins(8, 12, 8, 10)
        sg_layout.setSpacing(6)

        # Signal x(t) section
        x_title = QLabel("Signal  x(t)")
        x_title.setStyleSheet("font-weight: bold; color: #1f77b4;")
        sg_layout.addWidget(x_title)

        x_src_layout = QHBoxLayout()
        x_src_layout.addWidget(QLabel("Source:"))
        self._x_source_combo = QComboBox()
        self._x_source_combo.addItems(["Preset", "Custom Draw"])
        self._x_source_combo.currentIndexChanged.connect(self._on_x_source_changed)
        x_src_layout.addWidget(self._x_source_combo, 1)
        sg_layout.addLayout(x_src_layout)

        # x Preset selector
        self._x_preset_combo = QComboBox()
        for name in X_PRESET_NAMES:
            self._x_preset_combo.addItem(name)
        self._x_preset_combo.currentIndexChanged.connect(self._on_x_preset_changed)
        sg_layout.addWidget(self._x_preset_combo)

        # x Custom draw buttons
        self._x_draw_box = QWidget()
        x_btn_layout = QHBoxLayout(self._x_draw_box)
        x_btn_layout.setContentsMargins(0, 2, 0, 2)
        x_btn_layout.setSpacing(6)
        self._x_draw_btn = QPushButton("✏️ Draw x(t)")
        self._x_draw_btn.clicked.connect(self._on_draw_x)
        self._x_clear_btn = QPushButton("Clear")
        self._x_clear_btn.clicked.connect(self._on_clear_x)
        x_btn_layout.addWidget(self._x_draw_btn, 1)
        x_btn_layout.addWidget(self._x_clear_btn)
        self._x_draw_box.setVisible(False)
        sg_layout.addWidget(self._x_draw_box)

        # x Status / preview text
        self._x_status_label = QLabel("Preset: Rectangular Pulse")
        self._x_status_label.setWordWrap(True)
        self._x_status_label.setStyleSheet("font-size: 11px; color: #444; padding-top: 2px;")
        sg_layout.addWidget(self._x_status_label)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        sg_layout.addWidget(line)

        # Signal h(t) section
        h_title = QLabel("Kernel  h(t)")
        h_title.setStyleSheet("font-weight: bold; color: #ff7f0e;")
        sg_layout.addWidget(h_title)

        h_src_layout = QHBoxLayout()
        h_src_layout.addWidget(QLabel("Source:"))
        self._h_source_combo = QComboBox()
        self._h_source_combo.addItems(["Preset", "Custom Draw"])
        self._h_source_combo.currentIndexChanged.connect(self._on_h_source_changed)
        h_src_layout.addWidget(self._h_source_combo, 1)
        sg_layout.addLayout(h_src_layout)

        # h Preset selector
        self._h_preset_combo = QComboBox()
        for name in H_PRESET_NAMES:
            self._h_preset_combo.addItem(name)
        self._h_preset_combo.currentIndexChanged.connect(self._on_h_preset_changed)
        sg_layout.addWidget(self._h_preset_combo)

        # h Custom draw buttons
        self._h_draw_box = QWidget()
        h_btn_layout = QHBoxLayout(self._h_draw_box)
        h_btn_layout.setContentsMargins(0, 2, 0, 2)
        h_btn_layout.setSpacing(6)
        self._h_draw_btn = QPushButton("✏️ Draw h(t)")
        self._h_draw_btn.clicked.connect(self._on_draw_h)
        self._h_clear_btn = QPushButton("Clear")
        self._h_clear_btn.clicked.connect(self._on_clear_h)
        h_btn_layout.addWidget(self._h_draw_btn, 1)
        h_btn_layout.addWidget(self._h_clear_btn)
        self._h_draw_box.setVisible(False)
        sg_layout.addWidget(self._h_draw_box)

        # h Status / preview text
        self._h_status_label = QLabel("Preset: Rectangular Pulse")
        self._h_status_label.setWordWrap(True)
        self._h_status_label.setStyleSheet("font-size: 11px; color: #444; padding-top: 2px;")
        sg_layout.addWidget(self._h_status_label)

        ctrl_layout.addWidget(signal_group)

        # Prepare button
        self._prepare_button = QPushButton("Prepare Convolution")
        self._prepare_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        self._prepare_button.setToolTip(
            "Resample both signals onto a common grid\n"
            "and compute the reference convolution."
        )
        self._prepare_button.clicked.connect(self._on_prepare)
        ctrl_layout.addWidget(self._prepare_button)

        # Playback group
        pb_group = QGroupBox("Playback")
        pb_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        pb_layout = QVBoxLayout(pb_group)
        pb_layout.setContentsMargins(8, 12, 8, 10)
        pb_layout.setSpacing(8)

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
        self._speed_slider.setRange(0, 50)
        self._speed_slider.setValue(25)
        self._speed_slider.setToolTip("Animation speed (0=slowest, 50=fastest)")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        self._speed_label = QLabel("25")
        self._speed_label.setMinimumWidth(24)
        speed_row.addWidget(self._speed_slider, 1)
        speed_row.addWidget(self._speed_label)
        pb_layout.addLayout(speed_row)

        ctrl_layout.addWidget(pb_group)

        # Manual shift group
        shift_group = QGroupBox("Manual Shift  t")
        shift_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sh_layout = QVBoxLayout(shift_group)
        sh_layout.setContentsMargins(8, 12, 8, 10)
        sh_layout.setSpacing(6)
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
        info_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 12, 8, 10)
        info_layout.setSpacing(6)
        self._frame_label = QLabel("Frame: --")
        self._conv_value_label = QLabel("y(t) = --")
        self._conv_value_label.setWordWrap(True)
        info_layout.addWidget(self._frame_label)
        info_layout.addWidget(self._conv_value_label)
        ctrl_layout.addWidget(info_group)

        # Educational note
        note_group = QGroupBox("Convention")
        note_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        note_layout = QVBoxLayout(note_group)
        note_layout.setContentsMargins(8, 12, 8, 10)
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
        self._scroll_area.setWidget(ctrl_panel)
        root.addWidget(self._scroll_area, 0)

        # ==============================================================
        # --- Right: Stacked Views (Simulation vs Drawing) ---
        # ==============================================================
        self._stacked_widget = QStackedWidget()

        # Page 0: Simulation canvas (3 subplots)
        self._simulation_widget = QWidget()
        sim_layout = QVBoxLayout(self._simulation_widget)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 8), constrained_layout=False)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=False)
        sim_layout.addWidget(self.canvas)
        self._stacked_widget.addWidget(self._simulation_widget)

        # Page 1: Dedicated 1D Signal Drawing View
        self._drawing_widget = QWidget()
        draw_layout = QVBoxLayout(self._drawing_widget)
        draw_layout.setContentsMargins(4, 4, 4, 4)
        draw_layout.setSpacing(6)

        # Drawing Header Banner
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(8, 8, 8, 8)
        self._draw_title_label = QLabel("✏️ Drawing Signal x(t)")
        self._draw_title_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._draw_instruction_label = QLabel(
            "Press and drag the left mouse button across the canvas to shape your 1D signal.\n"
            "Axes represent Time t in [-1.0, 1.0] and Amplitude in [-1.0, 1.0]. Click 'Use This Signal' when done."
        )
        self._draw_instruction_label.setStyleSheet("font-size: 11px; color: #555;")
        self._draw_instruction_label.setWordWrap(True)
        header_layout.addWidget(self._draw_title_label)
        header_layout.addWidget(self._draw_instruction_label)
        draw_layout.addWidget(header_widget)

        # Canvas widget
        self._drawing_canvas = SignalDrawingCanvas(
            self,
            num_samples=_NUM_SAMPLES,
            t_range=(-1.0, 1.0),
            amp_range=(-1.0, 1.0),
        )
        draw_layout.addWidget(self._drawing_canvas, 1)

        # Drawing Action Buttons
        draw_btn_bar = QHBoxLayout()
        self._use_drawn_btn = QPushButton("✓ Use This Signal")
        self._use_drawn_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2e7d32; color: white; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #388e3c; }"
        )
        self._use_drawn_btn.clicked.connect(self._on_use_drawn_signal)

        self._clear_canvas_btn = QPushButton("Clear Canvas")
        self._clear_canvas_btn.clicked.connect(self._drawing_canvas.clear_drawing)

        self._cancel_draw_btn = QPushButton("Cancel / Back to Simulation")
        self._cancel_draw_btn.clicked.connect(self._on_cancel_drawing)

        draw_btn_bar.addWidget(self._use_drawn_btn)
        draw_btn_bar.addWidget(self._clear_canvas_btn)
        draw_btn_bar.addStretch()
        draw_btn_bar.addWidget(self._cancel_draw_btn)
        draw_layout.addLayout(draw_btn_bar)

        self._stacked_widget.addWidget(self._drawing_widget)

        root.addWidget(self._stacked_widget, 1)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Source Switching and Custom Drawing Handlers
    # ------------------------------------------------------------------

    def _on_x_source_changed(self, index: int) -> None:
        """Handle switching between Preset and Custom Draw for Signal x(t)."""
        is_custom = index == 1
        self._x_mode = "custom" if is_custom else "preset"
        self._x_preset_combo.setVisible(not is_custom)
        self._x_draw_box.setVisible(is_custom)

        self._update_x_status()
        self._reset_simulation_data()
        self._update_previews()

    def _on_h_source_changed(self, index: int) -> None:
        """Handle switching between Preset and Custom Draw for Signal h(t)."""
        is_custom = index == 1
        self._h_mode = "custom" if is_custom else "preset"
        self._h_preset_combo.setVisible(not is_custom)
        self._h_draw_box.setVisible(is_custom)

        self._update_h_status()
        self._reset_simulation_data()
        self._update_previews()

    def _on_x_preset_changed(self) -> None:
        """Handle preset selection change for x(t)."""
        self._update_x_status()
        self._reset_simulation_data()
        self._update_previews()

    def _on_h_preset_changed(self) -> None:
        """Handle preset selection change for h(t)."""
        self._update_h_status()
        self._reset_simulation_data()
        self._update_previews()

    def _update_x_status(self) -> None:
        """Update the status label for Signal x(t)."""
        if self._x_mode == "preset":
            name = self._x_preset_combo.currentText()
            self._x_status_label.setText(
                f"Preset: {name} ✓\nSamples: {_NUM_SAMPLES} | Range: [-1.0, 1.0]"
            )
        else:
            if self._custom_x is not None:
                t, _ = self._custom_x
                t_min, t_max = float(t[0]), float(t[-1])
                self._x_status_label.setText(
                    f"Signal x(t): Custom ✓\nSamples: {len(t)} | Range: [{t_min:.2g}, {t_max:.2g}]"
                )
            else:
                self._x_status_label.setText("Signal x(t): Not drawn yet\n(Click 'Draw x(t)')")

    def _update_h_status(self) -> None:
        """Update the status label for Kernel h(t)."""
        if self._h_mode == "preset":
            name = self._h_preset_combo.currentText()
            self._h_status_label.setText(
                f"Preset: {name} ✓\nSamples: {_NUM_SAMPLES} | Range: [-1.0, 1.0]"
            )
        else:
            if self._custom_h is not None:
                t, _ = self._custom_h
                t_min, t_max = float(t[0]), float(t[-1])
                self._h_status_label.setText(
                    f"Kernel h(t): Custom ✓\nSamples: {len(t)} | Range: [{t_min:.2g}, {t_max:.2g}]"
                )
            else:
                self._h_status_label.setText("Kernel h(t): Not drawn yet\n(Click 'Draw h(t)')")

    def _on_draw_x(self) -> None:
        """Activate the drawing canvas for Signal x(t)."""
        self._active_drawing_target = "x"
        self._draw_title_label.setText("✏️ Drawing Signal x(t)")
        self._drawing_canvas.start_drawing(
            title="Drawing Signal x(t)", initial_signal=self._custom_x
        )
        self._stacked_widget.setCurrentIndex(1)

    def _on_draw_h(self) -> None:
        """Activate the drawing canvas for Signal h(t)."""
        self._active_drawing_target = "h"
        self._draw_title_label.setText("✏️ Drawing Kernel h(t)")
        self._drawing_canvas.start_drawing(
            title="Drawing Kernel h(t)", initial_signal=self._custom_h
        )
        self._stacked_widget.setCurrentIndex(1)

    def _on_use_drawn_signal(self) -> None:
        """Finalize drawing and save as x(t) or h(t) independently."""
        try:
            t, x_vals = self._drawing_canvas.finish_drawing()
        except ValueError as exc:
            QMessageBox.warning(self, "Drawing not ready", str(exc))
            return

        if self._active_drawing_target == "x":
            self._custom_x = (t.copy(), x_vals.copy())
            self._update_x_status()
        else:
            self._custom_h = (t.copy(), x_vals.copy())
            self._update_h_status()

        self._stacked_widget.setCurrentIndex(0)
        self._reset_simulation_data()
        self._update_previews()

    def _on_clear_x(self) -> None:
        """Clear the custom drawing for x(t) without touching h(t)."""
        self._custom_x = None
        self._update_x_status()
        self._reset_simulation_data()
        self._update_previews()

    def _on_clear_h(self) -> None:
        """Clear the custom drawing for h(t) without touching x(t)."""
        self._custom_h = None
        self._update_h_status()
        self._reset_simulation_data()
        self._update_previews()

    def _on_cancel_drawing(self) -> None:
        """Return to the simulation view without modifying signals."""
        self._stacked_widget.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Signal Preparation and Validation
    # ------------------------------------------------------------------

    def _reset_simulation_data(self) -> None:
        """Reset active animation state when inputs change."""
        self._timer.stop()
        self._result = None
        self._frame_index = 0
        self._output_t_so_far = np.array([], dtype=float)
        self._y_so_far = np.array([], dtype=float)
        self._handles = {}
        self._shift_slider.blockSignals(True)
        self._shift_slider.setValue(0)
        self._shift_slider.blockSignals(False)
        self._frame_label.setText("Frame: --")
        self._conv_value_label.setText("y(t) = --")
        self._shift_value_label.setText("t = --")
        self._set_state(_IDLE)

    def _get_active_signals(
        self,
    ) -> tuple[str | tuple[np.ndarray, np.ndarray], str | tuple[np.ndarray, np.ndarray], str, str]:
        """Validate and return the input representations and labels for x and h.

        Raises
        ------
        ValueError
            If either custom signal has not been drawn yet.
        """
        # Resolve x
        if self._x_mode == "preset":
            x_input = self._x_preset_combo.currentText()
            x_name = f"x(t): {x_input}"
        else:
            if self._custom_x is None:
                raise ValueError("Signal x(t) is set to Custom Draw, but has not been drawn yet.")
            x_input = self._custom_x
            x_name = "x(t) [Custom]"

        # Resolve h
        if self._h_mode == "preset":
            h_input = self._h_preset_combo.currentText()
            h_name = f"h(t): {h_input}"
        else:
            if self._custom_h is None:
                raise ValueError("Kernel h(t) is set to Custom Draw, but has not been drawn yet.")
            h_input = self._custom_h
            h_name = "h(t) [Custom]"

        return x_input, h_input, x_name, h_name

    def _on_prepare(self) -> None:
        """Resample inputs, compute reference convolution, and prepare animation."""
        # 1. Validation
        if self._x_mode == "custom" and self._custom_x is None:
            QMessageBox.warning(
                self,
                "Signal x(t) missing",
                "Signal x(t) is set to Custom Draw, but has not been drawn yet.\n"
                "Please click '✏️ Draw x(t)' to draw the signal first.",
            )
            return

        if self._h_mode == "custom" and self._custom_h is None:
            QMessageBox.warning(
                self,
                "Kernel h(t) missing",
                "Kernel h(t) is set to Custom Draw, but has not been drawn yet.\n"
                "Please click '✏️ Draw h(t)' to draw the kernel first.",
            )
            return

        try:
            x_input, h_input, x_name, h_name = self._get_active_signals()
            result = ConvolutionAnalyzer.prepare_signals(
                x_input=x_input,
                h_input=h_input,
                num_samples=_NUM_SAMPLES,
                x_name=x_name,
                h_name=h_name,
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
    # Previews & Placeholders
    # ------------------------------------------------------------------

    def _update_previews(self) -> None:
        """Render a live preview of the currently selected inputs on axes."""
        if self._result is not None:
            return  # Active simulation exists

        tau = np.linspace(-1.0, 1.0, _NUM_SAMPLES)
        for ax in self.axes:
            ax.clear()
            ax.set_visible(True)

        # Top panel: x(τ) preview
        ax0 = self.axes[0]
        if self._x_mode == "preset":
            x_name = self._x_preset_combo.currentText()
            _, x_fn = CONVOLUTION_PRESETS[x_name]
            x_vals = x_fn(tau)
            ax0.plot(tau, x_vals, color="#1f77b4", linewidth=1.8, label=x_name)
            ax0.set_title(f"Fixed signal   x(τ)  [{x_name}]", fontsize=10, loc="left")
        else:
            if self._custom_x is not None:
                t_x, x_vals = self._custom_x
                ax0.plot(t_x, x_vals, color="#1f77b4", linewidth=1.8, label="Custom x(t)")
                ax0.set_title("Fixed signal   x(τ)  [Custom ✓]", fontsize=10, loc="left")
            else:
                ax0.text(
                    0.5, 0.5,
                    "Signal x(t) not drawn yet.\nClick '✏️ Draw x(t)' to draw.",
                    ha="center", va="center", transform=ax0.transAxes, color="#888",
                )
                ax0.set_title("Fixed signal   x(τ)  [Custom - Not Drawn]", fontsize=10, loc="left")

        # Middle panel: h(τ) preview
        ax1 = self.axes[1]
        if self._h_mode == "preset":
            h_name = self._h_preset_combo.currentText()
            _, h_fn = CONVOLUTION_PRESETS[h_name]
            h_vals = h_fn(tau)
            ax1.plot(tau, h_vals, color="#ff7f0e", linewidth=1.8, label=h_name)
            ax1.set_title(
                f"Kernel   h(τ)  [{h_name}]  (will be reversed & shifted during simulation)",
                fontsize=10,
                loc="left",
            )
        else:
            if self._custom_h is not None:
                t_h, h_vals = self._custom_h
                ax1.plot(t_h, h_vals, color="#ff7f0e", linewidth=1.8, label="Custom h(t)")
                ax1.set_title(
                    "Kernel   h(τ)  [Custom ✓]  (will be reversed & shifted)",
                    fontsize=10,
                    loc="left",
                )
            else:
                ax1.text(
                    0.5, 0.5,
                    "Kernel h(t) not drawn yet.\nClick '✏️ Draw h(t)' to draw.",
                    ha="center", va="center", transform=ax1.transAxes, color="#888",
                )
                ax1.set_title("Kernel   h(τ)  [Custom - Not Drawn]", fontsize=10, loc="left")

        # Bottom panel: convolution output placeholder
        ax2 = self.axes[2]
        ax2.set_title("Convolution output   y(t) = x(t) ∗ h(t)", fontsize=10, loc="left")
        ax2.text(
            0.5, 0.5,
            "Click 'Prepare Convolution' to compute overlap and start animation.",
            ha="center", va="center", transform=ax2.transAxes, color="#666",
        )

        for ax in self.axes:
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.25)
            ax.axhline(0.0, color="0.7", linewidth=0.8)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

        self.axes[2].set_xlabel("t")
        self.figure.tight_layout(h_pad=1.2)
        self.canvas.draw_idle()

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
        """Adjust QTimer interval based on speed slider (0=slowest, 50=fastest)."""
        self._speed_label.setText(str(value))
        if value <= 25:
            interval = int(round(200 - (value / 25.0) * (200 - 30)))
        else:
            interval = int(round(30 - ((value - 25.0) / 25.0) * (30 - 5)))
        interval = max(4, interval)
        self._timer.setInterval(interval)

    def _on_shift_slider_changed(self, slider_value: int) -> None:
        """Manual scrub: pause animation and jump to the selected frame."""
        if self._result is None or self._state == _IDLE:
            return
        if self._state == _PLAYING:
            self._timer.stop()
            self._set_state(_PAUSED)
        frame = int(np.clip(slider_value, 0, self._result.num_frames - 1))
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
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop the QTimer before the window is destroyed."""
        self._timer.stop()
        super().closeEvent(event)
