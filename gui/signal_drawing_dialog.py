"""Popup dialog for 1D custom signal drawing.

This dialog follows the same interaction model as the convolution window's
dedicated drawing page:
  - A :class:`~gui.drawing_canvas.SignalDrawingCanvas` occupies the main area.
  - Three action buttons are provided: Use This Signal / Clear Canvas / Cancel.
  - On confirmation the caller receives (t, x) arrays via the ``signal_accepted``
    Qt signal or by inspecting ``result_t`` / ``result_x`` after exec().

Usage example (in MainWindow)::

    dlg = SignalDrawingDialog(self)
    dlg.signal_accepted.connect(self.finish_custom_drawing)
    dlg.show()
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.drawing_canvas import SignalDrawingCanvas

# Number of samples on the time grid produced by the drawing canvas
_NUM_SAMPLES = 1001


class SignalDrawingDialog(QDialog):
    """Non-modal dialog that lets the user draw a 1D custom signal freehand.

    When the user clicks *Use This Signal*, ``signal_accepted`` is emitted with
    ``(t, x)`` arrays and the dialog closes.  The caller should connect to that
    signal before calling :py:meth:`show`.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget (keeps the dialog on top of the main window).
    num_samples : int
        Resolution of the uniform time grid returned on acceptance.
    t_range : (float, float)
        ``(t_min, t_max)`` limits of the drawing canvas.
    amp_range : (float, float)
        ``(amp_min, amp_max)`` limits of the drawing canvas.
    """

    #: Emitted when the user confirms the drawing.  Carries ``(t, x)``.
    signal_accepted = Signal(object, object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        num_samples: int = _NUM_SAMPLES,
        t_range: tuple[float, float] = (0.0, 1.0),
        amp_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw Custom Signal")
        self.setMinimumSize(720, 540)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Keep above parent but do not block interaction with the main window
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.result_t: np.ndarray | None = None
        self.result_x: np.ndarray | None = None

        self._build_layout(num_samples, t_range, amp_range)
        self._drawing_canvas.start_drawing(title="Draw 1D Signal  (t = 0 to 1)")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(
        self,
        num_samples: int,
        t_range: tuple[float, float],
        amp_range: tuple[float, float],
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- Header ---
        title_label = QLabel("\u270f\ufe0f  Draw 1D Custom Signal")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        instruction_label = QLabel(
            "Press and drag the left mouse button to draw your signal.\n"
            "Time axis: {} to {}   |   Amplitude axis: {} to {}.\n"
            "Click \u2018Use This Signal\u2019 when satisfied.".format(
                t_range[0], t_range[1], amp_range[0], amp_range[1]
            )
        )
        instruction_label.setStyleSheet("font-size: 11px; color: #555;")
        instruction_label.setWordWrap(True)
        root.addWidget(title_label)
        root.addWidget(instruction_label)

        # --- Drawing canvas ---
        self._drawing_canvas = SignalDrawingCanvas(
            self,
            num_samples=num_samples,
            t_range=t_range,
            amp_range=amp_range,
        )
        root.addWidget(self._drawing_canvas, 1)

        # --- Button bar ---
        btn_bar = QHBoxLayout()

        self._use_btn = QPushButton("\u2713  Use This Signal")
        self._use_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2e7d32;"
            " color: white; padding: 6px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388e3c; }"
        )
        self._use_btn.clicked.connect(self._on_use_signal)

        self._clear_btn = QPushButton("Clear Canvas")
        self._clear_btn.clicked.connect(self._drawing_canvas.clear_drawing)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)

        btn_bar.addWidget(self._use_btn)
        btn_bar.addWidget(self._clear_btn)
        btn_bar.addStretch()
        btn_bar.addWidget(self._cancel_btn)
        root.addLayout(btn_bar)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_use_signal(self) -> None:
        """Validate, store, emit, and close when the user accepts the drawing."""
        try:
            t, x = self._drawing_canvas.finish_drawing()
        except ValueError as exc:
            QMessageBox.warning(self, "Drawing not ready", str(exc))
            return

        self.result_t = t
        self.result_x = x
        self.signal_accepted.emit(t, x)
        self.accept()
