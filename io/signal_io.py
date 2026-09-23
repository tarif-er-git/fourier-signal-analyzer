"""Compatibility location for the project's signal I/O helpers.

Use ``import signal_io`` in application code because Python's built-in
``io`` module takes precedence over a project package with the same name.
"""

from signal_io import (
    LoadedCurve,
    LoadedSignal,
    export_analysis_report,
    export_curve_coefficients,
    export_curve_csv,
    export_curve_reconstruction,
    export_error,
    export_reconstruction,
    export_spectrum,
    load_curve,
    load_signal,
    save_curve,
    save_signal,
)

__all__ = [
    "LoadedCurve",
    "LoadedSignal",
    "export_analysis_report",
    "export_curve_coefficients",
    "export_curve_csv",
    "export_curve_reconstruction",
    "export_error",
    "export_reconstruction",
    "export_spectrum",
    "load_curve",
    "load_signal",
    "save_curve",
    "save_signal",
]
