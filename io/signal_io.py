"""Compatibility location for the project's signal I/O helpers.

Use ``import signal_io`` in application code because Python's built-in
``io`` module takes precedence over a project package with the same name.
"""

from signal_io import (
    LoadedSignal,
    export_analysis_report,
    export_error,
    export_reconstruction,
    export_spectrum,
    load_signal,
    save_signal,
)

__all__ = [
    "LoadedSignal",
    "export_analysis_report",
    "export_error",
    "export_reconstruction",
    "export_spectrum",
    "load_signal",
    "save_signal",
]
