"""Readable signal storage and numerical CSV export helpers.

The requested ``io/signal_io.py`` compatibility module re-exports this
implementation. The importable project module is named ``signal_io`` because
Python's standard-library ``io`` module is not a package and cannot reliably
contain project submodules.
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LoadedSignal:
    """A validated signal and its optional saved metadata."""

    time: np.ndarray
    signal: np.ndarray
    metadata: dict[str, object]


def _validate_signal_arrays(
    time: Sequence[float], signal: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate arrays before saving or exporting them."""
    time_values = np.asarray(time, dtype=float)
    signal_values = np.asarray(signal, dtype=float)
    if time_values.ndim != 1 or signal_values.ndim != 1:
        raise ValueError("time and signal must be one-dimensional")
    if time_values.size < 2:
        raise ValueError("at least two samples are required")
    if time_values.shape != signal_values.shape:
        raise ValueError("time and signal must have the same shape")
    if not np.all(np.isfinite(time_values)) or not np.all(np.isfinite(signal_values)):
        raise ValueError("time and signal must contain only finite values")
    if not np.all(np.diff(time_values) > 0):
        raise ValueError("time values must be strictly increasing")
    return time_values.copy(), signal_values.copy()


def _json_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Validate that metadata can be represented in JSON."""
    if metadata is None:
        return {}
    result = dict(metadata)
    try:
        json.dumps(result)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain JSON-compatible values") from error
    return result


def save_signal(
    path: str | Path,
    time: Sequence[float],
    signal: Sequence[float],
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save a signal as readable JSON containing arrays and metadata."""
    time_values, signal_values = _validate_signal_arrays(time, signal)
    saved_metadata = _json_metadata(metadata)
    saved_metadata.setdefault("number_of_samples", int(time_values.size))
    saved_metadata.setdefault("start_time", float(time_values[0]))
    saved_metadata.setdefault("end_time", float(time_values[-1]))
    payload = {
        "format": "signal-sketch-and-decompose.signal.v1",
        "metadata": saved_metadata,
        "time": time_values.tolist(),
        "signal": signal_values.tolist(),
    }
    destination = Path(path)
    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as error:
        raise OSError(f"could not save signal to {destination}") from error


def load_signal(path: str | Path) -> LoadedSignal:
    """Load and validate a JSON signal file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid signal JSON from {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("signal file must contain a JSON object")
    if payload.get("format") != "signal-sketch-and-decompose.signal.v1":
        raise ValueError("unsupported or missing signal file format")
    if "time" not in payload or "signal" not in payload:
        raise ValueError("signal file must contain time and signal fields")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    time_values, signal_values = _validate_signal_arrays(
        payload["time"], payload["signal"]
    )
    return LoadedSignal(time_values, signal_values, dict(metadata))


def _write_csv(path: str | Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Write a small UTF-8 CSV file with a header row."""
    destination = Path(path)
    try:
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
    except OSError as error:
        raise OSError(f"could not write CSV to {destination}") from error


def export_reconstruction(
    path: str | Path,
    time: Sequence[float],
    reconstructed: Sequence[float],
) -> None:
    """Export clean time and reconstructed values as numerical CSV columns."""
    time_values, reconstructed_values = _validate_signal_arrays(time, reconstructed)
    headers = ["time", "reconstructed"]
    rows = zip(time_values, reconstructed_values)
    _write_csv(path, headers, rows)


def export_spectrum(
    path: str | Path,
    harmonics: Sequence[int],
    magnitude: Sequence[float],
    phase: Sequence[float],
) -> None:
    """Export harmonic number, magnitude, and phase as CSV."""
    harmonic_values = np.asarray(harmonics)
    magnitude_values = np.asarray(magnitude, dtype=float)
    phase_values = np.asarray(phase, dtype=float)
    if (
        harmonic_values.ndim != 1
        or magnitude_values.ndim != 1
        or phase_values.ndim != 1
        or not (
            harmonic_values.shape == magnitude_values.shape == phase_values.shape
        )
    ):
        raise ValueError("harmonics, magnitude, and phase must have matching one-dimensional shapes")
    if not np.all(np.isfinite(harmonic_values)) or not np.all(
        np.isfinite(magnitude_values)
    ) or not np.all(np.isfinite(phase_values)):
        raise ValueError("spectrum values must be finite")
    _write_csv(path, ["harmonic", "magnitude", "phase"], zip(harmonic_values, magnitude_values, phase_values))


def export_error(path: str | Path, time: Sequence[float], error: Sequence[float]) -> None:
    """Export time and precomputed reconstruction error as CSV."""
    time_values, error_values = _validate_signal_arrays(time, error)
    _write_csv(path, ["time", "error"], zip(time_values, error_values))


def export_analysis_report(path: str | Path, report: Mapping[str, object]) -> None:
    """Export a JSON analysis report containing already-computed results."""
    try:
        text = json.dumps(dict(report), indent=2)
    except (TypeError, ValueError) as error:
        raise ValueError("analysis report must contain JSON-compatible values") from error
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as error:
        raise OSError(f"could not write analysis report to {path}") from error


