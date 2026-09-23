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

from signal.curve2d import Curve2D


@dataclass(frozen=True)
class LoadedSignal:
    """A validated signal and its optional saved metadata."""

    time: np.ndarray
    signal: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class LoadedCurve:
    """A validated 2D curve and its saved metadata."""

    curve: Curve2D
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
    if (
        payload.get("format") == "signal-sketch-and-decompose.curve2d.v1"
        or payload.get("type") == "2d_closed_curve"
    ):
        raise ValueError(
            "This file contains a 2D closed curve. Please use File -> Load 2D Curve instead."
        )
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


def save_curve(
    path: str | Path,
    curve: Curve2D,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save raw and processed numerical data for a closed 2D curve."""
    if not isinstance(curve, Curve2D) or not curve.is_valid:
        raise ValueError("a valid closed 2D curve is required")
    saved_metadata = _json_metadata(metadata)
    saved_metadata.setdefault("point_count", curve.point_count)
    saved_metadata.setdefault("perimeter", curve.perimeter)
    payload = {
        "format": "signal-sketch-and-decompose.curve2d.v1",
        "type": "2d_closed_curve",
        "version": 1,
        "closed": curve.is_closed,
        "raw_points": curve.raw_points.tolist(),
        "cleaned_points": curve.cleaned_points.tolist(),
        "normalized_points": curve.normalized_points.tolist(),
        "metadata": saved_metadata,
    }
    destination = Path(path)
    try:
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as error:
        raise OSError(f"could not save 2D curve to {destination}") from error


def load_curve(path: str | Path) -> LoadedCurve:
    """Load a versioned 2D curve JSON and rebuild its processed model."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid 2D curve JSON from {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("2D curve file must contain a JSON object")
    if payload.get("format") == "signal-sketch-and-decompose.signal.v1":
        raise ValueError(
            "This file contains a 1D signal. Please use File -> Load Signal instead."
        )
    if payload.get("format") != "signal-sketch-and-decompose.curve2d.v1":
        raise ValueError("unsupported or missing 2D curve file format")
    if payload.get("type") != "2d_closed_curve":
        raise ValueError("file is not a 2D closed curve")
    if payload.get("version") != 1:
        raise ValueError("unsupported 2D curve file version")
    raw_points = payload.get("raw_points")
    if raw_points is None:
        raw_points = payload.get("cleaned_points")
    if raw_points is None:
        raise ValueError("2D curve file must contain points")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    curve = Curve2D.from_points(raw_points, close=bool(payload.get("closed", True)))
    if not curve.is_valid:
        raise ValueError("loaded 2D curve must contain at least three valid points")
    return LoadedCurve(curve, dict(metadata))


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


def export_curve_csv(path: str | Path, curve: Curve2D) -> None:
    """Export cleaned mathematical curve vertices as ``index,x,y`` CSV."""
    if not isinstance(curve, Curve2D) or not curve.is_valid:
        raise ValueError("a valid closed 2D curve is required")
    rows = ((index, point[0], point[1]) for index, point in enumerate(curve.cleaned_points))
    _write_csv(path, ["index", "x", "y"], rows)


def export_curve_reconstruction(path: str | Path, coefficients, reconstruction) -> None:
    """Export original/reconstructed 2D samples and pointwise distance error."""
    original_x = np.asarray(coefficients.x_samples, dtype=float)
    original_y = np.asarray(coefficients.y_samples, dtype=float)
    reconstructed_x = np.asarray(reconstruction.x, dtype=float)
    reconstructed_y = np.asarray(reconstruction.y, dtype=float)
    if not (
        original_x.shape == original_y.shape == reconstructed_x.shape == reconstructed_y.shape
    ) or original_x.ndim != 1:
        raise ValueError("2D reconstruction arrays must have matching one-dimensional shapes")
    if not all(np.all(np.isfinite(values)) for values in (original_x, original_y, reconstructed_x, reconstructed_y)):
        raise ValueError("2D reconstruction values must be finite")
    error = np.hypot(original_x - reconstructed_x, original_y - reconstructed_y)
    rows = zip(range(original_x.size), original_x, original_y, reconstructed_x, reconstructed_y, error)
    _write_csv(path, ["index", "original_x", "original_y", "reconstructed_x", "reconstructed_y", "error"], rows)


def export_curve_coefficients(path: str | Path, coefficients) -> None:
    """Export signed X/Y complex Fourier coefficients and derived values."""
    rows = zip(
        coefficients.harmonics,
        coefficients.x_coefficients.real,
        coefficients.x_coefficients.imag,
        coefficients.x_magnitudes,
        coefficients.x_phases,
        coefficients.y_coefficients.real,
        coefficients.y_coefficients.imag,
        coefficients.y_magnitudes,
        coefficients.y_phases,
    )
    _write_csv(
        path,
        ["k", "Cx_real", "Cx_imag", "Cx_magnitude", "Cx_phase", "Cy_real", "Cy_imag", "Cy_magnitude", "Cy_phase"],
        rows,
    )


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


