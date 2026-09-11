"""Tests for signal persistence and numerical exports."""

import csv
import json

import numpy as np
import pytest

from signal_io import (
    export_error,
    export_reconstruction,
    export_spectrum,
    load_signal,
    save_signal,
)


def test_save_and_load_preserves_signal_and_metadata(tmp_path) -> None:
    time = np.linspace(0.0, 1.0, 101)
    signal = np.sin(2.0 * np.pi * time)
    path = tmp_path / "signal.json"

    save_signal(path, time, signal, {"name": "sine", "type": "preset", "period": 1.0})
    loaded = load_signal(path)

    np.testing.assert_allclose(loaded.time, time)
    np.testing.assert_allclose(loaded.signal, signal)
    assert loaded.metadata["name"] == "sine"
    assert loaded.metadata["type"] == "preset"
    assert loaded.metadata["period"] == 1.0
    assert loaded.metadata["number_of_samples"] == 101


def test_malformed_signal_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps({"time": [0.0], "signal": [0.0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="format"):
        load_signal(path)


def test_shape_and_nonfinite_values_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="same shape"):
        save_signal(tmp_path / "bad.json", [0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        save_signal(tmp_path / "bad.json", [0.0, np.nan], [1.0, 2.0])


def test_reconstruction_export_is_readable_csv(tmp_path) -> None:
    time = np.array([0.0, 0.5, 1.0])
    values = np.array([1.0, 0.0, -1.0])
    path = tmp_path / "reconstruction.csv"

    export_reconstruction(path, time, values)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[0]["time"] == "0.0"
    assert rows[1]["reconstructed"] == "0.0"


def test_spectrum_export_preserves_columns(tmp_path) -> None:
    path = tmp_path / "spectrum.csv"

    export_spectrum(path, [1, 3], [1.0, 0.333], [0.0, -1.57])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[0] == {"harmonic": "1", "magnitude": "1.0", "phase": "0.0"}
    assert rows[1]["harmonic"] == "3"


def test_error_export_preserves_values(tmp_path) -> None:
    path = tmp_path / "error.csv"

    export_error(path, [0.0, 1.0], [0.25, -0.5])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[1]["error"] == "-0.5"


def test_export_rejects_mismatched_spectrum_shapes(tmp_path) -> None:
    with pytest.raises(ValueError, match="matching"):
        export_spectrum(tmp_path / "bad.csv", [1, 2], [1.0], [0.0])
