"""Tests for 2D curve persistence and numerical exports."""

import csv
import json

import numpy as np
import pytest

from fourier.curve2d_analysis import analyze_curve, calculate_curve_error
from signal.curve2d import Curve2D
from signal_io import (
    export_curve_coefficients,
    export_curve_csv,
    export_curve_reconstruction,
    load_curve,
    save_curve,
)


@pytest.fixture
def square() -> Curve2D:
    return Curve2D.from_points(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    )


def test_curve_geometry_statistics(square: Curve2D) -> None:
    assert square.bounding_box == (0.0, 2.0, 0.0, 2.0)
    assert square.width == pytest.approx(2.0)
    assert square.height == pytest.approx(2.0)
    assert square.perimeter == pytest.approx(8.0)
    assert square.estimated_area == pytest.approx(4.0)


def test_save_and_load_curve_preserves_points_and_metadata(tmp_path, square: Curve2D) -> None:
    path = tmp_path / "curve.json"
    save_curve(path, square, {"name": "square"})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["type"] == "2d_closed_curve"
    assert payload["version"] == 1
    assert payload["closed"] is True
    assert payload["metadata"]["name"] == "square"
    loaded = load_curve(path)
    np.testing.assert_allclose(loaded.curve.raw_points, square.raw_points)
    np.testing.assert_allclose(loaded.curve.cleaned_points, square.cleaned_points)
    np.testing.assert_allclose(loaded.curve.normalized_points, square.normalized_points)


def test_invalid_and_wrong_type_curve_files_are_rejected(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        load_curve(malformed)
    wrong_type = tmp_path / "wrong.json"
    wrong_type.write_text(
        json.dumps({
            "format": "signal-sketch-and-decompose.curve2d.v1",
            "type": "signal",
            "version": 1,
            "raw_points": [[0, 0], [1, 0], [0, 1]],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a 2D"):
        load_curve(wrong_type)


def test_curve_csv_export_schema(tmp_path, square: Curve2D) -> None:
    path = tmp_path / "curve.csv"
    export_curve_csv(path, square)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows[0].keys() == {"index", "x", "y"}
    assert len(rows) == square.point_count
    assert rows[2]["x"] == "2.0"


def test_reconstruction_and_coefficient_exports_use_current_data(tmp_path, square: Curve2D) -> None:
    coefficients = analyze_curve(square, num_samples=64, maximum_harmonic=8)
    reconstruction = calculate_curve_error(coefficients, harmonic_count=3)
    reconstruction_path = tmp_path / "reconstruction.csv"
    coefficients_path = tmp_path / "coefficients.csv"

    export_curve_reconstruction(reconstruction_path, coefficients, reconstruction)
    export_curve_coefficients(coefficients_path, coefficients)
    reconstruction_rows = list(csv.DictReader(reconstruction_path.open(newline="", encoding="utf-8")))
    coefficient_rows = list(csv.DictReader(coefficients_path.open(newline="", encoding="utf-8")))

    assert reconstruction_rows[0].keys() == {
        "index", "original_x", "original_y", "reconstructed_x", "reconstructed_y", "error"
    }
    assert len(reconstruction_rows) == 64
    assert coefficient_rows[0].keys() == {
        "k", "Cx_real", "Cx_imag", "Cx_magnitude", "Cx_phase",
        "Cy_real", "Cy_imag", "Cy_magnitude", "Cy_phase",
    }
    assert len(coefficient_rows) == len(coefficients.harmonics)
