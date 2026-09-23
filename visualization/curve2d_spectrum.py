"""Spectrum data helpers for analyzed 2D Fourier curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fourier.curve2d_analysis import CurveFourierResult


@dataclass(frozen=True)
class Curve2DSpectrumRow:
    """Display-ready magnitudes and phases for one signed harmonic."""

    harmonic: int
    x_magnitude: float
    x_phase: float
    y_magnitude: float
    y_phase: float
    combined_magnitude: float


def build_spectrum_rows(
    coefficients: CurveFourierResult,
) -> tuple[Curve2DSpectrumRow, ...]:
    """Build rows directly from existing X/Y Fourier coefficients."""
    if not isinstance(coefficients, CurveFourierResult):
        raise ValueError("coefficients must be a CurveFourierResult")
    if coefficients.harmonics.size == 0:
        raise ValueError("coefficient data cannot be empty")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            coefficients.harmonics,
            coefficients.x_coefficients,
            coefficients.y_coefficients,
        )
    ):
        raise ValueError("coefficient data must be finite")
    return tuple(
        Curve2DSpectrumRow(
            harmonic=int(harmonic),
            x_magnitude=float(x_mag),
            x_phase=float(x_ph),
            y_magnitude=float(y_mag),
            y_phase=float(y_ph),
            combined_magnitude=float(np.hypot(x_mag, y_mag)),
        )
        for harmonic, x_mag, x_ph, y_mag, y_ph in zip(
            coefficients.harmonics,
            coefficients.x_magnitudes,
            coefficients.x_phases,
            coefficients.y_magnitudes,
            coefficients.y_phases,
        )
    )


def sort_spectrum_rows(
    rows: tuple[Curve2DSpectrumRow, ...] | list[Curve2DSpectrumRow],
    sort_by: str = "harmonic",
) -> tuple[Curve2DSpectrumRow, ...]:
    """Sort only the display rows by harmonic index or combined magnitude."""
    if sort_by == "harmonic":
        key = lambda row: row.harmonic
    elif sort_by == "combined_magnitude":
        key = lambda row: (-row.combined_magnitude, row.harmonic)
    else:
        raise ValueError("sort_by must be 'harmonic' or 'combined_magnitude'")
    return tuple(sorted(rows, key=key))


def dominant_harmonics(
    rows: tuple[Curve2DSpectrumRow, ...] | list[Curve2DSpectrumRow],
    *,
    count: int = 3,
) -> tuple[Curve2DSpectrumRow, ...]:
    """Return rows with the largest combined coefficient magnitudes."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    return sort_spectrum_rows(rows, "combined_magnitude")[:count]
