"""Data model and cleanup helpers for user-drawn 2D curves."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


Point = tuple[float, float]
BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Curve2D:
    """A captured 2D curve with cleaned and centered representations."""

    raw_points: np.ndarray
    cleaned_points: np.ndarray
    normalized_points: np.ndarray
    bounding_box: BoundingBox | None
    is_closed: bool

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        *,
        close: bool = True,
    ) -> "Curve2D":
        """Create a curve from captured points without duplicating its closure."""
        raw_points = _as_points(points)
        cleaned_points = _clean_points(raw_points)
        bounding_box = _bounding_box(cleaned_points)
        normalized_points = _normalize_points(cleaned_points, bounding_box)
        return cls(
            raw_points=raw_points.copy(),
            cleaned_points=cleaned_points,
            normalized_points=normalized_points,
            bounding_box=bounding_box,
            is_closed=bool(close and cleaned_points.shape[0] >= 3),
        )

    @property
    def is_valid(self) -> bool:
        """Return whether the curve contains enough distinct points to close."""
        return self.is_closed and self.cleaned_points.shape[0] >= 3

    @property
    def x(self) -> np.ndarray:
        """Return cleaned x coordinates without a duplicate closing point."""
        return self.cleaned_points[:, 0].copy()

    @property
    def y(self) -> np.ndarray:
        """Return cleaned y coordinates without a duplicate closing point."""
        return self.cleaned_points[:, 1].copy()

    @property
    def closed_points(self) -> np.ndarray:
        """Return points with one logical closing segment for plotting."""
        if self.cleaned_points.size == 0:
            return self.cleaned_points.copy()
        return np.vstack((self.cleaned_points, self.cleaned_points[0]))

    @property
    def point_count(self) -> int:
        """Return the number of unique cleaned vertices."""
        return int(self.cleaned_points.shape[0])

    @property
    def width(self) -> float:
        """Return the curve bounding-box width."""
        if self.bounding_box is None:
            return 0.0
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def height(self) -> float:
        """Return the curve bounding-box height."""
        if self.bounding_box is None:
            return 0.0
        return self.bounding_box[3] - self.bounding_box[2]

    @property
    def perimeter(self) -> float:
        """Return the approximate closed polygonal path length."""
        if self.cleaned_points.shape[0] < 2:
            return 0.0
        segments = np.roll(self.cleaned_points, -1, axis=0) - self.cleaned_points
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    @property
    def estimated_area(self) -> float | None:
        """Return the absolute polygonal area estimate, if enough points exist."""
        if self.cleaned_points.shape[0] < 3:
            return None
        next_points = np.roll(self.cleaned_points, -1, axis=0)
        cross_sum = np.sum(
            self.cleaned_points[:, 0] * next_points[:, 1]
            - next_points[:, 0] * self.cleaned_points[:, 1]
        )
        return float(abs(cross_sum) / 2.0)


def _as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a point sequence to a two-column floating-point array."""
    try:
        values = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("curve points must be numeric pairs") from error
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("curve points must have shape (n, 2)")
    return values


def _clean_points(points: np.ndarray) -> np.ndarray:
    """Remove non-finite and consecutive duplicate points."""
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if finite_points.shape[0] < 2:
        return finite_points.copy()
    different = np.any(np.diff(finite_points, axis=0) != 0.0, axis=1)
    keep = np.concatenate(([True], different))
    return finite_points[keep].copy()


def _bounding_box(points: np.ndarray) -> BoundingBox | None:
    """Return ``(min_x, max_x, min_y, max_y)`` for cleaned points."""
    if points.size == 0:
        return None
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def _normalize_points(
    points: np.ndarray,
    bounding_box: BoundingBox | None,
) -> np.ndarray:
    """Center points and scale their largest span to two units."""
    if bounding_box is None:
        return np.empty((0, 2), dtype=float)
    min_x, max_x, min_y, max_y = bounding_box
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0.0:
        return np.zeros_like(points, dtype=float)
    return (points - center) * (2.0 / scale)
