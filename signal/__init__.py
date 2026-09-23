"""Signal-generation, sampling, and preprocessing utilities."""

from .curve2d import Curve2D
from .generator import sawtooth_wave, sine_wave, square_wave, triangle_wave
from .preprocessor import (
	normalize_signal,
	preprocess_signal,
	remove_nonfinite_samples,
	validate_signal_arrays,
)
from .sampler import create_time_grid, prepare_signal_points, resample_signal

__all__ = [
	"Curve2D",
	"create_time_grid",
	"normalize_signal",
	"prepare_signal_points",
	"preprocess_signal",
	"remove_nonfinite_samples",
	"resample_signal",
	"sawtooth_wave",
	"sine_wave",
	"square_wave",
	"triangle_wave",
	"validate_signal_arrays",
]
