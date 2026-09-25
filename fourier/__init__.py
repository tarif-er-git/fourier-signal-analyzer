"""Fourier Series analysis utilities."""

from .analysis import FourierAnalyzer
from .curve2d_analysis import (
    CurveFourierResult,
    CurveReconstruction,
    analyze_curve,
    calculate_curve_error,
    parameterize_curve,
    pointwise_curve_error,
    reconstruct_curve,
    resample_curve,
)
from .curve2d_error import (
    CurveConvergenceResult,
    CurveErrorMetrics,
    analyze_curve_convergence,
    evaluate_curve_error,
)
from .synthesis import (
    FourierSynthesizer,
    compute_fourier_series,
    compute_mse_vs_harmonics,
)
from .spectrum import FourierSpectrum

__all__ = [
    "CurveConvergenceResult",
    "CurveErrorMetrics",
    "CurveFourierResult",
    "CurveReconstruction",
    "FourierAnalyzer",
    "FourierSpectrum",
    "FourierSynthesizer",
    "analyze_curve",
    "analyze_curve_convergence",
    "calculate_curve_error",
    "compute_fourier_series",
    "compute_mse_vs_harmonics",
    "evaluate_curve_error",
    "parameterize_curve",
    "pointwise_curve_error",
    "reconstruct_curve",
    "resample_curve",
]
