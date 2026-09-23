"""Accuracy metrics for sampled signals."""

from .convergence import ConvergenceResult, analyze_convergence
from .error import (
    calculate_error,
    calculate_mae,
    calculate_max_error,
    calculate_mse,
    calculate_rmse,
)
from .gibbs import (
    GibbsResult,
    analyze_gibbs,
    detect_discontinuity,
    theoretical_gibbs_overshoot,
)

__all__ = [
    "ConvergenceResult",
    "GibbsResult",
    "analyze_convergence",
    "analyze_gibbs",
    "calculate_error",
    "calculate_mae",
    "calculate_max_error",
    "calculate_mse",
    "calculate_rmse",
    "detect_discontinuity",
    "theoretical_gibbs_overshoot",
]
