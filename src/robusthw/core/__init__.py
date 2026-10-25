"""Mathematical core: robust fitting, score recovery, and affine certificates."""

from .model import (
    MODEL_VERSION,
    ZERO_SCORE_RISK,
    Cut,
    FitResult,
    Problem,
    RelaxedResult,
    SolveOptions,
)
from .oracle import constant_fit, fit_fixed, fit_relaxed

__all__ = [
    "MODEL_VERSION",
    "ZERO_SCORE_RISK",
    "Problem",
    "SolveOptions",
    "FitResult",
    "RelaxedResult",
    "Cut",
    "fit_fixed",
    "fit_relaxed",
    "constant_fit",
]
