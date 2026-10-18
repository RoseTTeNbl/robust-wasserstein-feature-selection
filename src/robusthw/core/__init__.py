"""Mathematical core: robust fitting, score recovery, and affine certificates."""

from .model import Cut, FitResult, Problem, SolveOptions
from .oracle import constant_fit, fit_fixed

__all__ = ["Problem", "SolveOptions", "FitResult", "Cut", "fit_fixed", "constant_fit"]
