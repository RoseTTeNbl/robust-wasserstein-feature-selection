"""Algorithm of note/main.tex: the oracle Eq. (4) and outer approximation (Algorithm 1)."""

from .model import MODEL_VERSION, ZERO_SCORE_RISK, Cut, FitResult, Problem, SolveOptions
from .oracle import constant_fit, fit_fixed
from .selection import SearchResult, select

__all__ = [
    "MODEL_VERSION",
    "ZERO_SCORE_RISK",
    "Problem",
    "SolveOptions",
    "FitResult",
    "Cut",
    "SearchResult",
    "fit_fixed",
    "constant_fit",
    "select",
]
