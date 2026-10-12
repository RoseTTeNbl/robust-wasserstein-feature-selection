"""Robust hypothesis testing with certified global-cut feature selection."""

from .oracle import FixedSupportOracle, GateOracle, RobustTest
from .problem import Cut, OracleResult, Problem
from .solver import SelectionResult, enumerate_supports, solve

__version__ = "0.7.0"

SolveResult = SelectionResult

__all__ = ["Cut", "OracleResult", "Problem", "SelectionResult", "SolveResult", "GateOracle", "FixedSupportOracle",
           "RobustTest", "enumerate_supports", "solve"]
