"""The predeclared experiment protocol; no experiment runs on import."""

from .data import DataSplit, original_anchor_split, synthetic_split
from .protocol import ExperimentSpec, build_plan

__all__ = ["DataSplit", "ExperimentSpec", "build_plan", "original_anchor_split", "synthetic_split"]
