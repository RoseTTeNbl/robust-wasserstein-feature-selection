"""Problem data and certificates for exact-budget robust feature selection.

The Wasserstein radii stay fixed as features are selected.  Each positive-mass
observation remains a separate atom, including repeated observations.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from functools import cached_property
from typing import Any

import numpy as np


def _array(value, *, ndim, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


def _weights(value, n, name):
    result = _array(np.full(n, 1 / n) if value is None else value,
                    ndim=1, name=name)
    if result.shape != (n,) or np.any(result < 0):
        raise ValueError(f"{name} must have {n} nonnegative entries")
    if not np.isclose(result.sum(), 1.0, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    result = result / result.sum()
    result.setflags(write=False)
    return result


def json_value(value):
    """Convert certificates to JSON data without discarding numerical fields."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_dict"):
        return json_value(value.as_dict())
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    raise TypeError(f"Cannot serialize certificate of type {type(value).__name__}")


@dataclass(frozen=True)
class Problem:
    x0: np.ndarray
    x1: np.ndarray
    budget: int
    theta0: float
    theta1: float
    loss: str = "overlap"
    q0: np.ndarray | None = None
    q1: np.ndarray | None = None

    def __post_init__(self):
        x0 = _array(self.x0, ndim=2, name="x0")
        x1 = _array(self.x1, ndim=2, name="x1")
        if min(x0.shape) == 0 or min(x1.shape) == 0:
            raise ValueError("Both classes and the feature dimension must be nonempty")
        if x0.shape[1] != x1.shape[1]:
            raise ValueError("Both classes must have the same feature dimension")
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, np.integer)):
            raise ValueError("budget must be an integer")
        if not 0 <= self.budget <= x0.shape[1]:
            raise ValueError("budget must lie between zero and the feature dimension")
        if self.loss not in ("logistic", "overlap"):
            raise ValueError("loss must be 'logistic' or 'overlap'")
        for name in ("theta0", "theta1"):
            radius = float(getattr(self, name))
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, radius)
        object.__setattr__(self, "budget", int(self.budget))
        for k, samples in enumerate((x0, x1)):
            weights = _weights(getattr(self, f"q{k}"), len(samples), f"q{k}")
            positive = weights > 0
            samples, weights = samples[positive], weights[positive]
            samples.setflags(write=False)
            weights.setflags(write=False)
            object.__setattr__(self, f"x{k}", samples)
            object.__setattr__(self, f"q{k}", weights)

    @property
    def p(self):
        return self.x0.shape[1]

    @property
    def n0(self):
        return len(self.x0)

    @property
    def n1(self):
        return len(self.x1)

    @property
    def n(self):
        return self.n0 + self.n1

    @property
    def B(self):
        """Risk of the uninformative constant test; a universal upper bound."""
        return 1.0 if self.loss == "overlap" else float(2 * np.log(2))

    @property
    def U(self):
        return self.B / np.array([self.theta0, self.theta1])

    @property
    def theta(self):
        return np.array([self.theta0, self.theta1])

    @property
    def samples(self):
        return self.x0, self.x1

    @property
    def weights(self):
        return self.q0, self.q1

    @cached_property
    def points(self):
        points = np.vstack((self.x0, self.x1))
        points.setflags(write=False)
        return points

    @cached_property
    def delta0(self):
        delta = np.abs(self.x0[:, None, :] - self.points[None, :, :])
        if not np.isfinite(delta).all():
            raise ValueError("Coordinate differences overflow; rescale the input data")
        delta.setflags(write=False)
        return delta

    @cached_property
    def delta1(self):
        delta = np.abs(self.x1[:, None, :] - self.points[None, :, :])
        if not np.isfinite(delta).all():
            raise ValueError("Coordinate differences overflow; rescale the input data")
        delta.setflags(write=False)
        return delta

    @property
    def deltas(self):
        return self.delta0, self.delta1

    def mask(self, support):
        indices = tuple(support)
        if any(isinstance(j, bool) or not isinstance(j, (int, np.integer))
               for j in indices):
            raise ValueError("support must contain integer feature indices")
        if len(set(indices)) != len(indices) or any(j < 0 or j >= self.p for j in indices):
            raise ValueError("support contains duplicate or out-of-range indices")
        mask = np.zeros(self.p, dtype=int)
        mask[list(indices)] = 1
        return mask

    def support(self, mask):
        mask = np.asarray(mask)
        if mask.shape != (self.p,) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"mask must contain exactly {self.p} binary entries")
        return tuple(int(j) for j in np.flatnonzero(mask))


@dataclass(frozen=True)
class Cut:
    """An oracle-certified global affine lower bound on the risk."""

    intercept: float
    coefficients: np.ndarray

    def __post_init__(self):
        intercept = float(self.intercept)
        if not np.isfinite(intercept):
            raise ValueError("cut intercept must be finite")
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "coefficients",
                           _array(self.coefficients, ndim=1, name="cut coefficients"))

    def value(self, mask):
        return float(self.intercept + self.coefficients @ np.asarray(mask))

    def to_dict(self):
        return json_value(self)


@dataclass
class OracleResult:
    value: float
    upper: float
    gradient: np.ndarray
    intercept: float
    test: Any = None
    status: str = "optimal"
    seconds: float = 0.0
    certificate: dict = field(default_factory=dict)

    @property
    def lower(self):
        return max(0.0, self.value)

    @property
    def cut(self):
        return Cut(self.intercept, self.gradient)

    @property
    def support(self):
        return self.test.support if self.test is not None else ()

    @property
    def diagnostics(self):
        return self.certificate

    def to_dict(self):
        return {name: json_value(getattr(self, name)) for name in self.__dataclass_fields__}
