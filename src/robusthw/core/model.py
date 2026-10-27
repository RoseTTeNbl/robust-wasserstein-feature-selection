"""Data, controls and certificates for the standard-hinge specialization."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

MODEL_VERSION = "full_support_standard_hinge_cone_v3"
ZERO_SCORE_RISK = 2.0


def _readonly(value, ndim, name):
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _class_data(samples, weights, name):
    samples = _readonly(samples, 2, name)
    if min(samples.shape) == 0:
        raise ValueError("Both classes and the coordinate dimension must be nonempty")
    # Class-wise empirical masses: sum_i q_i^(k)=1; default q_i^(k)=1/n_k.
    # Unequal class sizes n_0 and n_1 are allowed.
    weights = _readonly(
        np.full(len(samples), 1 / len(samples)) if weights is None else weights,
        1,
        f"{name} weights",
    )
    if weights.shape != (len(samples),) or np.any(weights < 0):
        raise ValueError(f"{name} weights must have one nonnegative entry per row")
    if not np.isclose(weights.sum(), 1, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} weights must sum to one")
    positive = weights > 0
    return (
        _readonly(samples[positive], 2, name),
        _readonly(weights[positive] / weights[positive].sum(), 1, f"{name} weights"),
    )


@dataclass(frozen=True)
class Problem:
    """Two labeled sample matrices and two positive Wasserstein radii.

    Repeated observations remain separate atoms. Zero-mass rows are discarded;
    positive masses are normalized only within floating-point input tolerance.
    """

    x0: np.ndarray
    x1: np.ndarray
    radii: np.ndarray
    q0: np.ndarray | None = None
    q1: np.ndarray | None = None

    def __post_init__(self):
        for k in range(2):
            samples, weights = _class_data(getattr(self, f"x{k}"), getattr(self, f"q{k}"), f"x{k}")
            object.__setattr__(self, f"x{k}", samples)
            object.__setattr__(self, f"q{k}", weights)
        if self.x0.shape[1] != self.x1.shape[1]:
            raise ValueError("Both classes must have the same coordinate dimension")
        radii = _readonly(self.radii, 1, "radii")
        if radii.shape != (2,) or np.any(radii <= 0):
            raise ValueError("radii must contain two strictly positive values")
        object.__setattr__(self, "radii", radii)

    @property
    def dimension(self):
        return self.x0.shape[1]

    @property
    def size(self):
        return len(self.x0) + len(self.x1)

    @cached_property
    def x(self):
        """Internal target labels: n0 rows followed by n1 rows, not equal classes."""
        # Storage order: class 0 then class 1; N=n_0+n_1.
        # The same rows index the full target support: target r is x[r].
        return _readonly(np.vstack((self.x0, self.x1)), 2, "x")

    @cached_property
    def y(self):
        # Source class labels y[i] in {0,1}; distinct from the paper's target y in X.
        array = np.r_[np.zeros(len(self.x0), dtype=int), np.ones(len(self.x1), dtype=int)]
        array.setflags(write=False)
        return array

    @cached_property
    def q(self):
        # sum_i q[i]=2: one unit of mass per class, matching the sum of two risks.
        return _readonly(np.r_[self.q0, self.q1], 1, "q")

    @cached_property
    def delta(self):
        """Coordinatewise distances Delta[i,r,j], with O(N²D) storage."""
        # Axes: source i, target r, coordinate m; delta[i,r,m]=|x_i[m]-x_r[m]|.
        # Selected transport cost: sum_{m in S} delta[i,r,m]=||x_i[S]-x_r[S]||_1.
        with np.errstate(over="ignore"):
            delta = np.abs(self.x[:, None, :] - self.x[None, :, :])
        return _readonly(delta, 3, "coordinate distances (rescale overflowing data)")

    @cached_property
    def price_bounds(self):
        """Conservative cut-slope scale; this is not an oracle constraint."""
        bounds = []
        for k in range(2):
            rows = self.y == k
            bounds.append(self.q[rows] @ self.delta[rows].max(axis=1) / self.radii[k])
        return _readonly(2 * np.max(bounds, axis=0), 1, "price bounds")

    def support(self, support):
        values = tuple(support)
        if any(
            isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) for j in values
        ):
            raise ValueError("support must contain integer coordinate indices")
        if len(set(values)) != len(values) or any(j < 0 or j >= self.dimension for j in values):
            raise ValueError("support contains repeated or out-of-range coordinates")
        return tuple(sorted(int(j) for j in values))

    def mask(self, support):
        # Binary selection z=1_S; code indices are zero-based.
        mask = np.zeros(self.dimension, dtype=float)
        mask[list(self.support(support))] = 1
        return mask

    def relaxation(self, z):
        """Validate a continuous selection in the domain of the value function."""
        z = _readonly(z, 1, "z")
        if z.shape != (self.dimension,) or np.any((z < 0) | (z > 1)):
            raise ValueError("z must have one value in [0, 1] per coordinate")
        return z


@dataclass(frozen=True)
class SolveOptions:
    time_limit: float = 300
    fit_time_limit: float = 60
    absolute_tolerance: float = 1e-3
    inner_tolerance: float = 1e-4
    constraint_tolerance: float = 1e-7
    threads: int = 1
    swap_limit: int = 10
    swap_improvement: float = 1e-4

    def __post_init__(self):
        for name in (
            "time_limit",
            "fit_time_limit",
            "absolute_tolerance",
            "inner_tolerance",
            "constraint_tolerance",
            "swap_improvement",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or ("tolerance" in name and value == 0):
                raise ValueError(f"{name} must be finite and nonnegative (tolerances positive)")
        for name in ("threads", "swap_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.threads < 1:
            raise ValueError("threads must be positive")


@dataclass(frozen=True)
class Cut:
    """The numerical global lower bound intercept + slope @ z."""

    intercept: float
    slope: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.intercept):
            raise ValueError("cut intercept must be finite")
        object.__setattr__(self, "slope", _readonly(self.slope, 1, "cut slope"))

    def evaluate(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != self.slope.shape or not np.isfinite(z).all():
            raise ValueError("cut evaluation requires a finite coordinate vector")
        # Eq. (7): c_s(z)=2*sqrt(H(gamma^s))-M(gamma^s)-(mu^s)^T*z <= F(z).
        # Stored intercept includes the numerical margin; slope=-mu^s.
        return float(self.intercept + self.slope @ z)


@dataclass(frozen=True)
class FitResult:
    problem: Problem = field(repr=False)
    support: tuple[int, ...]
    scores: np.ndarray
    source_bounds: np.ndarray
    prices: np.ndarray
    upper: float
    lower: float
    status: str
    runtime: float
    cut: Cut | None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "support", self.problem.support(self.support))
        for name, size in (
            ("scores", self.problem.size),
            ("source_bounds", self.problem.size),
            ("prices", 2),
        ):
            value = _readonly(getattr(self, name), 1, name)
            if value.shape != (size,):
                raise ValueError(f"{name} has the wrong dimension")
            object.__setattr__(self, name, value)
        if (
            np.any(np.abs(self.scores) > 1)
            or np.any(self.source_bounds < 0)
            or np.any(self.prices < 0)
        ):
            raise ValueError("finite score certificate violates variable bounds")

    @property
    def gap(self):
        # Fixed-selection interval: lower <= V(S)=F(1_S) <= upper.
        # This gap concerns the inner solve, not the search over S.
        return max(0.0, self.upper - self.lower)

    def predict_score(self, x):
        """Recover the detector g from Appendix A's distance interpolation."""
        queries = np.atleast_2d(np.asarray(x, dtype=float))
        if (
            queries.ndim != 2
            or queries.shape[1] != self.problem.dimension
            or not np.isfinite(queries).all()
        ):
            raise ValueError(
                "Prediction rows must be finite with the training coordinate dimension"
            )
        result = np.empty(len(queries))
        selected = list(self.support)
        # Appendix A: each target row is the projected query xi=x[S].
        source = self.problem.x[:, selected]
        class_rows = [np.flatnonzero(self.problem.y == k) for k in range(2)]
        for start in range(0, len(queries), 2048):
            target = queries[start : start + 2048, selected]
            with np.errstate(over="ignore"):
                distances = np.abs(source[:, None, :] - target[None, :, :]).sum(axis=2)
            if not np.isfinite(distances).all():
                raise ValueError("Prediction distances overflow; rescale the query rows")
            anchors = []
            for k, rows in enumerate(class_rows):
                # E_k(xi)=min_i {a_i^(k)+lambda_k*||x_i^(k)[S]-xi||_1}.
                # Anchor i_k minimizes the full envelope, including a_i^(k).
                envelope = self.source_bounds[rows, None] + self.prices[k] * distances[rows]
                anchors.append(rows[np.argmin(envelope, axis=0)])
            i0, i1 = anchors
            # r_0=||x_{i_0}^(0)[S]-xi||_1; b=distance between the two anchors.
            # Detector extension: tau=min(1,r_0/b), with tau=0 for coincident anchors.
            r0 = distances[i0, np.arange(len(target))]
            anchor_distance = np.abs(source[i0] - source[i1]).sum(axis=1)
            tau = np.minimum(
                1.0,
                np.divide(
                    r0, anchor_distance, out=np.zeros(len(target)), where=anchor_distance > 0
                ),
            )
            # g(xi)=(1-tau)*g_{x_{i_0}}+tau*g_{x_{i_1}}.
            # Loss envelopes: ell(-g(xi))<=E_0(xi), ell(g(xi))<=E_1(xi).
            result[start : start + len(target)] = (1 - tau) * self.scores[i0] + tau * self.scores[
                i1
            ]
            # Preserve fitted scores at observed projections; duplicates share one score.
            matches = distances == 0
            observed = matches.any(axis=0)
            indices = np.flatnonzero(observed)
            result[start + indices] = self.scores[np.argmax(matches[:, observed], axis=0)]
        return np.clip(result, -1, 1)


@dataclass(frozen=True)
class RelaxedResult:
    """A primal/dual evaluation of F(z), not a detector for fractional z.

    ``coordinate_weights[k, m]`` is w_km in the epigraph formulation.
    Diagnostics retain scaled plans, their dependent mass alpha=M(gamma),
    and coordinate multipliers.
    """

    problem: Problem = field(repr=False)
    z: np.ndarray
    scores: np.ndarray
    source_bounds: np.ndarray
    prices: np.ndarray
    coordinate_weights: np.ndarray
    upper: float
    lower: float
    status: str
    runtime: float
    cut: Cut | None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "z", self.problem.relaxation(self.z))
        for name, shape in (
            ("scores", (self.problem.size,)),
            ("source_bounds", (self.problem.size,)),
            ("prices", (2,)),
            ("coordinate_weights", (2, self.problem.dimension)),
        ):
            value = _readonly(getattr(self, name), len(shape), name)
            if value.shape != shape:
                raise ValueError(f"{name} has the wrong dimension")
            object.__setattr__(self, name, value)

    @property
    def gap(self):
        return max(0.0, self.upper - self.lower)
