"""Data, options and results. Equation numbers refer to note/main.tex.

The numerical instance is the standard hinge loss ell(t) = max(0, 1 + t), for
which Eq. (4) is a linear program. Scores may be restricted to [-1, 1]: clipping
a hinge score never increases ell(g) or ell(-g).
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

MODEL_VERSION = "hinge_ratio_cut_v5"
# V(empty) = 2 * ell(0): the zero score is always feasible with risk 2.
ZERO_SCORE_RISK = 2.0


def _frozen(values):
    array = np.array(values, dtype=float)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Problem:
    """Two labeled samples and the radii (rho_0, rho_1).

    Optional weights q0, q1 replace the uniform 1/n_k; rows of weight zero are dropped.
    """

    x0: np.ndarray
    x1: np.ndarray
    radii: np.ndarray
    q0: np.ndarray | None = None
    q1: np.ndarray | None = None

    def __post_init__(self):
        for k in range(2):
            x = np.atleast_2d(np.asarray(getattr(self, f"x{k}"), dtype=float))
            q = getattr(self, f"q{k}")
            q = np.full(len(x), 1 / len(x)) if q is None else np.asarray(q, dtype=float)
            keep = q > 0
            object.__setattr__(self, f"x{k}", _frozen(x[keep]))
            object.__setattr__(self, f"q{k}", _frozen(q[keep] / q[keep].sum()))
        radii = np.asarray(self.radii, dtype=float)
        if radii.shape != (2,) or not np.all(radii > 0):
            raise ValueError("radii must be two positive numbers")
        if self.x0.shape[1] != self.x1.shape[1]:
            raise ValueError("both classes need the same coordinates")
        object.__setattr__(self, "radii", _frozen(radii))

    @property
    def dimension(self):
        return self.x0.shape[1]

    @property
    def size(self):
        return len(self.x0) + len(self.x1)

    @cached_property
    def x(self):
        """Pooled samples: they are both the transport sources and the targets X."""
        return _frozen(np.vstack((self.x0, self.x1)))

    @cached_property
    def y(self):
        labels = np.r_[np.zeros(len(self.x0), int), np.ones(len(self.x1), int)]
        labels.setflags(write=False)
        return labels

    @cached_property
    def q(self):
        """Source masses 1/n_k (or the given weights); each class sums to one."""
        return _frozen(np.r_[self.q0, self.q1])

    @cached_property
    def delta(self):
        """delta[i, r, m] = |x_i[m] - x_r[m]|, so c_k(gamma)[m] = sum gamma_ir delta[i, r, m]."""
        return _frozen(np.abs(self.x[:, None, :] - self.x[None, :, :]))

    def distance(self, support):
        """||x_i[S] - x_r[S]||_1 for all sources i and targets r."""
        return self.delta[:, :, list(support)].sum(axis=2)

    def support(self, support):
        values = tuple(int(j) for j in support)
        if len(set(values)) != len(values) or any(not 0 <= j < self.dimension for j in values):
            raise ValueError("support must list distinct coordinate indices")
        return tuple(sorted(values))

    def mask(self, support):
        z = np.zeros(self.dimension)
        z[list(self.support(support))] = 1
        return z


@dataclass(frozen=True)
class SolveOptions:
    time_limit: float = 300
    fit_time_limit: float = 60
    absolute_tolerance: float = 1e-3  # stop Algorithm 1 when U - L <= this
    inner_tolerance: float = 1e-4  # a fit is "optimal" when upper - lower <= this
    constraint_tolerance: float = 1e-7
    threads: int = 1
    swap_limit: int = 10
    swap_improvement: float = 1e-4


@dataclass(frozen=True)
class Cut:
    """Ratio cut Eq. (8): 1/F(z) <= intercept + slope @ z."""

    intercept: float
    slope: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "slope", _frozen(self.slope))

    def evaluate(self, z):
        """The lower bound F(z) >= 1 / (intercept + slope @ z)."""
        return 1.0 / (self.intercept + float(self.slope @ np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class FitResult:
    """Eq. (4) at S: scores g, source bounds a, prices lambda; lower <= V(S) <= upper."""

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
        for name in ("scores", "source_bounds", "prices"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @property
    def gap(self):
        return max(0.0, self.upper - self.lower)

    def predict_score(self, x):
        """The detector f(x) = g(x[S]), extended off the samples as in Eq. (5)."""
        query = np.atleast_2d(np.asarray(x, dtype=float))[:, list(self.support)]
        source = self.problem.x[:, list(self.support)]
        result = np.empty(len(query))
        for start in range(0, len(query), 2048):
            xi = query[start : start + 2048]
            distance = np.abs(source[:, None, :] - xi[None, :, :]).sum(axis=2)
            # Anchors i, j: smallest minimizers of E_k(xi) = min_i a_i + lambda_k ||x_i[S] - xi||.
            anchor = []
            for k in range(2):
                rows = np.flatnonzero(self.problem.y == k)
                envelope = self.source_bounds[rows, None] + self.prices[k] * distance[rows]
                anchor.append(rows[np.argmin(envelope, axis=0)])
            i, j = anchor
            columns = np.arange(len(xi))
            r0, r1 = distance[i, columns], distance[j, columns]
            # Weigh the anchors by distance; r0 + r1 = 0 only at a sample, overwritten below.
            tau = np.divide(r0, r0 + r1, out=np.zeros_like(r0), where=r0 + r1 > 0)
            score = (1 - tau) * self.scores[i] + tau * self.scores[j]
            # At a projected sample the fitted score is used (coincident samples share it).
            hit = distance == 0
            seen = hit.any(axis=0)
            score[seen] = self.scores[np.argmax(hit[:, seen], axis=0)]
            result[start : start + len(xi)] = score
        return np.clip(result, -1, 1)
