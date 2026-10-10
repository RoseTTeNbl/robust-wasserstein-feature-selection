"""Completed binary tests, owned cuts, and a fixed-kernel candidate forecast.

Forecast means and standard deviations only rank candidate evaluations.
They are never risk certificates or fitted candidate tests.
"""
from math import comb
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular


class FixedKernelPredictor:
    """Gaussian conditioning with a fixed Hamming kernel and appended factors."""

    model_tag = "fixed_hamming_gp_v1"

    def __init__(self, problem, anchor_target):
        self.dimension = problem.p
        self.risk_scale = float(problem.B)
        if self.risk_scale <= 0:
            raise ValueError("the forecast requires a positive risk scale")
        self.length_scale = float(max(1, problem.budget))
        self.noise_variance = max(1e-8, (anchor_target / self.risk_scale)**2)
        self.points = np.empty((0, self.dimension))
        self.responses = np.empty(0)
        self.factor = np.empty((0, 0))
        self.alpha = np.empty(0)
        self.factor_updates = self.prediction_batches = self.predicted_points = 0
        self.factor_reuses = 0
        self.update_seconds = self.prediction_seconds = 0.

    def _points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.all(
                (points == 0) | (points == 1)):
            raise ValueError("forecast inputs must be binary feature vectors")
        return points

    def kernel(self, left, right):
        left, right = self._points(left), self._points(right)
        hamming = np.abs(left[:, None, :]-right[None, :, :]).sum(axis=2)
        return .25 * np.exp(-hamming / self.length_scale)

    def append(self, point, actual_risk):
        """Append one completed evaluation without rebuilding the covariance."""
        started = perf_counter()
        point = self._points(point)
        if len(point) != 1:
            raise ValueError("append expects one completed evaluation")
        actual_risk = float(actual_risk)
        if not np.isfinite(actual_risk) or not 0 <= actual_risk <= self.risk_scale:
            raise ValueError("forecast responses must be actual risk bounds in [0, B]")
        n = len(self.responses)
        factor = np.zeros((n+1, n+1))
        if n:
            cross = self.kernel(self.points, point)[:, 0]
            row = solve_triangular(self.factor, cross, lower=True, check_finite=False)
            factor[:n, :n] = self.factor
            factor[n, :n] = row
            schur = .25 + self.noise_variance-float(row @ row)
        else:
            schur = .25 + self.noise_variance
        # The latent conditional variance is nonnegative, so the exact Schur
        # complement is at least the fixed nugget; clamp only rounding error.
        factor[n, n] = np.sqrt(max(self.noise_variance, schur))
        points = np.vstack((self.points, point))
        responses = np.append(self.responses, actual_risk)
        centered = responses / self.risk_scale-.5
        intermediate = solve_triangular(factor, centered, lower=True, check_finite=False)
        alpha = solve_triangular(factor.T, intermediate, lower=False, check_finite=False)
        self.points, self.responses, self.factor, self.alpha = points, responses, factor, alpha
        self.factor_updates += 1
        self.update_seconds += perf_counter()-started

    def predict(self, points):
        """Return batched mean and latent standard deviation in risk units."""
        started = perf_counter()
        points = self._points(points)
        if len(self.responses):
            cross = self.kernel(self.points, points)
            projected = solve_triangular(self.factor, cross, lower=True, check_finite=False)
            mean = self.risk_scale * (.5 + cross.T @ self.alpha)
            variance = np.maximum(0., .25-np.square(projected).sum(axis=0))
            sigma = self.risk_scale * np.sqrt(variance)
            self.factor_reuses += 1
        else:
            mean = np.full(len(points), .5 * self.risk_scale)
            sigma = np.full(len(points), .5 * self.risk_scale)
        self.prediction_batches += 1
        self.predicted_points += len(points)
        self.prediction_seconds += perf_counter()-started
        return mean, sigma


class EvaluationHistory:
    def __init__(self, problem, *, anchor_target=1e-8):
        self.problem = problem
        self.predictor = FixedKernelPredictor(problem, anchor_target)
        self.cuts, self.points, self.anchor_gaps = [], [], []
        self.tests, self.records, self.uppers = {}, [], []
        self.accuracy_flags = []
        movement = np.array([q @ delta.max(axis=1)
                             for q, delta in zip(problem.weights, problem.deltas)])
        self.metric_weights = (movement / problem.theta[:, None]).max(axis=0)

    def add(self, z, *, test, intercept, gradient, accurate):
        """Store exactly one owned tangent and actual test for this evaluation."""
        z = np.asarray(z, dtype=float)
        support = self.problem.support(z)
        if len(support) != self.problem.budget:
            raise ValueError("history requires a full-budget binary evaluation")
        if support in self.tests:
            raise ValueError("a completed support cannot be evaluated twice")
        if test is None or tuple(test.support) != support:
            raise ValueError("the retained test must belong to the evaluated support")
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != (self.problem.p,) or not np.isfinite(gradient).all():
            raise ValueError("invalid owned tangent gradient")
        if not np.isfinite(intercept) or not np.isfinite(test.upper):
            raise ValueError("evaluation bounds must be finite")
        own_lower = float(intercept + gradient @ z)
        gap = float(test.upper - own_lower)
        if gap < -1e-7 * max(1., self.problem.B):
            raise ArithmeticError("owned tangent exceeds the actual feasible test")
        self.cuts.append((float(intercept), gradient.copy()))
        self.points.append(z.copy())
        self.tests[support] = test
        self.uppers.append(float(test.upper))
        self.anchor_gaps.append(max(0., gap))
        self.accuracy_flags.append(bool(accurate))
        if accurate:
            self.predictor.append(z, test.upper)
        self.records.append(dict(support=list(support), point=z.tolist(),
                                 intercept=float(intercept), gradient=gradient.tolist(),
                                 anchor_lower=own_lower, actual_upper=float(test.upper),
                                 anchor_gap=max(0., gap), accurate=bool(accurate)))

    def predict(self, points):
        return self.predictor.predict(points)

    def lower(self, z):
        return float(max([0., *[intercept + float(gradient @ z)
                               for intercept, gradient in self.cuts]]))

    def distance(self, z, other):
        return float(self.metric_weights @ np.abs(np.asarray(z) - np.asarray(other)))

    def scalar_upper(self, z):
        """Conservative risk bound, never a feasible-test construction."""
        return float(min([self.problem.B, *[
            upper + self.problem.B * self.distance(z, point)
            for point, upper in zip(self.points, self.uppers)
        ]]))

    def bounds(self, z):
        lower, upper = self.lower(z), self.scalar_upper(z)
        if lower > upper + 1e-7 * max(1., self.problem.B):
            raise ArithmeticError("history lower bound exceeds geometric risk bound")
        return lower, upper, max(0., upper - lower)

    @property
    def accurate_count(self):
        return sum(self.accuracy_flags)

    def quantitative_bound(self, epsilon, delta):
        """Data-dependent bound for the risk-gate screen a <= U-epsilon/2."""
        radius = (epsilon / 2 - delta) / self.problem.B
        if radius <= 0:
            raise ValueError("delta must be smaller than half the outer tolerance")
        weights = np.sort(self.metric_weights)[::-1]
        tails = np.r_[np.cumsum(weights[::-1])[::-1], 0.]
        dimension = int(np.flatnonzero(tails < radius)[0])
        patterns = sum(comb(dimension, i)
                       for i in range(min(self.problem.budget, dimension) + 1))
        return dict(separation_radius=float(radius),
                    precision_coordinates=dimension,
                    accurate_evaluation_bound=min(comb(self.problem.p, self.problem.budget),
                                                  patterns),
                    count_assumptions_met=all(self.accuracy_flags))


def proposal_pool(problem, history, incumbent, *, count, rng, guard=None, remaining=None):
    """Bounded active-cut proposals, swaps, and random diversity."""
    supports = {tuple(incumbent)}
    if guard is not None:
        supports.add(tuple(guard))
    seeds = list(supports)
    for _, gradient in history.cuts[-4:]:
        supports.add(tuple(sorted(np.argsort(gradient, kind="stable")[:problem.budget])))
    for support in seeds:
        if remaining is not None and remaining() <= 0:
            raise TimeoutError("wall-time budget exhausted during proposal construction")
        outside = [j for j in range(problem.p) if j not in support]
        z = problem.mask(support)
        active = max(history.cuts, key=lambda cut: cut[0]+float(cut[1] @ z),
                     default=(0., np.zeros(problem.p)))[1]
        limit = max(1, count // 6)
        removals = sorted(support, key=lambda j: -active[j])[:limit]
        additions = sorted(outside, key=lambda j: active[j])[:limit]
        swaps = [tuple(sorted((*[j for j in support if j != removed], added)))
                 for removed in removals for added in additions]
        swaps.sort(key=lambda candidate: history.lower(problem.mask(candidate)))
        supports.update(swaps[:max(1, count // 3)])
    attempts = 0
    while len(supports) < count and attempts < 4 * count:
        if remaining is not None and remaining() <= 0:
            raise TimeoutError("wall-time budget exhausted during diversity proposals")
        supports.add(tuple(sorted(rng.choice(problem.p, problem.budget, replace=False))))
        attempts += 1
    ordered = sorted(supports, key=lambda candidate:
                     (history.lower(problem.mask(candidate)), candidate))
    chosen = ordered[:count]
    if guard is not None and tuple(guard) not in chosen:
        chosen[-1] = tuple(guard)
    return [problem.mask(candidate).astype(float) for candidate in chosen]
