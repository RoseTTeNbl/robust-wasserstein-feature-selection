"""Bounds from completed binary evaluations; no extra convex solves or tests.

The scalar prediction is an upper bound on the optimum risk at a candidate.
It is not a fitted or feasible candidate test and never updates the incumbent.
"""
from math import comb

import numpy as np


class EvaluationHistory:
    def __init__(self, problem):
        self.problem = problem
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
        self.records.append(dict(support=list(support), point=z.tolist(),
                                 intercept=float(intercept), gradient=gradient.tolist(),
                                 anchor_lower=own_lower, actual_upper=float(test.upper),
                                 anchor_gap=max(0., gap), accurate=bool(accurate)))

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
