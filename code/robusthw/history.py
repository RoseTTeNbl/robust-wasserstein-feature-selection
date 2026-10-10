"""Reusable feasible tests and dual certificates; no new convex solves here."""
import numpy as np


class SearchHistory:
    def __init__(self, problem, oracle, *, ratios=False):
        self.problem, self.oracle = problem, oracle
        self.use_ratios = ratios
        self.cuts, self.ratios, self.tests = [], [], {}
        self.points, self.anchor_gaps = [], []
        self.upper_cache = {}
        self.transfer_calls, self.extra_cuts = 0, 0
        movement = np.array([q @ delta.max(axis=1)
                             for q, delta in zip(problem.weights, problem.deltas)])
        self.metric_weights = (movement / problem.theta[:, None]).max(axis=0)

    def add(self, z, result):
        self.cuts.append((float(result.intercept), np.asarray(result.gradient).copy()))
        certificate = result.certificate
        tau = float(certificate.get("tau", 0.))
        numerator = max(0., float(certificate.get("entropy", 0.)) -
                        float(certificate.get("margin", 0.)))
        if tau > 0 and numerator > 0:
            self.ratios.append((numerator, tau, np.asarray(certificate["mu"]).copy()))
        if result.test is not None:
            support = self.problem.support(z)
            previous = self.tests.get(support)
            if previous is None or result.test.upper < previous.upper:
                self.tests[support] = result.test
                self.upper_cache.clear()

    def lower(self, z):
        value = max([0., *[intercept + float(gradient @ z)
                           for intercept, gradient in self.cuts]])
        if self.use_ratios:
            value = max(value, max((h / (tau + float(mu @ z))
                                    for h, tau, mu in self.ratios), default=0.))
        return float(value)

    def upper(self, z, remaining=None):
        """Reprice each stored score vector, returning an actual feasible test."""
        key = self.problem.support(z)
        if key in self.upper_cache:
            return self.upper_cache[key]
        best = self.oracle._constant(key)
        for test in self.tests.values():
            if remaining is not None and remaining() <= 0:
                raise TimeoutError("wall-time budget exhausted during history transfer")
            if test.support == key:
                transferred = test
            else:
                transferred = self.oracle.reprice(test, z)
                self.transfer_calls += 1
            if transferred.upper < best.upper:
                best = transferred
        self.upper_cache[key] = best
        return best

    def densify(self, points, remaining=None):
        """Use complete old fractions to create stronger tangents at new points."""
        for z in points:
            if remaining is not None and remaining() <= 0:
                raise TimeoutError("wall-time budget exhausted during certificate reuse")
            if not self.ratios:
                return
            h, tau, mu = max(self.ratios, key=lambda item:
                             item[0] / (item[1] + float(item[2] @ z)))
            denominator = tau + float(mu @ z)
            value = h / denominator
            if value <= self.lower(z) + 1e-9:
                continue
            gradient = -h * mu / denominator**2
            margin = 1e-10 * (1 + abs(value) + abs(float(gradient @ z)))
            self.cuts.append((value - margin - float(gradient @ z), gradient))
            self.extra_cuts += 1

    def completed(self, z, gap):
        self.points.append(np.asarray(z).copy())
        self.anchor_gaps.append(float(gap))

    def distance(self, z, other):
        return float(self.metric_weights @ np.abs(np.asarray(z) - np.asarray(other)))


def proposal_pool(problem, history, incumbent, *, count, rng, guard=None, remaining=None):
    """History-directed swaps, active-price rankings, and diverse proposals."""
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
    # Protect the guard while keeping repricing work bounded.
    ordered = sorted(supports, key=lambda candidate: (history.lower(problem.mask(candidate)), candidate))
    chosen = ordered[:count]
    if guard is not None and tuple(guard) not in chosen:
        chosen[-1] = tuple(guard)
    return [problem.mask(candidate).astype(float) for candidate in chosen]
