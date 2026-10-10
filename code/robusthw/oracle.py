"""Open LP/cone oracles, feasible risk certificates, and deployable tests.

Binary supports use the original finite test primal. Fractional masks use the
selected gate primal. Every call reoptimizes both sides; no historical plan
mixture defines an oracle. Repaired dual objectives are lower bounds only.
"""

import warnings
from dataclasses import dataclass, field
from functools import cached_property
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import xlogy

from .problem import OracleResult, Problem


def envelope(p0, p1, loss):
    """Bayes-envelope sum, homogeneous of degree one in the two masses."""
    if loss == "overlap":
        return float(np.minimum(p0, p1).sum())
    total = p0 + p1
    return float((xlogy(total, total) - xlogy(p0, p0) - xlogy(p1, p1)).sum())


def _remaining(deadline):
    return None if deadline is None else max(0.0, deadline - perf_counter())


def _lp(c, a_ub, b_ub, a_eq=None, b_eq=None, bounds=None, deadline=None, tolerance=1e-9):
    remaining = _remaining(deadline)
    if remaining is not None and remaining <= 0:
        return None, {"status": "time_limit", "message": "No solve time remains"}
    # HiGHS rejects feasibility tolerances below 1e-10. Keep the earlier
    # 1e-9 default, but actually tighten it when refinement requests more.
    effective_tolerance = max(1e-10, min(1e-9, float(tolerance)))
    options = {"primal_feasibility_tolerance": effective_tolerance,
               "dual_feasibility_tolerance": effective_tolerance,
               "threads": 1}
    if remaining is not None:
        options["time_limit"] = remaining
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=bounds, method="highs", options=options)
    statuses = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    info = {"status": statuses.get(result.status, "solver_error"),
            "message": result.message, "reported_objective": result.fun,
            "iterations": result.nit, "solver_tolerance": effective_tolerance}
    x = result.x
    return (x if x is not None and np.isfinite(x).all() else None), info


def _coo(rows, columns, values, shape):
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(columns))), shape=shape).tocsr()


@dataclass
class RobustTest:
    """Full-space test from a feasible original Wasserstein test certificate."""

    problem: Problem
    support: tuple[int, ...]
    potentials: tuple[np.ndarray, np.ndarray]
    multipliers: np.ndarray
    primal_objective: float
    diagnostics: dict = field(default_factory=dict)
    pool_losses: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        # Keep the finite scores' losses in memory for certificate transfer.
        # They are deliberately omitted from the existing serialization API.
        if self.pool_losses is not None:
            losses = np.array(self.pool_losses, dtype=float, copy=True)
            if losses.shape != (2, self.problem.n) or not np.isfinite(losses).all() or np.any(losses < 0):
                raise ValueError("pool_losses must be a finite nonnegative 2-by-N array")
            losses.setflags(write=False)
            self.pool_losses = losses

    def interval(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.ndim != 2 or x.shape[1] != self.problem.p or not np.isfinite(x).all():
            raise ValueError("Prediction data must be finite with the training dimension")
        selected = list(self.support)
        a = []
        for k, samples in enumerate(self.problem.samples):
            distance = np.abs(samples[:, None, selected] - x[None, :, selected]).sum(-1)
            a.append((self.potentials[k][:, None] + self.multipliers[k] * distance).min(0))
        if self.problem.loss == "overlap":
            lo, hi = np.maximum(0, 1 - a[0]), np.minimum(1, a[1])
        else:
            lo, hi = np.exp(-a[0]), -np.expm1(-a[1])
        if np.max(lo - hi, initial=0.0) > 1e-7:
            raise ArithmeticError("The stored envelopes do not define a feasible test")
        return np.clip(lo, 0, 1), np.clip(hi, 0, 1)

    def prob_class0(self, x):
        lo, hi = self.interval(x)
        return np.clip((lo + hi) / 2, 0, 1)

    def predict(self, x):
        """Class labels; ties at probability 1/2 select class 0."""
        return (self.prob_class0(x) < 0.5).astype(int)

    def predict_class0(self, x):
        return self.prob_class0(x)

    @property
    def upper(self):
        return self.primal_objective

    def as_dict(self):
        return {"support": list(self.support), "potentials": [a.tolist() for a in self.potentials],
                "multipliers": self.multipliers.tolist(), "primal_objective": self.primal_objective,
                "diagnostics": self.diagnostics}


class GateOracle:
    """Fixed box, shared triangle, or objective-scaled risk gate oracle.

    ``solve`` accepts any z in [0,1]^D. A feasible original test is attached only
    at binary z. At fractional z, ``upper`` is a feasible gate-primal objective.
    ``value`` and its tangent always come from a repaired dual certificate.
    """

    def __init__(self, problem, gate="risk", core_point=False, *, tolerance=1e-8):
        if gate not in ("risk", "triangle", "box"):
            raise ValueError("gate must be 'risk', 'triangle', or 'box'")
        self.problem, self.gate = problem, gate
        self.core_point, self.tolerance = bool(core_point), float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("oracle tolerance must be finite and positive")
        self._cone_models = {}
        self._reuse_counts = dict(models_built=0, model_reuses=0,
                                  warm_start_attempts=0, solver_workspace_reuses=0)

    @property
    def reuse_statistics(self):
        """Observed reuse, rather than a claim based only on a solver option."""
        return self._reuse_counts.copy()

    @cached_property
    def mu_bounds(self):
        """Raw risk-certificate price bounds from source-weighted distances."""
        return self.problem.price_bounds

    def _mask(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.problem.p,) or not np.isfinite(z).all() or np.any((z < 0) | (z > 1)):
            raise ValueError("Gate coordinates must lie in [0,1]")
        return z

    def _constant(self, support=()):
        p = self.problem
        a = tuple(np.full(len(q), p.B / 2) for q in p.weights)
        return RobustTest(p, tuple(support), a, np.zeros(2), p.B,
                          {"status": "constant_fallback"},
                          np.full((2, p.n), p.B / 2))

    def _potentials(self, losses, weights):
        return tuple(np.maximum(0, (losses[k][None, :] -
                                    np.einsum("imj,j->im", delta, weights[k])).max(1)) + 1e-10
                     for k, delta in enumerate(self.problem.deltas))

    def reprice(self, test, z):
        """Transfer a finite feasible certificate to a new binary support.

        Scores and class prices are retained; source allowances are recomputed.
        The returned bound belongs to the returned original feasible test and
        requires neither an oracle solve nor inversion of rounded probabilities.
        """
        z = self._mask(z)
        support = self.problem.support(z)
        if test.problem is not self.problem or test.pool_losses is None:
            raise ValueError("repricing requires a stored pool-loss certificate for this problem")
        lam = np.array(test.multipliers, dtype=float, copy=True)
        if lam.shape != (2,) or not np.isfinite(lam).all() or np.any(lam < 0):
            raise ValueError("repricing requires two finite nonnegative class prices")
        potentials = self._potentials(test.pool_losses, lam[:, None] * z)
        objective = sum(float(q @ a) for q, a in zip(self.problem.weights, potentials))
        objective += float(self.problem.theta @ lam)
        return RobustTest(self.problem, support, potentials, lam, objective,
                          {"status": "repriced_certificate", "source_support": list(test.support)},
                          test.pool_losses)

    def _repair_primal(self, z, a_lam, scores, weights=None):
        p = self.problem
        lam = np.maximum(np.asarray(a_lam, dtype=float), 0)
        if p.loss == "overlap":
            r = np.clip(scores, 0, 1)
            losses = np.array([1 - r, r])
        else:
            u = np.asarray(scores, dtype=float)
            losses = np.array([np.logaddexp(0, -u), np.logaddexp(0, u)])
        if weights is None:
            w = lam[:, None] * z
        else:
            w = np.minimum(np.maximum(weights, 0), lam[:, None])
            if self.gate == "box":
                w = np.minimum(w, p.U[:, None] * z)
            elif self.gate == "triangle":
                cost = p.theta @ w
                scale = np.minimum(1, np.divide(p.B * z, cost,
                                               out=np.ones(p.p), where=cost > 0))
                w *= scale

        a = self._potentials(losses, w)
        objective = sum(float(q @ ak) for q, ak in zip(p.weights, a)) + float(p.theta @ lam)
        if weights is not None and self.gate == "risk":
            cost = p.theta @ w
            scale = np.minimum(1, np.divide(objective * z, cost,
                                           out=np.ones(p.p), where=cost > 0))
            w *= scale
            # Decreasing nonnegative weights increases the recomputed potentials,
            # so one pass suffices for theta*w <= z*J after the repair.
            a = self._potentials(losses, w)
            objective = sum(float(q @ ak) for q, ak in zip(p.weights, a)) + float(p.theta @ lam)
        return a, lam, w, float(objective), losses

    def _primal_lp(self, z, deadline, fixed):
        if deadline is not None and _remaining(deadline) <= 0:
            return None, {"status": "time_limit", "message": "No construction time remains"}
        p, n, d = self.problem, self.problem.n, self.problem.p
        a0, lam0, score0, w0 = 0, n, n + 2, 2 * n + 2
        count = w0 if fixed else w0 + 2 * d
        c = np.zeros(count)
        c[:n] = np.concatenate(p.weights)
        c[lam0:lam0 + 2] = p.theta
        rows, cols, data, rhs = [], [], [], []
        offset = 0
        for k, delta in enumerate(p.deltas):
            nk = delta.shape[0]
            rr = np.arange(offset, offset + nk * n)
            rows.extend([rr, rr])
            cols.extend([a0 + (0 if k == 0 else p.n0) + np.repeat(np.arange(nk), n),
                         score0 + np.tile(np.arange(n), nk)])
            data.extend([-np.ones(nk * n), np.full(nk * n, -1 if k == 0 else 1)])
            if fixed:
                rows.append(rr)
                cols.append(np.full(nk * n, lam0 + k))
                data.append(-np.einsum("imj,j->im", delta, z).ravel())
            else:
                rows.append(np.repeat(rr, d))
                cols.append(w0 + k * d + np.tile(np.arange(d), nk * n))
                data.append(-delta.ravel())
            rhs.append(np.full(nk * n, -1.0 if k == 0 else 0.0))
            offset += nk * n
        if not fixed:
            rr = np.arange(offset, offset + 2 * d)
            rows.extend([rr, rr])
            cols.extend([w0 + np.arange(2 * d), lam0 + np.repeat(np.arange(2), d)])
            data.extend([np.ones(2 * d), -np.ones(2 * d)])
            rhs.append(np.zeros(2 * d))
            offset += 2 * d
            if self.gate == "box":
                rr = np.arange(offset, offset + 2 * d)
                rows.append(rr)
                cols.append(w0 + np.arange(2 * d))
                data.append(np.ones(2 * d))
                rhs.append((p.U[:, None] * z).ravel())
                offset += 2 * d
            else:
                rr = np.arange(offset, offset + d)
                for k in range(2):
                    rows.append(rr)
                    cols.append(w0 + k * d + np.arange(d))
                    data.append(np.full(d, p.theta[k]))
                if self.gate == "risk":
                    nz = np.flatnonzero(c)
                    rows.append(np.repeat(rr, len(nz)))
                    cols.append(np.tile(nz, d))
                    data.append((-z[:, None] * c[nz][None, :]).ravel())
                    rhs.append(np.zeros(d))
                else:
                    rhs.append(p.B * z)
                offset += d
        matrix = _coo(rows, cols, data, (offset, count))
        bounds = [(0, None)] * count
        bounds[score0:score0 + n] = [(0, 1)] * n
        x, info = _lp(c, matrix, np.concatenate(rhs), bounds=bounds, deadline=deadline,
                      tolerance=self.tolerance)
        if x is None:
            return None, info
        weights = None if fixed else x[w0:].reshape(2, d)
        return self._repair_primal(z, x[lam0:lam0 + 2], x[score0:score0 + n], weights), info

    def _dual_lp(self, z, deadline):
        if deadline is not None and _remaining(deadline) <= 0:
            return None, {"status": "time_limit", "message": "No construction time remains"}
        p, n, d = self.problem, self.problem.n, self.problem.p
        beta0, mu0 = n * n, n * n + 2 * d
        mudim = 2 * d if self.gate == "box" else d
        tau0 = mu0 + mudim
        h0 = tau0 + (self.gate == "risk")
        count = h0 + n
        c = np.zeros(count)
        c[h0:] = -1
        if self.gate == "box":
            c[mu0:tau0] = (p.U[:, None] * z).ravel()
        elif self.gate == "triangle":
            c[mu0:tau0] = p.B * z
        er, ec, ev, eqrhs = [], [], [], []
        gamma0, source0 = 0, 0
        for k, q in enumerate(p.weights):
            nk = len(q)
            er.append(np.repeat(source0 + np.arange(nk), n))
            ec.append(gamma0 + np.arange(nk * n))
            ev.append(np.ones(nk * n))
            if self.gate == "risk":
                er.append(source0 + np.arange(nk))
                ec.append(np.full(nk, tau0))
                ev.append(-q)
                eqrhs.append(np.zeros(nk))
            else:
                eqrhs.append(q)
            gamma0 += nk * n
            source0 += nk
        if self.gate == "risk":
            er.append(np.full(d + 1, n))
            ec.append(np.concatenate(([tau0], mu0 + np.arange(d))))
            ev.append(np.concatenate(([1.0], z)))
            eqrhs.append(np.ones(1))
        eq = _coo(er, ec, ev, (n + (self.gate == "risk"), count))
        rows, cols, data, rhs = [], [], [], []
        gamma0 = 0
        for k, delta in enumerate(p.deltas):
            nk = delta.shape[0]
            rr = k * d + np.arange(d)
            rows.extend([np.repeat(rr, nk * n), rr, rr])
            cols.extend([np.tile(gamma0 + np.arange(nk * n), d), beta0 + k * d + np.arange(d),
                         mu0 + (k * d if self.gate == "box" else 0) + np.arange(d)])
            data.extend([delta.reshape(-1, d).T.ravel(), -np.ones(d),
                         -np.full(d, 1.0 if self.gate == "box" else p.theta[k])])
            gamma0 += nk * n
        rhs.append(np.zeros(2 * d))
        for k in range(2):
            rows.append(np.full(d, 2 * d + k))
            cols.append(beta0 + k * d + np.arange(d))
            data.append(np.ones(d))
            if self.gate == "risk":
                rows.append(np.array([2 * d + k]))
                cols.append(np.array([tau0]))
                data.append(np.array([-p.theta[k]]))
        rhs.append(np.zeros(2) if self.gate == "risk" else p.theta)
        gamma0 = 0
        for k, q in enumerate(p.weights):
            rr = 2 * d + 2 + k * n + np.arange(n)
            rows.extend([rr, np.tile(rr, len(q))])
            cols.extend([h0 + np.arange(n), gamma0 + np.arange(len(q) * n)])
            data.extend([np.ones(n), -np.ones(len(q) * n)])
            gamma0 += len(q) * n
        rhs.append(np.zeros(2 * n))
        ub = _coo(rows, cols, data, (2 * d + 2 + 2 * n, count))
        bub, beq = np.concatenate(rhs), np.concatenate(eqrhs)
        x, info = _lp(c, ub, bub, eq, beq, deadline=deadline, tolerance=self.tolerance)
        if x is None:
            return None, info
        if self.core_point and info["status"] == "optimal":
            core = np.full(d, p.budget / d)
            second = np.zeros(count)
            if self.gate == "risk":
                second[mu0:tau0] = core - z
            else:
                second[h0:] = -1
                second[mu0:tau0] = ((p.U[:, None] * core).ravel() if self.gate == "box"
                                       else p.B * core)
            face = sparse.vstack([ub, sparse.csr_matrix(c[None, :])], format="csr")
            x2, second_info = _lp(second, face, np.append(bub, c @ x + 1e-9), eq, beq,
                                  deadline=deadline, tolerance=self.tolerance)
            info["core_point"] = second_info
            if x2 is not None:
                x = x2
        plans = (x[:p.n0 * n].reshape(p.n0, n),
                 x[p.n0 * n:n * n].reshape(p.n1, n))
        return (plans, x[beta0:mu0].reshape(2, d),
                float(x[tau0]) if self.gate == "risk" else 1.0), info

    def _cvx_solve(self, model, deadline, *, persistent=False, model_reused=False):
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            return {"status": "time_limit", "message": "No solve time remains"}
        # A requested refinement updates even a cached Clarabel workspace.
        # Below 1e-12 we report an accuracy floor rather than promise arbitrary
        # precision from floating-point solves.
        effective_tolerance = max(1e-12, min(1e-9, self.tolerance))
        options = {"tol_gap_abs": effective_tolerance, "tol_gap_rel": effective_tolerance,
                   "tol_feas": effective_tolerance,
                   "max_iter": 300}
        if remaining is not None:
            options["time_limit"] = remaining
        if persistent:
            # Clarabel can update this workspace only when presolve does not
            # change its dimensions. Parameter changes still reoptimize fully.
            options["presolve_enable"] = False
        previous = getattr(model, "_solver_cache", {}).get("CLARABEL")
        if persistent and previous is not None:
            self._reuse_counts["warm_start_attempts"] += 1
        try:
            model.solve(solver="CLARABEL", verbose=False, warm_start=True, **options)
            reused_workspace = previous is not None and (
                getattr(model, "_solver_cache", {}).get("CLARABEL") is previous)
            if persistent and reused_workspace:
                self._reuse_counts["solver_workspace_reuses"] += 1
            return {"status": str(model.status), "reported_objective": model.value,
                    "solve_time": model.solver_stats.solve_time,
                    "model_reused": model_reused,
                    "solver_tolerance": effective_tolerance,
                    "warm_start_requested": persistent and previous is not None,
                    "solver_workspace_reused": persistent and reused_workspace}
        except Exception as exc:
            return {"status": "solver_error", "message": str(exc)}

    @staticmethod
    def _cvxpy():
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("Logistic requires optional CVXPY and Clarabel dependencies") from exc
        return cp

    def _primal_cone_model(self, z, fixed, *, parameterized=False):
        """Logistic finite-test model; parameters change data, not constraints."""
        cp, p = self._cvxpy(), self.problem
        a = [cp.Variable(len(q), nonneg=True) for q in p.weights]
        lam, u = cp.Variable(2, nonneg=True), cp.Variable(p.n)
        objective = sum(q @ ak for q, ak in zip(p.weights, a)) + p.theta @ lam
        w = None if fixed else cp.Variable((2, p.p), nonneg=True)
        parameter = None
        if parameterized:
            parameter = ([cp.Parameter((len(q), p.n), nonneg=True) for q in p.weights]
                         if fixed else cp.Parameter(p.p, nonneg=True))
        constraints = []
        for k, delta in enumerate(p.deltas):
            if fixed:
                cost = (cp.multiply(parameter[k], lam[k]) if parameterized else
                        np.einsum("imj,j->im", delta, z) * lam[k])
            else:
                cost = (delta.reshape(-1, p.p) @ w[k, :]).reshape(
                    (len(p.weights[k]), p.n), order="C")
            loss = cp.logistic(-u if k == 0 else u)
            constraints.append(a[k][:, None] + cost >= loss[None, :])
        if not fixed:
            constraints.append(w <= lam[:, None])
            if self.gate == "box":
                constraints.append(w <= p.U[:, None] * z)
            else:
                allowance = objective if self.gate == "risk" else p.B
                constraints.append(p.theta @ w <= (cp.multiply(parameter, allowance)
                                                   if parameterized else allowance * z))
        model = cp.Problem(cp.Minimize(objective), constraints)
        return dict(model=model, lam=lam, u=u, w=w, parameter=parameter)

    def _primal_cone(self, z, deadline, fixed):
        if deadline is not None and _remaining(deadline) <= 0:
            return None, {"status": "time_limit", "message": "No construction time remains"}
        if self.gate == "risk" and not self.core_point:
            return self._persistent_primal_cone(z, deadline, fixed)
        bundle = self._primal_cone_model(z, fixed)
        info = self._cvx_solve(bundle["model"], deadline)
        lam, u, w = bundle["lam"], bundle["u"], bundle["w"]
        if lam.value is None or u.value is None or not np.isfinite(u.value).all():
            return None, info
        weights = None if fixed else w.value
        return self._repair_primal(z, lam.value, u.value, weights), info

    def _persistent_primal_cone(self, z, deadline, fixed):
        key = ("primal", fixed)
        reused = key in self._cone_models
        if not reused:
            self._cone_models[key] = self._primal_cone_model(z, fixed, parameterized=True)
            self._reuse_counts["models_built"] += 1
        else:
            self._reuse_counts["model_reuses"] += 1
        bundle = self._cone_models[key]
        if fixed:
            for parameter, delta in zip(bundle["parameter"], self.problem.deltas):
                parameter.value = np.einsum("imj,j->im", delta, z)
        else:
            bundle["parameter"].value = z
        info = self._cvx_solve(bundle["model"], deadline, persistent=True, model_reused=reused)
        if info["status"] in ("solver_error", "time_limit"):
            return None, info
        lam, u, w = bundle["lam"], bundle["u"], bundle["w"]
        if lam.value is None or u.value is None or not np.isfinite(u.value).all():
            return None, info
        return self._repair_primal(z, lam.value, u.value, None if fixed else w.value), info

    def _dual_cone_model(self, z, *, parameterized=False):
        """Logistic certificate model with explicit mass and price constraints."""
        cp, p = self._cvxpy(), self.problem
        plans = [cp.Variable((len(q), p.n), nonneg=True) for q in p.weights]
        beta = cp.Variable((2, p.p), nonneg=True)
        mu = cp.Variable((2, p.p) if self.gate == "box" else p.p, nonneg=True)
        tau = cp.Variable(nonneg=True) if self.gate == "risk" else 1.0
        parameter = cp.Parameter(p.p, nonneg=True) if parameterized else None
        constraints = ([tau + mu @ parameter == 1, mu <= tau * self.mu_bounds]
                       if parameterized else [])
        for k, (q, delta) in enumerate(zip(p.weights, p.deltas)):
            costs = delta.reshape(-1, p.p).T @ cp.reshape(plans[k], (len(q) * p.n,), order="C")
            constraints.extend([cp.sum(plans[k], axis=1) == tau * q,
                                cp.sum(beta[k]) <= tau * p.theta[k],
                                costs <= beta[k] + (mu[k] if self.gate == "box" else p.theta[k] * mu)])
        mass = [cp.sum(g, axis=0) for g in plans]
        total = mass[0] + mass[1]
        h = cp.sum(-cp.rel_entr(mass[0], total) - cp.rel_entr(mass[1], total))
        if self.gate == "risk":
            if not parameterized:
                constraints.append(tau + mu @ z == 1)
            objective = h
        elif self.gate == "box":
            objective = h - cp.sum(cp.multiply(p.U[:, None] * z, mu))
        else:
            objective = h - p.B * (mu @ z)
        model = cp.Problem(cp.Maximize(objective), constraints)
        return dict(model=model, plans=plans, beta=beta, mu=mu, tau=tau,
                    parameter=parameter, constraints=constraints, objective=objective, entropy=h)

    def _dual_cone(self, z, deadline):
        if deadline is not None and _remaining(deadline) <= 0:
            return None, {"status": "time_limit", "message": "No construction time remains"}
        if self.gate == "risk" and not self.core_point:
            return self._persistent_dual_cone(z, deadline)
        bundle = self._dual_cone_model(z)
        model = bundle["model"]
        plans, beta, tau = bundle["plans"], bundle["beta"], bundle["tau"]
        info = self._cvx_solve(model, deadline)
        if any(g.value is None for g in plans) or beta.value is None:
            return None, info
        if self.core_point and info["status"] in ("optimal", "optimal_inaccurate"):
            cp, p = self._cvxpy(), self.problem
            objective, mu, h = bundle["objective"], bundle["mu"], bundle["entropy"]
            core = np.full(p.p, p.budget / p.p)
            face = bundle["constraints"] + [objective >= float(model.value) - 1e-8]
            if self.gate == "risk":
                second_objective = -mu @ (core - z)
            elif self.gate == "box":
                second_objective = h - cp.sum(cp.multiply(p.U[:, None] * core, mu))
            else:
                second_objective = h - p.B * (mu @ core)
            second = cp.Problem(cp.Maximize(second_objective), face)
            first = ([g.value.copy() for g in plans], beta.value.copy(),
                     float(tau.value) if self.gate == "risk" else 1.0)
            second_info = self._cvx_solve(second, deadline)
            info["core_point"] = second_info
            if any(g.value is None for g in plans) or beta.value is None:
                return first, info
        return ([g.value for g in plans], beta.value,
                float(tau.value) if self.gate == "risk" else 1.0), info

    def _persistent_dual_cone(self, z, deadline):
        key = ("dual", "risk")
        reused = key in self._cone_models
        if not reused:
            self._cone_models[key] = self._dual_cone_model(z, parameterized=True)
            self._reuse_counts["models_built"] += 1
        else:
            self._reuse_counts["model_reuses"] += 1
        bundle = self._cone_models[key]
        bundle["parameter"].value = z
        info = self._cvx_solve(bundle["model"], deadline, persistent=True, model_reused=reused)
        if info["status"] in ("solver_error", "time_limit"):
            return None, info
        plans, beta, tau = bundle["plans"], bundle["beta"], bundle["tau"]
        if any(g.value is None for g in plans) or beta.value is None or tau.value is None:
            return None, info
        return ([g.value for g in plans], beta.value, float(tau.value)), info

    def _repair_dual(self, z, raw):
        p = self.problem
        raw_plans, beta, tau = raw
        tau = max(0.0, float(tau)) if self.gate == "risk" else 1.0
        plans = []
        for k, (g, q) in enumerate(zip(raw_plans, p.weights)):
            g = np.maximum(np.asarray(g, dtype=float), 0)
            if not np.isfinite(g).all():
                raise ArithmeticError("Nonfinite dual mass")
            for i in range(len(q)):
                if g[i].sum() <= 0:
                    g[i, i + (0 if k == 0 else p.n0)] = 1
                g[i] *= tau * q[i] / g[i].sum()
            plans.append(g)
        beta = np.maximum(np.asarray(beta, dtype=float), 0)
        if not np.isfinite(beta).all():
            raise ArithmeticError("Nonfinite dual budget allocation")
        for k in range(2):
            if beta[k].sum() > tau * p.theta[k]:
                beta[k] *= tau * p.theta[k] / beta[k].sum()
        costs = np.array([np.einsum("im,imj->j", g, delta) for g, delta in zip(plans, p.deltas)])
        excess = np.maximum(costs - beta, 0)
        if self.gate == "box":
            mu = excess
        else:
            mu = np.max(excess / p.theta[:, None], axis=0)
        if self.gate == "risk":
            # Source rows have mass tau*q, hence movement in each coordinate
            # is bounded by tau times its source-weighted maximum distance.
            mu = np.minimum(mu, tau * self.mu_bounds)
            beta = np.maximum(costs - p.theta[:, None] * mu, 0)
            if np.any(beta.sum(1) > tau * p.theta + 1e-9 * np.maximum(1., tau * p.theta)):
                raise ArithmeticError("Capped dual coordinate prices exceed the class budget")
            normalization = tau + float(mu @ z)
            if normalization <= 1e-14:
                return 0.0, np.zeros(p.p), 0.0, {"status": "zero_certificate"}
            plans = [g / normalization for g in plans]
            beta, costs, mu, tau = beta / normalization, costs / normalization, mu / normalization, tau / normalization
        h = envelope(plans[0].sum(0), plans[1].sum(0), p.loss)
        if self.gate == "risk":
            value, gradient = h, -h * mu
        elif self.gate == "triangle":
            gradient = -p.B * mu
            value = h + float(gradient @ z)
        else:
            gradient = -(p.U[:, None] * mu).sum(0)
            value = h + float(gradient @ z)
        margin = 1e-10 * (1 + abs(value) + abs(float(gradient @ z)))
        value -= margin
        intercept = value - float(gradient @ z)
        residual_mu = mu if self.gate == "box" else p.theta[:, None] * mu
        certificate = {"gamma": plans, "beta": beta, "mu": mu, "tau": tau, "entropy": h,
                       "costs": costs, "margin": margin,
                       "row_residual": max(float(np.abs(g.sum(1) - tau * q).max())
                                           for g, q in zip(plans, p.weights)),
                       "budget_residual": max(0., float(np.max(beta.sum(1) - tau * p.theta))),
                       "coordinate_residual": max(0., float(np.max(costs - beta - residual_mu))),
                       "normalization_residual": (abs(tau + float(mu @ z) - 1)
                                                  if self.gate == "risk" else 0.0),
                       "mu_bound_residual": (max(0., float(np.max(mu - tau * self.mu_bounds)))
                                             if self.gate == "risk" else 0.0)}
        return float(value), gradient, float(intercept), certificate

    def gate_primal(self, z, time_limit=None):
        """Independent gate primal, also callable at binary masks for audits."""
        z = self._mask(z)
        deadline = None if time_limit is None else perf_counter() + max(0., float(time_limit))
        method = self._primal_lp if self.problem.loss == "overlap" else self._primal_cone
        solution, info = method(z, deadline, fixed=False)
        return (self.problem.B if solution is None else min(self.problem.B, solution[3])), info

    def fixed_support(self, z, time_limit=None):
        z = self._mask(z)
        support = self.problem.support(z)
        deadline = None if time_limit is None else perf_counter() + max(0., float(time_limit))
        method = self._primal_lp if self.problem.loss == "overlap" else self._primal_cone
        solution, info = method(z, deadline, fixed=True)
        test = self._constant(support)
        if solution is not None and solution[3] < test.primal_objective:
            test = RobustTest(self.problem, support, solution[0], solution[1], solution[3], info,
                              solution[4])
        else:
            test.diagnostics = info
        return test

    def solve(self, z, time_limit=None):
        z = self._mask(z)
        started = perf_counter()
        if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
            raise ValueError("time_limit must be finite and nonnegative")
        deadline = None if time_limit is None else started + float(time_limit)
        primal_deadline = None if deadline is None else started + float(time_limit) / 2
        binary = bool(np.all((z == 0) | (z == 1)))
        method = self._primal_lp if self.problem.loss == "overlap" else self._primal_cone
        primal, primal_info = method(z, primal_deadline, fixed=binary)
        upper, test = self.problem.B, None
        if binary:
            test = self._constant(self.problem.support(z))
        if primal is not None and primal[3] < upper:
            upper = primal[3]
            if binary:
                test = RobustTest(self.problem, self.problem.support(z), primal[0], primal[1], upper,
                                  primal_info, primal[4])
        dual_method = self._dual_lp if self.problem.loss == "overlap" else self._dual_cone
        dual, dual_info = dual_method(z, deadline)
        if dual is None:
            value, gradient, intercept, certificate = 0., np.zeros(self.problem.p), 0., {}
        else:
            value, gradient, intercept, certificate = self._repair_dual(z, dual)
        if value < 0:
            value, gradient, intercept = 0., np.zeros(self.problem.p), 0.
            certificate["zero_cut_used"] = True
        certificate.update({"gate": self.gate, "primal": primal_info, "dual": dual_info,
                            "binary": binary, "gap": upper - value,
                            "model_reuse": self.reuse_statistics})
        if value > upper + 1e-6:
            raise ArithmeticError(f"Repaired certificates are inconsistent: {value} > {upper}")
        if upper - value <= max(1e-6, self.tolerance):
            status = "optimal"
        elif "time_limit" in (primal_info["status"], dual_info["status"]) or (
                deadline is not None and _remaining(deadline) <= 0):
            status = "time_limit"
        elif "solver_error" in (primal_info["status"], dual_info["status"]):
            status = "solver_error"
        else:
            status = "accuracy_limit"
        return OracleResult(value, float(upper), gradient, intercept, test, status,
                            perf_counter() - started, certificate)

    def solve_relaxation(self, z, time_limit=None):
        return self.solve(z, time_limit=time_limit)


class FixedSupportOracle:
    """Small wrapper exposing only original fixed-support primal test fitting."""

    def __init__(self, problem):
        self.problem = problem
        self._oracle = GateOracle(problem)

    def solve(self, z, time_limit=None):
        return self._oracle.fixed_support(z, time_limit=time_limit)
