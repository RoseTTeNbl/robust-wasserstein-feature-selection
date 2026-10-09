"""Independent small numerical checks of the generalized loss theory.

This file is confined to an exclusive system-TEMP directory. It imports no
robusthw code, reads no research data/configuration/results, and builds no PDFs.
Raw conic residuals are recorded. Deployment uses a separately documented
conservative source-intercept repair, not an assumed exact solver return.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import clarabel
import cvxpy as cp
import numpy as np

SEED = 20261009
SOLVER = dict(solver="CLARABEL", tol_gap_abs=2e-10,
              tol_gap_rel=2e-10, tol_feas=2e-10, max_iter=1000)
ACCEPTANCE = 1e-6
DEPLOYMENT_MARGIN = 1e-9
LOSSES = ("half_hinge", "logistic", "exponential", "squared_hinge")
DATA = (
    dict(name="unequal_weights_D2",
         x0=[[-1.0, 0.2], [0.1, -0.7]],
         x1=[[0.8, 0.5], [1.6, -0.2]],
         q0=[0.3, 0.7], q1=[0.65, 0.35],
         theta=[0.18, 0.27],
         fractional=[[0.2, 0.7], [0.5, 0.5]]),
    dict(name="unequal_sizes_weights_D3",
         x0=[[-1.2, 0.5, -0.1], [-0.3, -0.8, 0.4], [0.4, 0.3, -0.6]],
         x1=[[0.9, -0.1, 0.8], [1.5, 0.7, 0.2]],
         q0=[0.2, 0.5, 0.3], q1=[0.4, 0.6],
         theta=[0.22, 0.11],
         fractional=[[0.15, 0.55, 0.8], [0.6, 0.25, 0.45]]),
)


def phi_expr(loss, value):
    if loss == "half_hinge":
        return 0.5 * cp.pos(1 + value)
    if loss == "logistic":
        return cp.logistic(value)
    if loss == "exponential":
        return cp.exp(value)
    return cp.square(cp.pos(1 + value))


def phi_numpy(loss, value):
    value = np.asarray(value, float)
    if loss == "half_hinge":
        return 0.5 * np.maximum(1 + value, 0)
    if loss == "logistic":
        return np.logaddexp(0, value)
    if loss == "exponential":
        return np.exp(value)
    return np.maximum(1 + value, 0) ** 2


def psi_expr(loss, p, q):
    if loss == "half_hinge":
        return cp.sum(cp.minimum(p, q))
    if loss == "logistic":
        return cp.sum(-cp.rel_entr(p, p + q) - cp.rel_entr(q, p + q))
    if loss == "exponential":
        return cp.sum(cp.hstack([2 * cp.geo_mean(cp.hstack([p[m], q[m]]))
                                for m in range(p.shape[0])]))
    return cp.sum(cp.hstack([(p[m] + q[m]) -
                            cp.quad_over_lin(p[m] - q[m], p[m] + q[m])
                            for m in range(p.shape[0])]))


def psi_numpy(loss, p, q):
    # Preserve raw-mass feasibility diagnostics separately; this avoids
    # evaluating entropy or square roots at tiny negative conic roundoff.
    p, q = np.maximum(p, 0), np.maximum(q, 0)
    if loss == "half_hinge":
        return float(np.minimum(p, q).sum())
    if loss == "logistic":
        def xlogx(x):
            out = np.zeros_like(x)
            mask = x > 0
            out[mask] = x[mask] * np.log(x[mask])
            return out
        return float((xlogx(p + q) - xlogx(p) - xlogx(q)).sum())
    if loss == "exponential":
        return float((2 * np.sqrt(p * q)).sum())
    return float(np.divide(4 * p * q, p + q, out=np.zeros_like(p),
                           where=(p + q) > 0).sum())


def positive_max(value):
    return float(max(0.0, np.max(np.asarray(value, float))))


def solve_checked(problem):
    assert problem.is_dcp()
    started = perf_counter()
    problem.solve(**SOLVER)
    elapsed = perf_counter() - started
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Unexpected status {problem.status}")
    return elapsed


def primal(x, q, theta, targets, distance, z, loss, gate):
    D, N = targets.shape[1], len(targets)
    a = [cp.Variable(len(q[k])) for k in range(2)]
    u = cp.Variable(N)
    lam = cp.Variable(2, nonneg=True)
    J = sum(q[k] @ a[k] for k in range(2)) + theta @ lam
    constraints = []
    w = cp.Variable((2, D), nonneg=True) if gate else None
    eta = cp.Variable() if gate else None
    for k, sign in enumerate((-1, 1)):
        losses = phi_expr(loss, sign * u)
        for i in range(len(q[k])):
            movement = distance[k][i] @ w[k] if gate else lam[k] * (distance[k][i] @ z)
            constraints.append(losses <= a[k][i] + movement)
    if gate:
        constraints.extend([w[k] <= lam[k] for k in range(2)])
        constraints.extend([J <= eta, theta @ w <= eta * z])
    problem = cp.Problem(cp.Minimize(eta if gate else J), constraints)
    seconds = solve_checked(problem)
    av, uv, lv = [v.value.copy() for v in a], u.value.copy(), lam.value.copy()
    jv = float(sum(q[k] @ av[k] for k in range(2)) + theta @ lv)
    checks = dict(lambda_nonnegative=positive_max(-lv))
    if gate:
        wv, etav = w.value.copy(), float(eta.value)
        checks.update(w_nonnegative=positive_max(-wv),
                      w_lambda=positive_max(wv - lv[:, None]),
                      objective_epigraph=max(0.0, jv - etav),
                      scaled_gate=positive_max(theta @ wv - etav * z))
    else:
        wv, etav = lv[:, None] * z, jv
    checks["loss"] = max(positive_max(phi_numpy(loss, (-1 if k == 0 else 1) * uv)[None, :]
                                    - av[k][:, None]
                                    - np.einsum("imj,j->im", distance[k], wv[k]))
                         for k in range(2))
    return dict(value=float(problem.value), J=jv, status=problem.status,
                seconds=seconds, a=av, u=uv, lam=lv, residuals=checks)


def dual(q, theta, targets, distance, z, loss):
    D = targets.shape[1]
    tau = cp.Variable(nonneg=True)
    mu = cp.Variable(D, nonneg=True)
    plans = [cp.Variable((len(q[k]), len(targets)), nonneg=True) for k in range(2)]
    A = [cp.hstack([cp.sum(cp.multiply(plans[k], distance[k][:, :, j]))
                    for j in range(D)]) for k in range(2)]
    constraints = [cp.sum(plans[k], axis=1) == tau * q[k] for k in range(2)]
    constraints += [cp.sum(cp.pos(A[k] - theta[k] * mu)) <= tau * theta[k]
                    for k in range(2)]
    constraints += [tau + mu @ z == 1]
    p = [cp.sum(plans[k], axis=0) for k in range(2)]
    problem = cp.Problem(cp.Maximize(psi_expr(loss, *p)), constraints)
    seconds = solve_checked(problem)
    tv, mv = float(tau.value), mu.value.copy()
    pi = [v.value.copy() for v in plans]
    av = [np.einsum("im,imj->j", pi[k], distance[k]) for k in range(2)]
    pv = [pi[k].sum(axis=0) for k in range(2)]
    h = psi_numpy(loss, *pv)
    raw_solver_value = float(problem.value)
    checks = dict(
        tau_nonnegative=max(0.0, -tv), mu_nonnegative=positive_max(-mv),
        plan_nonnegative=max(positive_max(-v) for v in pi),
        row_conservation=max(float(np.max(np.abs(pi[k].sum(axis=1) - tv * q[k])))
                             for k in range(2)),
        budget=max(max(0.0, float(np.maximum(av[k] - theta[k] * mv, 0).sum()
                                 - tv * theta[k])) for k in range(2)),
        normalization=abs(tv + mv @ z - 1),
    )
    if tv <= 1e-12:
        raise RuntimeError("Unexpected zero-scale optimum in a nontrivial positive-radius fixture")
    raw_pi, raw_mu = [v / tv for v in pi], mv / tv
    raw_a = [v / tv for v in av]
    raw_h = psi_numpy(loss, *[v.sum(axis=0) for v in raw_pi])
    fraction = raw_h / (1 + raw_mu @ z)
    raw_checks = dict(
        row_conservation=max(float(np.max(np.abs(raw_pi[k].sum(axis=1) - q[k])))
                             for k in range(2)),
        budget=max(max(0.0, float(np.maximum(raw_a[k] - theta[k] * raw_mu, 0).sum()
                                 - theta[k])) for k in range(2)),
        plan_nonnegative=max(positive_max(-v) for v in raw_pi),
        mu_nonnegative=positive_max(-raw_mu),
    )
    return dict(value=h, solver_value=raw_solver_value if np.isfinite(raw_solver_value) else None,
                status=problem.status, seconds=seconds, tau=tv,
                normalized_residuals=checks, raw_residuals=raw_checks,
                fraction_value=float(fraction), fraction_difference=abs(fraction-h),
                entropy_roundoff_objective_difference=abs(h-raw_solver_value)
                if np.isfinite(raw_solver_value) else None)


def deploy_check(x, q, theta, targets, distance, z, loss, solution, rng):
    # Independently recover an original binary feasible test. Saturate the
    # prices to lambda*z, clip lambda's conic roundoff, and raise each intercept
    # just enough to satisfy every original target inequality plus 1e-9.
    lam = np.maximum(solution["lam"], 0)
    u = solution["u"]
    a = []
    for k, sign in enumerate((-1, 1)):
        original_cost = np.einsum("imj,j->im", distance[k], z) * lam[k]
        needed = (phi_numpy(loss, sign*u)[None, :] - original_cost).max(axis=1)
        a.append(np.maximum(np.maximum(solution["a"][k], needed), 0)
                 + DEPLOYMENT_MARGIN)
    J = float(sum(q[k] @ a[k] for k in range(2)) + theta @ lam)
    offsets = (0, len(x[0]))

    def evaluate(points):
        points = np.atleast_2d(points)
        r = [np.einsum("pij,j->pi", np.abs(points[:, None, :] - x[k][None, :, :]), z)
             for k in range(2)]
        envelope_values = [a[k][None, :] + lam[k] * r[k] for k in range(2)]
        index = [v.argmin(axis=1) for v in envelope_values]  # deterministic first-index tie
        E = [envelope_values[k][np.arange(len(points)), index[k]] for k in range(2)]
        c = (np.abs(x[0][index[0]] - x[1][index[1]]) * z).sum(axis=1)
        r0 = r[0][np.arange(len(points)), index[0]]
        alpha = np.divide(np.minimum(c, r0), c, out=np.zeros_like(c), where=c > 0)
        score = ((1-alpha) * u[offsets[0]+index[0]]
                 + alpha * u[offsets[1]+index[1]])
        return score, E

    midpoints = np.array([(a0+a1)/2 for a0 in x[0] for a1 in x[1]])
    points = np.vstack([targets, midpoints,
                        rng.normal(size=(160, targets.shape[1])),
                        rng.normal(size=(80, targets.shape[1])) * 100])
    score, E = evaluate(points)
    envelope_violation = max(positive_max(phi_numpy(loss, -score) - E[0]),
                             positive_max(phi_numpy(loss, score) - E[1]))
    score_bound_violation = positive_max(np.abs(score) - np.max(np.abs(u)))
    unobserved = np.flatnonzero(z == 0)
    changed = points.copy()
    changed[:, unobserved] += 73.0
    changed_score, _ = evaluate(changed)
    invariance = float(np.max(np.abs(changed_score-score)))

    couplings = []
    for fraction in (0.2, 0.6, 1.0):
        risks, costs = [], []
        for k, sign in enumerate((-1, 1)):
            directions = rng.normal(size=x[k].shape)
            directions /= np.abs(directions).sum(axis=1, keepdims=True)
            moved = x[k] + theta[k] * fraction * (1-1e-9) * directions
            gs, _ = evaluate(moved)
            risks.append(float(q[k] @ phi_numpy(loss, sign*gs)))
            costs.append(float(q[k] @ np.abs(moved-x[k]).sum(axis=1)))
        couplings.append(dict(kind="all_source_atoms_shifted", fraction=fraction,
                              class_costs=costs, risk_sum=sum(risks),
                              risk_violation=max(0.0, sum(risks)-J)))
    for target in (np.zeros(targets.shape[1]), 8 + 2*np.arange(targets.shape[1])):
        risks, costs, moved_masses = [], [], []
        for k, sign in enumerate((-1, 1)):
            average_distance = float(q[k] @ np.abs(x[k]-target).sum(axis=1))
            moved_mass = min(0.8, theta[k] * (1-1e-9) / average_distance)
            own_score, _ = evaluate(x[k])
            new_score, _ = evaluate(target)
            risk = ((1-moved_mass) * (q[k] @ phi_numpy(loss, sign*own_score))
                    + moved_mass * phi_numpy(loss, sign*new_score)[0])
            risks.append(float(risk)); costs.append(moved_mass*average_distance)
            moved_masses.append(moved_mass)
        couplings.append(dict(kind="common_new_point_mixture", target=target.tolist(),
                              moved_masses=moved_masses, class_costs=costs,
                              risk_sum=sum(risks), risk_violation=max(0.0, sum(risks)-J)))
    return dict(repaired_J=J, repair_increase=J-solution["J"],
                repair_vs_gate_objective=J-solution["value"],
                point_count=len(points), envelope_violation=envelope_violation,
                bounded_score_violation=score_bound_violation,
                unselected_coordinate_invariance_error=invariance,
                coupling_count=len(couplings), couplings=couplings,
                coupling_budget_violation=max(max(0.0, c["class_costs"][k]-theta[k])
                                              for c in couplings for k in range(2)),
                coupling_risk_violation=max(c["risk_violation"] for c in couplings))


def main():
    rows = []
    started = perf_counter()
    for fixture_index, fixture in enumerate(DATA):
        x = [np.array(fixture[f"x{k}"], float) for k in range(2)]
        q = [np.array(fixture[f"q{k}"], float) for k in range(2)]
        theta = np.array(fixture["theta"], float)
        targets = np.vstack(x)
        distance = [np.abs(x[k][:, None, :] - targets[None, :, :]) for k in range(2)]
        binary = [np.array(v, float) for v in itertools.product((0, 1), repeat=targets.shape[1])]
        anchors = binary + [np.array(v, float) for v in fixture["fractional"]]
        for loss_index, loss in enumerate(LOSSES):
            for anchor_index, z in enumerate(anchors):
                pr = primal(x, q, theta, targets, distance, z, loss, gate=True)
                du = dual(q, theta, targets, distance, z, loss)
                row = dict(fixture=fixture["name"], loss=loss, z=z.tolist(),
                           binary=anchor_index < len(binary),
                           primal_value=pr["value"], primal_J=pr["J"],
                           dual=du, absolute_primal_dual_gap=abs(pr["value"]-du["value"]),
                           signed_primal_minus_dual=pr["value"]-du["value"],
                           primal_status=pr["status"], primal_seconds=pr["seconds"],
                           primal_residuals=pr["residuals"])
                if row["binary"]:
                    original = primal(x, q, theta, targets, distance, z, loss, gate=False)
                    row.update(original_value=original["value"],
                               original_status=original["status"],
                               original_residuals=original["residuals"],
                               binary_original_gate_difference=abs(original["value"]-pr["value"]))
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [SEED, fixture_index, loss_index, anchor_index]))
                    row["deployment"] = deploy_check(x,q,theta,targets,distance,z,loss,pr,rng)
                rows.append(row)
            print(f"Completed {fixture['name']} / {loss}: {len(anchors)} anchor pairs", flush=True)

    summary = dict(
        anchor_pairs=len(rows), binary_pairs=sum(r["binary"] for r in rows),
        fractional_pairs=sum(not r["binary"] for r in rows),
        max_absolute_primal_dual_gap=max(r["absolute_primal_dual_gap"] for r in rows),
        min_signed_primal_minus_dual=min(r["signed_primal_minus_dual"] for r in rows),
        max_primal_residual=max(max(r["primal_residuals"].values()) for r in rows),
        max_normalized_dual_residual=max(max(r["dual"]["normalized_residuals"].values()) for r in rows),
        max_raw_dual_residual=max(max(r["dual"]["raw_residuals"].values()) for r in rows),
        max_fraction_difference=max(r["dual"]["fraction_difference"] for r in rows),
        max_binary_original_gate_difference=max(r.get("binary_original_gate_difference",0) for r in rows),
        max_original_primal_residual=max(max(r.get("original_residuals",{"absent":0}).values()) for r in rows),
        min_tau=min(r["dual"]["tau"] for r in rows),
        max_envelope_violation=max(r.get("deployment",{}).get("envelope_violation",0) for r in rows),
        max_coupling_risk_violation=max(r.get("deployment",{}).get("coupling_risk_violation",0) for r in rows),
        max_coupling_budget_violation=max(r.get("deployment",{}).get("coupling_budget_violation",0) for r in rows),
        max_unselected_coordinate_invariance_error=max(r.get("deployment",{}).get("unselected_coordinate_invariance_error",0) for r in rows),
        max_deployment_repair_increase=max(r.get("deployment",{}).get("repair_increase",0) for r in rows),
        space_points_checked=sum(r.get("deployment",{}).get("point_count",0) for r in rows),
        explicit_couplings_checked=sum(r.get("deployment",{}).get("coupling_count",0) for r in rows),
        elapsed_seconds=perf_counter()-started,
    )
    summary["passed"] = all(summary[name] <= ACCEPTANCE for name in (
        "max_absolute_primal_dual_gap", "max_primal_residual",
        "max_normalized_dual_residual", "max_raw_dual_residual",
        "max_fraction_difference", "max_binary_original_gate_difference",
        "max_original_primal_residual", "max_envelope_violation",
        "max_coupling_risk_violation", "max_coupling_budget_violation",
        "max_unselected_coordinate_invariance_error"))
    record = dict(
        purpose="Independent numerical verification of generalized theory, not a research experiment",
        source=Path(__file__).name,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cvxpy_version=cp.__version__, clarabel_version=clarabel.__version__,
        numpy_version=np.__version__, solver_settings=SOLVER,
        acceptance_tolerance=ACCEPTANCE, deployment_intercept_margin=DEPLOYMENT_MARGIN,
        mathematical_models=dict(
            primal="min eta: J<=eta, Phi(sigma*u)<=a+d*w, 0<=w<=lambda, theta*w_j<=eta*z_j",
            dual="max sum Psi: row(Pi_k)=tau*q_k, sum_j (A_kj-theta_k*mu_j)_+<=tau*theta_k, tau+mu*z=1",
            weights="Each class separately sums to one; total source weight is two",
            normalized_scaling="Pi_raw=Pi_bar/tau, mu_raw=mu_bar/tau, H_raw/(1+mu_raw*z)=H_bar",
            deployment="Root's deterministic first-argmin endpoint convex combination; it need not interpolate pool scores",
            numerical_limits="Ordinary floating-point conic checks and conservative primal repair; no directed-rounding certification",
        ), fixtures=DATA, summary=summary, records=rows)
    destination = Path(__file__).with_name("general_loss_verification.json")
    destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    print(f"RESULT_JSON={destination}",flush=True)
    if not summary["passed"]:
        raise SystemExit("Numerical acceptance failed; inspect retained details")


if __name__ == "__main__":
    main()
