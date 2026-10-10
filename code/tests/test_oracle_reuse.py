"""Certificate transfer and actual persistent-model reuse.

The overlap reference is an independent transport LP with only source
marginals and original total-distance budgets; it has no risk gate.
"""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

import robusthw.oracle as oracle_module
from robusthw import GateOracle, Problem


def original_overlap(p, z):
    sizes = [p.n0 * p.n, p.n1 * p.n]
    starts = [0, sizes[0]]
    hstart, count = sum(sizes), sum(sizes) + p.n
    eq, rhs, ub, bounds = [], [], [], []
    for k, (delta, q) in enumerate(zip(p.deltas, p.weights)):
        for i in range(len(q)):
            row = np.zeros(count)
            row[starts[k] + i * p.n:starts[k] + (i + 1) * p.n] = 1
            eq.append(row)
            rhs.append(q[i])
        row = np.zeros(count)
        row[starts[k]:starts[k] + sizes[k]] = (delta @ z).ravel()
        ub.append(row)
        bounds.append(p.theta[k])
        for target in range(p.n):
            row = np.zeros(count)
            row[hstart + target] = 1
            row[starts[k] + target:starts[k] + sizes[k]:p.n] = -1
            ub.append(row)
            bounds.append(0)
    result = linprog(np.r_[np.zeros(hstart), -np.ones(p.n)],
                     A_ub=ub, b_ub=bounds, A_eq=eq, b_eq=rhs, method="highs")
    assert result.success
    return -result.fun


def fixture_problem(loss="overlap"):
    rng = np.random.default_rng(451)
    x0, x1 = rng.normal(scale=.25, size=(3, 3)), rng.normal(scale=.25, size=(2, 3))
    x1[:, 0] += 2
    x1[:, 1] += .8
    return Problem(x0, x1, 1, .13, .21, loss, q0=[.2, .3, .5], q1=[.65, .35])


@pytest.mark.parametrize("loss", ["overlap", "logistic"])
def test_transferred_certificate_is_an_original_feasible_test(loss):
    if loss == "logistic":
        pytest.importorskip("cvxpy")
    p = fixture_problem(loss)
    oracle = GateOracle(p)
    original = oracle.solve([1, 0, 0]).test
    z = np.array([0, 1, 0])
    transferred = oracle.reprice(original, z)
    assert transferred.support == (1,)
    assert not transferred.pool_losses.flags.writeable
    assert np.array_equal(transferred.pool_losses, original.pool_losses)
    assert np.array_equal(transferred.multipliers, original.multipliers)
    assert "pool_losses" not in transferred.as_dict()
    for k, delta in enumerate(p.deltas):
        residual = (transferred.pool_losses[k][None, :] - transferred.potentials[k][:, None]
                    - transferred.multipliers[k] * (delta @ z))
        assert residual.max() <= 1e-12
    if loss == "overlap":
        assert transferred.upper >= original_overlap(p, z) - 1e-9
    same = oracle.reprice(original, [1, 0, 0])
    assert abs(same.upper - original.upper) < 5e-10

    # A source-wise translation defines a complete budget-feasible coupling.
    rng = np.random.default_rng(881)
    for _ in range(12):
        risk = 0
        for k, (samples, q) in enumerate(zip(p.samples, p.weights)):
            shift = rng.normal(size=samples.shape)
            shift *= p.theta[k] / np.max(np.abs(shift).sum(1))
            assert q @ np.abs(shift).sum(1) <= p.theta[k] + 1e-12
            probability = transferred.prob_class0(samples + shift)
            losses = (1 - probability if k == 0 else probability) if loss == "overlap" else (
                -np.log(probability) if k == 0 else -np.log1p(-probability))
            risk += q @ losses
        assert risk <= transferred.upper + 1e-8
    query = rng.normal(size=(40, p.p)) * 15
    moved = query.copy()
    moved[:, [0, 2]] += 1000
    assert np.allclose(transferred.prob_class0(query), transferred.prob_class0(moved))


def test_reprice_rejects_fractional_or_missing_history():
    p = fixture_problem()
    oracle = GateOracle(p)
    test = oracle.solve([1, 0, 0]).test
    with pytest.raises(ValueError):
        oracle.reprice(test, [.5, .5, 0])
    test.pool_losses = None
    with pytest.raises(ValueError):
        oracle.reprice(test, [0, 1, 0])


@pytest.mark.parametrize("loss", ["overlap", "logistic"])
def test_repaired_risk_certificates_obey_geometric_price_bounds(loss):
    if loss == "logistic":
        pytest.importorskip("cvxpy")
    p = fixture_problem(loss)
    oracle = GateOracle(p)
    for z in ([1, 0, 0], [.2, .7, .3], [0, 0, 0]):
        result = oracle.solve(z)
        certificate = result.certificate
        mu, tau, beta = certificate["mu"], certificate["tau"], certificate["beta"]
        assert np.max(mu - tau * oracle.mu_bounds) < 1e-9
        assert np.max(beta.sum(1) - tau * p.theta) < 1e-9
        assert certificate["coordinate_residual"] < 1e-9
        assert certificate["normalization_residual"] < 1e-9
        assert certificate["mu_bound_residual"] < 1e-9
        if loss == "overlap":
            for support in combinations(range(p.p), p.budget):
                other = p.mask(support)
                assert result.cut.value(other) <= original_overlap(p, other) + 1e-8


@pytest.mark.parametrize("fixed", [False, True])
def test_logistic_parameter_models_and_solver_workspaces_are_reused(fixed):
    pytest.importorskip("cvxpy")
    p = fixture_problem("logistic")
    oracle = GateOracle(p)
    points = ([1, 0, 0], [0, 1, 0]) if fixed else ([.2, .4, .3], [.7, .1, .5])
    first = oracle.solve(points[0])
    primal = oracle._cone_models[("primal", fixed)]
    dual = oracle._cone_models[("dual", "risk")]
    first_models = (primal["model"], dual["model"])
    first_solvers = tuple(model._solver_cache["CLARABEL"] for model in first_models)
    previous_score = primal["u"].value.copy()
    assert np.isfinite(previous_score).all()
    second = oracle.solve(points[1])
    assert tuple(oracle._cone_models[key]["model"] for key in
                 (("primal", fixed), ("dual", "risk"))) == first_models
    assert all(model.is_dpp() for model in first_models)
    assert all(model._solver_cache["CLARABEL"] is solver
               for model, solver in zip(first_models, first_solvers))
    stats = second.certificate["model_reuse"]
    assert stats == dict(models_built=2, model_reuses=2,
                        warm_start_attempts=2, solver_workspace_reuses=2)
    assert first.certificate["model_reuse"]["model_reuses"] == 0
    assert second.certificate["primal"]["model_reused"]
    assert second.certificate["dual"]["solver_workspace_reused"]
    assert second.upper - second.value < 2e-6
    # A fresh full solve still supplies the same optimum, rather than a
    # restriction to the old model's scores or transport plan.
    fresh = GateOracle(p).solve(points[1])
    assert abs(second.value - fresh.value) < 2e-6
    assert abs(second.upper - fresh.upper) < 2e-6


def test_constant_fallback_has_repriceable_exact_pool_losses():
    p = fixture_problem("logistic")
    oracle = GateOracle(p)
    result = oracle.solve([1, 0, 0], time_limit=0)
    assert result.status == "time_limit"
    transferred = oracle.reprice(result.test, [0, 0, 1])
    assert transferred.upper >= p.B
    assert transferred.upper - p.B < 1e-8
    assert np.allclose(transferred.prob_class0(p.points), .5)


def test_lp_refinement_changes_actual_options_and_respects_highs_floor(monkeypatch):
    options = []

    def no_solution(*args, **kwargs):
        options.append(kwargs["options"])
        return SimpleNamespace(status=1, message="time limit", fun=None, nit=0, x=None)

    monkeypatch.setattr(oracle_module, "linprog", no_solution)
    oracle = GateOracle(fixture_problem())
    for tolerance in (1e-8, 1e-10, 1e-13):
        oracle.tolerance = tolerance
        oracle._primal_lp(np.array([1., 0, 0]), None, True)
        oracle._dual_lp(np.array([1., 0, 0]), None)
    expected = [1e-9, 1e-9, 1e-10, 1e-10, 1e-10, 1e-10]
    assert [item["primal_feasibility_tolerance"] for item in options] == expected
    assert [item["dual_feasibility_tolerance"] for item in options] == expected


def test_cone_refinement_changes_actual_options_and_documents_floor():
    options = []

    class Model:
        status, value = "optimal", .5
        solver_stats = SimpleNamespace(solve_time=0.)

        def solve(self, **kwargs):
            options.append(kwargs)

    oracle = GateOracle(fixture_problem("logistic"))
    for tolerance in (1e-8, 1e-11, 1e-15):
        oracle.tolerance = tolerance
        info = oracle._cvx_solve(Model(), None, persistent=True, model_reused=True)
        assert info["solver_tolerance"] == max(1e-12, min(1e-9, tolerance))
    for key in ("tol_gap_abs", "tol_gap_rel", "tol_feas"):
        assert [item[key] for item in options] == [1e-9, 1e-11, 1e-12]
    assert all(item["warm_start"] for item in options)
