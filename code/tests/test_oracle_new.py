"""Independent original-budget checks for the rewritten open oracle."""

from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from robusthw import GateOracle, Problem


def original_transport_value(problem, mask):
    """Independent LP: adversary source marginals and two total-cost budgets.

    There are no gate variables, multiplier bounds, normalized masses, beta,
    or mu in this reference. It directly maximizes transported overlap.
    """
    n = problem.n
    gamma_sizes = [problem.n0 * n, problem.n1 * n]
    starts = [0, gamma_sizes[0]]
    hstart = sum(gamma_sizes)
    count = hstart + n
    c = np.r_[np.zeros(hstart), -np.ones(n)]
    eq, beq, ub, bub = [], [], [], []
    targets = np.vstack((problem.x0, problem.x1))
    selected = np.flatnonzero(mask)
    for k, (samples, q) in enumerate(zip(problem.samples, problem.weights)):
        distance = np.abs(samples[:, None, selected] - targets[None, :, selected]).sum(2)
        for i in range(len(samples)):
            row = np.zeros(count)
            row[starts[k] + i * n:starts[k] + (i + 1) * n] = 1
            eq.append(row)
            beq.append(q[i])
        row = np.zeros(count)
        row[starts[k]:starts[k] + gamma_sizes[k]] = distance.ravel()
        ub.append(row)
        bub.append(problem.theta[k])
        for m in range(n):
            row = np.zeros(count)
            row[hstart + m] = 1
            row[starts[k] + m:starts[k] + gamma_sizes[k]:n] = -1
            ub.append(row)
            bub.append(0)
    result = linprog(c, A_ub=ub, b_ub=bub, A_eq=eq, b_eq=beq, method="highs")
    assert result.success, result.message
    return -result.fun


def masks(dimension, budget):
    for indices in combinations(range(dimension), budget):
        z = np.zeros(dimension)
        z[list(indices)] = 1
        yield z


@pytest.mark.parametrize("gate", ["box", "triangle", "risk"])
def test_one_coordinate_exact_curve(gate):
    p = Problem([[0]], [[1]], 1, .2, .2)
    oracle = GateOracle(p, gate)
    for z in [0., .1, .2, .4, .7, 1.]:
        expected = {"box": max(.4, 1 - 3 * z),
                    "triangle": max(.4, 1 - 1.5 * z),
                    "risk": 1 / (1 + 1.5 * z)}[gate]
        result = oracle.solve([z])
        assert result.value <= expected + 1e-8
        assert result.upper >= expected - 1e-8
        assert result.upper - result.value < 1e-7
        assert result.status == "optimal"
        assert bool(result.test is not None) == (z in (0, 1))


@pytest.mark.parametrize("seed", range(20))
def test_random_integer_exactness_cuts_and_fractional_order(seed):
    rng = np.random.default_rng(seed + 4300)
    n0, n1, d = 3 + seed % 2, 3 + (seed // 2) % 2, 6
    q0, q1 = rng.uniform(.1, 1, n0), rng.uniform(.1, 1, n1)
    p = Problem(rng.normal(size=(n0, d)), rng.normal(size=(n1, d)), 1 + seed % 3,
                .10 + .02 * (seed % 4), .21 + .03 * (seed % 3),
                q0=q0 / q0.sum(), q1=q1 / q1.sum())
    supports = list(masks(d, p.budget))
    reference = [original_transport_value(p, z) for z in supports]
    oracles = [GateOracle(p, gate) for gate in ("box", "triangle", "risk")]
    for oracle in oracles:
        for z, expected in zip(supports, reference):
            result = oracle.solve(z)
            gate_value, diagnostics = oracle.gate_primal(z)
            assert diagnostics["status"] == "optimal"
            assert abs(gate_value - expected) < 1e-7
            assert abs(result.value - expected) < 1e-7
            assert abs(result.upper - expected) < 1e-7
            assert result.value <= result.upper + 1e-9
            assert abs(result.intercept + result.gradient @ z - result.value) < 1e-10
            for other, truth in zip(supports, reference):
                assert result.intercept + result.gradient @ other <= truth + 1e-7
            for key in ("row_residual", "budget_residual", "coordinate_residual",
                        "normalization_residual"):
                assert result.certificate[key] < 1e-8
    for _ in range(3):
        fractional = rng.uniform(.01, .99, d)
        results = [o.solve(fractional) for o in oracles]
        assert all(r.upper - r.value < 1e-7 for r in results)
        assert results[0].value <= results[1].upper + 1e-8
        assert results[1].value <= results[2].upper + 1e-8
        other = rng.uniform(0, 1, d)
        for oracle, result in zip(oracles, results):
            truth = oracle.gate_primal(other)[0]
            assert result.intercept + result.gradient @ other <= truth + 1e-7


@pytest.mark.parametrize("gate", ["box", "triangle", "risk"])
@pytest.mark.parametrize("core", [False, True])
def test_duplicates_unequal_classes_weights_and_budget_endpoints(gate, core):
    x0 = np.array([[0., 2.], [0., 2.], [10., -4.]])
    x1 = np.array([[0., 2.], [2., 2.]])
    for budget in [0, 1, 2]:
        p = Problem(x0, x1, budget, .15, .3, q0=[.2, .8, 0], q1=[.7, .3])
        assert p.n0 == 2
        assert p.p == 2
        oracle = GateOracle(p, gate, core_point=core)
        for z in masks(p.p, budget):
            result = oracle.solve(z)
            expected = original_transport_value(p, z)
            assert abs(result.value - expected) < 1e-7
            assert result.upper - result.value < 1e-7
            if budget == 0:
                assert abs(result.upper - 1) < 1e-8
                assert np.allclose(result.test.prob_class0([[50, -200], [-6, 8]]), .5)


@pytest.mark.parametrize("loss", ["overlap", "logistic"])
def test_full_space_envelope_and_source_atom_transport(loss):
    if loss == "logistic":
        pytest.importorskip("cvxpy")
    rng = np.random.default_rng(771)
    p = Problem(rng.normal(size=(4, 3)), rng.normal(size=(3, 3)) + .6, 2, .2, .3, loss)
    result = GateOracle(p).solve([1, 0, 1])
    test = result.test
    assert test.support == (0, 2)
    queries = rng.normal(size=(200, 3)) * 5
    lo, hi = test.interval(queries)
    assert np.all(lo <= hi + 1e-9)
    probability = test.prob_class0(queries)
    assert np.all(probability >= lo - 1e-10)
    assert np.all(probability <= hi + 1e-10)
    changed = queries.copy()
    changed[:, 1] += 10000
    assert np.allclose(probability, test.prob_class0(changed), atol=1e-12)
    assert np.isin(test.predict(queries), [0, 1]).all()
    for _ in range(20):
        measured_risk = 0
        for k, (source, q) in enumerate(zip(p.samples, p.weights)):
            direction = rng.normal(size=3)
            direction *= p.theta[k] / np.abs(direction).sum()
            shifted = source + direction
            prob = test.prob_class0(shifted)
            losses = (1 - prob if k == 0 else prob) if loss == "overlap" else (
                -np.log(prob) if k == 0 else -np.log1p(-prob))
            measured_risk += q @ losses
        assert measured_risk <= test.primal_objective + 1e-7
    record = test.as_dict()
    assert "problem" not in record and "x0" not in record
    with pytest.raises(ValueError):
        test.prob_class0(np.zeros((2, 2, 3)))


@pytest.mark.parametrize("gate", ["box", "triangle", "risk"])
def test_zero_time_is_a_bound_and_status_not_optimal(gate):
    p = Problem([[0]], [[1]], 1, .2, .2)
    result = GateOracle(p, gate).solve([1], time_limit=0)
    assert result.value == 0 and result.upper == 1
    assert result.status == "time_limit"
    assert np.all(result.gradient == 0)
    assert result.test is not None
    assert np.allclose(result.test.prob_class0([[0], [1]]), .5)


@pytest.mark.parametrize("gate", ["box", "triangle", "risk"])
def test_logistic_analytic_and_cross_support(gate):
    pytest.importorskip("cvxpy")
    p = Problem([[0]], [[1]], 1, .2, .2, "logistic")
    expected = -2 * (.8 * np.log(.8) + .2 * np.log(.2))
    result = GateOracle(p, gate).solve([1])
    assert abs(result.value - expected) < 2e-7
    assert abs(result.upper - expected) < 2e-7
    rng = np.random.default_rng(94)
    p = Problem(rng.normal(size=(2, 3)), rng.normal(size=(3, 3)), 1, .2, .3, "logistic")
    oracle = GateOracle(p, gate)
    supports = list(masks(3, 1))
    outcomes = [oracle.solve(z) for z in supports]
    for z, result in zip(supports, outcomes):
        gate_value, info = oracle.gate_primal(z)
        assert info["status"] in ("optimal", "optimal_inaccurate")
        assert abs(gate_value - result.upper) < 2e-6
        assert result.upper - result.value < 2e-6
        for other, expected_result in zip(supports, outcomes):
            assert result.intercept + result.gradient @ other <= expected_result.upper + 1e-7


@pytest.mark.parametrize("name,value", [("theta0", 0), ("theta1", -1), ("theta0", np.nan)])
def test_input_radii_are_positive(name, value):
    arguments = dict(x0=[[0]], x1=[[1]], budget=1, theta0=.2, theta1=.2)
    arguments[name] = value
    with pytest.raises(ValueError):
        Problem(**arguments)
