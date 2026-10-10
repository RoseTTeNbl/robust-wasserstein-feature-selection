"""Transparent selectors; Wang's paper objective is implemented independently."""

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression


def coordinate_contrasts(x0, x1):
    """Eq. (13), with selection-only median cross-squared-distance bandwidths."""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    if x0.shape != x1.shape or len(x0) < 2:
        raise ValueError("Wang's paired estimator requires equal class sizes >= 2")
    xx = (x0[:, None, :] - x0[None, :, :]) ** 2
    yy = (x1[:, None, :] - x1[None, :, :]) ** 2
    xy = (x0[:, None, :] - x1[None, :, :]) ** 2
    bandwidth2 = np.maximum(np.median(xy, axis=(0, 1)), 1e-12)
    kx, ky, kxy = (np.exp(-distance / (2 * bandwidth2)) for distance in (xx, yy, xy))
    return kx + ky - kxy - kxy.transpose(1, 0, 2), bandwidth2


def wang_coefficients(x0, x1, regularization=0.1):
    """Return paper-correct negative-variance A, linear mean t, bandwidths.

    The author's revision source instead returns positive-variance A.  Its
    coefficients are also exposed for a source-comparison fixture.  Neither
    source code nor its deprecated dependencies are imported or redistributed.
    """
    h, bandwidth2 = coordinate_contrasts(x0, x1)
    n = len(x0)
    diagonal = np.einsum("iij->j", h)
    t = (h.sum(axis=(0, 1)) - diagonal) / (n * (n - 1))
    row_means = h.mean(axis=1)
    centered = row_means - row_means.mean(axis=0)
    variance_matrix = 4 * centered.T @ centered / n
    a = -float(regularization) * variance_matrix
    revision = -a
    eigmin = np.linalg.eigvalsh(revision)[0]
    if eigmin <= 0:
        revision = revision - eigmin * np.eye(x0.shape[1])
    return a, t, bandwidth2, revision


def trust_region_maximum(a, t):
    """Global max of x'Ax+t'x on the unit sphere, including the hard case.

    Stationarity is (lambda I-A)x=t/2.  A root of the secular norm equation
    above lambda_max(A), or a hard-case top-eigenspace component, gives the
    global solution.  The shift lambda I-A is positive semidefinite.
    """
    a, t = np.asarray(a, float), np.asarray(t, float).reshape(-1)
    if a.shape != (len(t), len(t)) or not len(t):
        raise ValueError("A must be a nonempty square matrix aligned with t")
    eigenvalues, vectors = np.linalg.eigh((a + a.T) / 2)
    gaps = eigenvalues[-1] - eigenvalues
    q = vectors.T @ (t / 2)
    scale = max(1.0, np.linalg.norm(a, ord=2), np.linalg.norm(t))
    top = gaps <= 1e-12 * scale
    lower = ~top
    candidate = np.zeros_like(q)
    candidate[lower] = q[lower] / gaps[lower]
    norm2 = float(candidate @ candidate)
    if np.linalg.norm(q[top]) <= 1e-13 * scale and norm2 <= 1:
        candidate[np.flatnonzero(top)[0]] = np.sqrt(max(0, 1 - norm2))
    else:
        def secular(shift):
            if shift == 0 and np.any(q[top] != 0):
                return np.inf
            denominator = gaps + shift
            ratios = np.divide(q, denominator, out=np.zeros_like(q), where=denominator != 0)
            return float(ratios @ ratios - 1)

        upper = max(scale, np.linalg.norm(q))
        while secular(upper) > 0:
            upper *= 2
        shift = brentq(secular, 0, upper, xtol=1e-14 * scale, rtol=1e-14)
        candidate = q / (gaps + shift)
    x = vectors @ candidate
    x /= np.linalg.norm(x)
    return x, float(x @ a @ x + t @ x)


def sparse_trust_region(a, t, budget, *, exact_dimension=12):
    """Exact support enumeration for small D; forward/swap for larger D."""
    dimension = len(t)
    if not 1 <= budget <= dimension:
        raise ValueError("Wang's unit-norm selector requires a positive valid budget")
    cache = {}

    def fit(support):
        support = tuple(sorted(support))
        if support not in cache:
            cache[support] = trust_region_maximum(a[np.ix_(support, support)], t[list(support)])
        return cache[support]

    if dimension <= exact_dimension:
        support = max(combinations(range(dimension), budget), key=lambda s: fit(s)[1])
        status = "exact_support_enumeration"
    else:
        support = ()
        for _ in range(budget):
            options = [tuple(sorted((*support, j))) for j in range(dimension) if j not in support]
            support = max(options, key=lambda s: fit(s)[1])
        for _ in range(10):
            options = [tuple(sorted((set(support) - {old}) | {new}))
                       for old in support for new in range(dimension) if new not in support]
            candidate = max(options, key=lambda s: fit(s)[1]) if options else support
            if fit(candidate)[1] <= fit(support)[1] + 1e-10:
                break
            support = candidate
        status = "approximate_forward_swap"
    weights = np.zeros(dimension)
    weights[list(support)] = fit(support)[0]
    # Zero coefficient entries need not be counted as observed variables.
    active = tuple(int(j) for j in np.flatnonzero(np.abs(weights) > 1e-10))
    return active, weights, fit(support)[1], status, len(cache)


def screen_mmd(x0, x1, budget):
    h, _ = coordinate_contrasts(x0, x1)
    n = len(x0)
    values = (h.sum(axis=(0, 1)) - np.einsum("iij->j", h)) / (n * (n - 1))
    order = np.lexsort((np.arange(len(values)), -values))
    return tuple(sorted(int(j) for j in order[:budget]))


def select(method, x0, x1, budget, seed=0, *, wang_regularization=0.1):
    """A common result schema for non-robust selectors."""
    started = perf_counter()
    p = x0.shape[1]
    if not 0 <= budget <= p:
        raise ValueError("budget must lie between zero and dimension")
    diagnostics = {}
    weights = None
    if budget == 0:
        support, status = (), "empty_budget"
    elif method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 81003]))
        support = tuple(sorted(int(j) for j in rng.choice(p, budget, replace=False)))
        status = "random"
    elif method == "mmd":
        support, status = screen_mmd(x0, x1, budget), "marginal_screening"
    elif method == "l1":
        x = np.vstack((x0, x1))
        y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
        model = LogisticRegression(l1_ratio=1, solver="liblinear", C=1,
                                   random_state=seed, max_iter=2000).fit(x, y)
        magnitudes = np.abs(model.coef_[0])
        variance = x.var(axis=0)
        order = np.lexsort((np.arange(p), -variance, -magnitudes))
        support = tuple(sorted(int(j) for j in order[:budget]))
        status = "l1_top_budget"
        diagnostics = {"coefficient_magnitudes": magnitudes.tolist(), "C": 1.0}
    elif method == "wang":
        a, t, widths, _ = wang_coefficients(x0, x1, wang_regularization)
        support, weights, value, status, calls = sparse_trust_region(a, t, budget)
        diagnostics = {"objective": value, "subproblems": calls,
                       "coordinate_bandwidth_squared": widths.tolist(),
                       "regularization": wang_regularization,
                       "objective_sign": "mean_minus_variance"}
    else:
        raise ValueError(f"Unknown baseline selector: {method}")
    return {"support": support, "status": status, "seconds": perf_counter() - started,
            "weights": weights, "diagnostics": diagnostics}
