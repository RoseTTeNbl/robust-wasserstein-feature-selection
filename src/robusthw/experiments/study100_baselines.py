"""Small selection baselines; no Wasserstein fitting or experiment I/O here."""

from inspect import signature

import numpy as np
from sklearn.linear_model import LogisticRegression


def _samples(x0, x1, budget):
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    if (
        x0.ndim != 2
        or x1.ndim != 2
        or x0.shape[1] != x1.shape[1]
        or min(len(x0), len(x1)) < 2
        or not np.isfinite(x0).all()
        or not np.isfinite(x1).all()
    ):
        raise ValueError("Need two finite class matrices with at least two observations")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= x0.shape[1]:
        raise ValueError("budget must be an integer between 1 and the dimension")
    return x0, x1


def fisher_ranking(x0, x1):
    """Squared mean separation divided by within-class variance; largest first."""
    x0, x1 = _samples(x0, x1, 1)
    # Fisher score T_m=(mean_0[m]-mean_1[m])^2/(var_0[m]+var_1[m]).
    # Unbiased within-class variances; denominator floored at 1e-12.
    variance = x0.var(axis=0, ddof=1) + x1.var(axis=0, ddof=1)
    scores = (x0.mean(axis=0) - x1.mean(axis=0)) ** 2 / np.maximum(variance, 1e-12)
    # Descending score, with coordinate index as the deterministic tie-breaker.
    return tuple(int(j) for j in np.lexsort((np.arange(x0.shape[1]), -scores)))


def random_support(dimension, budget, seed):
    # Uniform size-budget subset, sampled without replacement.
    return tuple(
        sorted(int(j) for j in np.random.default_rng(seed).choice(dimension, budget, replace=False))
    )


def _logistic(penalty, C, seed):
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be positive")
    parameter = signature(LogisticRegression).parameters.get("penalty")
    settings = (
        {"penalty": penalty}
        if parameter is not None and parameter.default != "deprecated"
        else {"l1_ratio": 1 if penalty == "l1" else 0}
    )
    # Balanced weights v_i=N/(2*n_k); class totals N/2.
    # Objective: penalty(theta)+C*weighted_logistic_loss.
    return LogisticRegression(
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=10000,
        random_state=seed,
        **settings,
    )


def l1_support(x0, x1, budget, C=1.0, seed=0):
    """Rank by absolute L1-logistic coefficients and retain exactly d coordinates."""
    x0, x1 = _samples(x0, x1, budget)
    labels = np.r_[np.zeros(len(x0), int), np.ones(len(x1), int)]
    # min_theta ||theta||_1+C*sum_i v_i*log(1+exp(-t_i*theta^T x_tilde_i)).
    # t_i=2*label_i-1, theta=(b,beta), x_tilde=(1,x); liblinear penalizes b too.
    model = _logistic("l1", C, seed).fit(np.vstack((x0, x1)), labels)
    coefficients = model.coef_[0]
    # Top-budget |beta_m| ranking; L1 fitting does not impose exact cardinality.
    # Zero-coefficient selections are retained in diagnostics.
    order = np.lexsort((np.arange(len(coefficients)), -np.abs(coefficients)))
    support = tuple(sorted(int(j) for j in order[:budget]))
    return support, {
        "C": float(C),
        "nonzero_coefficients": int(np.count_nonzero(coefficients)),
        "selected_zero_coefficients": int(np.count_nonzero(coefficients[list(support)] == 0)),
        "iterations": int(model.n_iter_.max()),
    }


def fit_logistic(x0, x1, support, C=1.0, seed=0):
    """Ordinary L2 logistic on the same selected coordinates, equal class weight."""
    support = tuple(support)
    x0, x1 = _samples(x0, x1, len(support))
    labels = np.r_[np.zeros(len(x0), int), np.ones(len(x1), int)]
    # Fixed-S refit: min_theta ||theta||_2^2/2
    #                  +C*sum_i v_i*log(1+exp(-t_i*theta^T (1,x_i[S]))).
    return _logistic("l2", C, seed).fit(np.vstack((x0, x1))[:, support], labels)


def logistic_score(model, x, support):
    """The theory's positive score means class 0; sklearn's means class 1."""
    # g(x[S])=-(beta^T x[S]+b): positive scores indicate class 0.
    return -np.asarray(model.decision_function(np.asarray(x)[:, tuple(support)]), float)
