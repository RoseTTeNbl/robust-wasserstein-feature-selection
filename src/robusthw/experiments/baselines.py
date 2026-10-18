"""Five explicit selection criteria, all using training observations only."""

from inspect import signature
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score


class SelectionTimeout(TimeoutError):
    """Retain the last complete selection stage when the shared budget expires."""

    def __init__(self, partial_support=()):
        self.partial_support = tuple(sorted(partial_support))
        super().__init__("Feature-selection time budget exhausted")


def mmd_squared(x0, x1):
    """Unbiased squared Gaussian MMD; the median positive distance fixes h."""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    if min(len(x0), len(x1)) < 2 or x0.ndim != 2 or x1.shape[1] != x0.shape[1]:
        raise ValueError("MMD requires aligned samples with at least two rows per class")
    if x0.shape[1] == 0:
        return 0.0
    distances = pdist(np.vstack((x0, x1)))
    positive = distances[distances > 0]
    if not len(positive):
        return 0.0
    bandwidth = float(np.median(positive))
    k00 = np.exp(-cdist(x0, x0, "sqeuclidean") / (2 * bandwidth**2))
    k11 = np.exp(-cdist(x1, x1, "sqeuclidean") / (2 * bandwidth**2))
    k01 = np.exp(-cdist(x0, x1, "sqeuclidean") / (2 * bandwidth**2))
    n0, n1 = len(x0), len(x1)
    return float(
        (k00.sum() - np.trace(k00)) / (n0 * (n0 - 1))
        + (k11.sum() - np.trace(k11)) / (n1 * (n1 - 1))
        - 2 * k01.mean()
    )


def marginal_mmd_scores(x0, x1):
    return np.array([mmd_squared(x0[:, [j]], x1[:, [j]]) for j in range(x0.shape[1])])


def rank_features(x0, x1):
    scores = marginal_mmd_scores(x0, x1)
    return tuple(int(j) for j in np.lexsort((np.arange(len(scores)), -scores)))


def _mrmr(x0, x1, budget, check):
    pooled = np.vstack((x0, x1))
    labels = np.r_[np.zeros(len(x0), int), np.ones(len(x1), int)]
    # Quantile ties are merged; labels are never used to set bin boundaries.
    discrete = np.column_stack(
        [
            np.searchsorted(
                np.unique(np.quantile(pooled[:, j], [1 / 3, 2 / 3])), pooled[:, j], side="right"
            )
            for j in range(pooled.shape[1])
        ]
    )
    relevance = np.array(
        [mutual_info_score(discrete[:, j], labels) for j in range(pooled.shape[1])]
    )
    redundancy = {}
    chosen = []
    for _ in range(budget):
        values = []
        for j in range(pooled.shape[1]):
            check(chosen)
            if j in chosen:
                continue
            penalties = []
            for old in chosen:
                pair = tuple(sorted((old, j)))
                if pair not in redundancy:
                    redundancy[pair] = mutual_info_score(discrete[:, old], discrete[:, j])
                penalties.append(redundancy[pair])
            values.append((relevance[j] - (np.mean(penalties) if penalties else 0.0), -j, j))
        chosen.append(max(values)[2])
    check(chosen)
    return tuple(sorted(chosen))


def _joint_mmd(x0, x1, budget, swap_limit, min_improvement, check):
    cache = {}
    selected = ()

    def value(support):
        check(selected)
        key = tuple(sorted(support))
        if key not in cache:
            cache[key] = mmd_squared(x0[:, key], x1[:, key])
        return cache[key]

    for _ in range(budget):
        options = [tuple(sorted((*selected, j))) for j in range(x0.shape[1]) if j not in selected]
        selected = min(options, key=lambda support: (-value(support), support))
    for _ in range(swap_limit):
        options = {
            tuple(sorted((set(selected) - {old}) | {new}))
            for old in selected
            for new in range(x0.shape[1])
            if new not in selected
        }
        if not options:
            break
        candidate = min(options, key=lambda support: (-value(support), support))
        if value(candidate) < value(selected) + min_improvement:
            break
        selected = candidate
    check(selected)
    return selected


def select_features(
    method, x0, x1, budget, seed=0, C=1, swap_limit=10, min_improvement=1e-4, time_limit=None
):
    started = perf_counter()

    def check(partial=()):
        if time_limit is not None and perf_counter() - started >= time_limit:
            raise SelectionTimeout(partial)

    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1]:
        raise ValueError("Both classes must have the same feature dimension")
    if not 0 <= budget <= x0.shape[1] or swap_limit < 0 or min_improvement <= 0:
        raise ValueError("Invalid feature budget or exchange settings")
    if not np.isfinite(x0).all() or not np.isfinite(x1).all():
        raise ValueError("Nonfinite baseline observations")
    if budget == 0:
        return ()
    check()
    if method == "marginal_mmd":
        scores = []
        for j in range(x0.shape[1]):
            check()
            scores.append(mmd_squared(x0[:, [j]], x1[:, [j]]))
        check()
        return tuple(
            sorted(int(j) for j in np.lexsort((np.arange(len(scores)), -np.array(scores)))[:budget])
        )
    if method == "random":
        return tuple(
            sorted(
                int(j)
                for j in np.random.default_rng(seed).choice(x0.shape[1], budget, replace=False)
            )
        )
    if method == "mrmr":
        return _mrmr(x0, x1, budget, check)
    if method == "joint_mmd":
        return _joint_mmd(x0, x1, budget, swap_limit, min_improvement, check)
    if method == "l1_logistic":
        if C <= 0:
            raise ValueError("Logistic C must be positive")
        labels = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
        penalty = signature(LogisticRegression).parameters.get("penalty")
        settings = (
            {"penalty": "l1"}
            if penalty is not None and penalty.default != "deprecated"
            else {"l1_ratio": 1}
        )
        model = LogisticRegression(
            solver="liblinear", C=C, random_state=seed, max_iter=10000, **settings
        ).fit(np.vstack((x0, x1)), labels)
        check()
        order = np.lexsort((np.arange(x0.shape[1]), -np.abs(model.coef_[0])))
        return tuple(sorted(int(j) for j in order[:budget]))
    raise ValueError(f"Unknown selector: {method}")
