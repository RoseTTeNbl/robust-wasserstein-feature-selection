"""Calibrated equality tests and separate individual-prediction metrics."""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import roc_auc_score


def rbf_bandwidth_squared(x):
    x = np.asarray(x, float)
    if x.shape[1] == 0:
        return 1.0
    return max(float(np.median(pdist(x, metric="sqeuclidean"))), 1e-12)


def rbf_gram(x, bandwidth2):
    return np.exp(-cdist(x, x, metric="sqeuclidean") / (2 * bandwidth2))


def permutation_mmd(gram, n0, *, permutations=199, seed=0):
    """Unbiased MMD and a plus-one Monte Carlo permutation p-value.

    The first group occupies the first n0 entries of the observed pooled Gram
    matrix.  Balanced label assignments are batched in a quadratic-form
    evaluation; bandwidth and Gram matrix are frozen before shuffling labels.
    """
    gram = np.asarray(gram, float)
    n = len(gram)
    n1 = n - n0
    if gram.shape != (n, n) or min(n0, n1) < 2 or permutations < 1:
        raise ValueError("Need a square Gram matrix, class sizes >=2, and permutations >=1")
    if not np.isfinite(gram).all():
        raise ValueError("Gram matrix must be finite")
    rng = np.random.default_rng(seed)
    # Random ranks select a uniform fixed-size subset without a Python loop.
    permutations_indices = np.argsort(rng.random((permutations, n)), axis=1)[:, :n0]
    membership = np.zeros((permutations + 1, n))
    membership[0, :n0] = 1
    membership[1 + np.arange(permutations)[:, None], permutations_indices] = 1
    kernel = gram.copy()
    np.fill_diagonal(kernel, 0)
    products = membership @ kernel
    within0 = np.einsum("ij,ij->i", products, membership)
    cross = products.sum(axis=1) - within0
    within1 = kernel.sum() - within0 - 2 * cross
    statistics = within0 / (n0 * (n0 - 1)) + within1 / (n1 * (n1 - 1))
    statistics -= 2 * cross / (n0 * n1)
    pvalue = (1 + np.count_nonzero(statistics[1:] >= statistics[0] - 1e-14)) / (permutations + 1)
    return {"statistic": float(statistics[0]), "pvalue": float(pvalue),
            "reject": bool(pvalue <= 0.05), "permutations": permutations}


def support_metrics(support, truth):
    if truth is None:
        return {}
    support, truth = set(support), set(truth)
    if not truth:
        raise ValueError("Recovery metrics are undefined for empty ground-truth support")
    hits = len(support & truth)
    precision, recall = hits / max(1, len(support)), hits / len(truth)
    return {"fdp": len(support - truth) / max(1, len(support)),
            "ndp": 1 - recall, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support_jaccard": hits / len(support | truth)}


def stability_jaccard(supports):
    values = []
    for i, left in enumerate(supports):
        for right in supports[i + 1:]:
            left, right = set(left), set(right)
            values.append(len(left & right) / len(left | right) if left | right else 1.0)
    return float(np.mean(values)) if values else None


def probability_metrics(p0_on0, p0_on1):
    """Class-zero probability; ties at 0.5 deterministically select class zero."""
    p0, p1 = (np.clip(np.asarray(p, float), 0, 1) for p in (p0_on0, p0_on1))
    if not len(p0) or not len(p1) or not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("Both classes need nonempty finite probability predictions")
    false_positive, false_negative = float(np.mean(p0 < 0.5)), float(np.mean(p1 >= 0.5))
    randomized0, randomized1 = float(np.mean(1 - p0)), float(np.mean(p1))
    y = np.r_[np.zeros(len(p0)), np.ones(len(p1))]
    scores = 1 - np.r_[p0, p1]
    log_loss_sum = float(-np.log(np.clip(p0, 1e-12, 1)).mean()
                         - np.log(np.clip(1 - p1, 1e-12, 1)).mean())
    return {"false_positive_rate": false_positive, "false_negative_rate": false_negative,
            "balanced_error": (false_positive + false_negative) / 2,
            "randomized_type_i": randomized0, "randomized_type_ii": randomized1,
            "randomized_balanced_error": (randomized0 + randomized1) / 2,
            "auc": float(roc_auc_score(y, scores)),
            "predictive_log_loss_sum": log_loss_sum,
            "predictive_log_loss_average": log_loss_sum / 2,
            "half_hinge_surrogate_sum": randomized0 + randomized1,
            "half_hinge_surrogate_average": (randomized0 + randomized1) / 2}


def score_metrics(score0, score1):
    """Class-one scores with decision threshold zero (e.g. an RBF SVM)."""
    score0, score1 = np.asarray(score0), np.asarray(score1)
    fp, fn = float(np.mean(score0 > 0)), float(np.mean(score1 <= 0))
    y = np.r_[np.zeros(len(score0)), np.ones(len(score1))]
    return {"false_positive_rate": fp, "false_negative_rate": fn,
            "balanced_error": (fp + fn) / 2,
            "auc": float(roc_auc_score(y, np.r_[score0, score1]))}


def translation_direction(head0, head1):
    direction = head1.mean(axis=0) - head0.mean(axis=0)
    length = np.abs(direction).sum()
    return direction / length if length > 0 else np.zeros_like(direction)


def translated_pair(pair, direction, rho):
    return pair[0] + rho * direction, pair[1] - rho * direction
