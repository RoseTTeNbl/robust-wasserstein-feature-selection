"""Prediction metrics use class-zero scores; independent repeats are explicit."""

import numpy as np


def classification_metrics(scores0, scores1):
    scores = [np.asarray(value, float) for value in (scores0, scores1)]
    if any(value.ndim != 1 or not len(value) or not np.isfinite(value).all() for value in scores):
        raise ValueError("Scores must be nonempty finite vectors")
    if any(np.max(np.abs(value)) > 1 + 1e-7 for value in scores):
        raise ValueError("Scores must lie in [-1, 1]")
    p0, p1 = [(1 + np.clip(value, -1, 1)) / 2 for value in scores]
    error0_prob, error1_prob = float(np.mean(1 - p0)), float(np.mean(p1))
    error0_hard, error1_hard = float(np.mean(scores[0] < 0)), float(np.mean(scores[1] >= 0))
    return {
        "ber_prob": (error0_prob + error1_prob) / 2,
        "ber_hard": (error0_hard + error1_hard) / 2,
        "error0_prob": error0_prob,
        "error1_prob": error1_prob,
        "error0_hard": error0_hard,
        "error1_hard": error1_hard,
        "half_hinge_risk": error0_prob + error1_prob,
    }


def recovery_metrics(support, signal_indices, budget):
    support, truth = set(support), set(signal_indices)
    if budget < 1 or len(support) != budget:
        raise ValueError("Recovery requires the declared completed positive budget")
    if not truth:
        return {}
    hits = len(support & truth)
    return {"signal_hits": hits, "precision": hits / budget, "recall": hits / len(truth)}


def paired_bootstrap_ci(differences, seed=20261018, repeats=2000):
    values = np.asarray(differences, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or repeats < 1:
        raise ValueError("Need finite paired differences and a positive resample count")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def subject_metrics(scores0, scores1, subjects0, subjects1):
    """Each held-out person has equal weight, independent of window counts."""
    subjects0, subjects1 = np.asarray(subjects0), np.asarray(subjects1)
    ids0, ids1 = set(subjects0), set(subjects1)
    if ids0 != ids1:
        raise ValueError("Both test classes must occur for every held-out person")
    return {
        int(person): classification_metrics(
            np.asarray(scores0)[subjects0 == person], np.asarray(scores1)[subjects1 == person]
        )
        for person in sorted(ids0)
    }
