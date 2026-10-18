"""Independent labeled samples and training-only feature transformations.

The four signal mechanisms and their Bayes-error calibration follow the current
experimental design. Budgets do not enter sampling: one data set is reused for
every budget and selector.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, f, norm


def random_stream(seed, purpose):
    """Stable named streams, independent of Python's randomized hash()."""
    tag = int.from_bytes(sha256(str(purpose).encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), tag]))


@dataclass(frozen=True)
class DataSplit:
    train0: np.ndarray
    train1: np.ndarray
    val0: np.ndarray
    val1: np.ndarray
    test0: np.ndarray
    test1: np.ndarray
    signal_indices: tuple[int, ...]
    feature_ids: tuple[int, ...]
    metadata: dict
    subjects0: np.ndarray | None = None
    subjects1: np.ndarray | None = None

    def __post_init__(self):
        dimension = len(self.feature_ids)
        for name in ("train0", "train1", "val0", "val1", "test0", "test1"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 2 or value.shape[1] != dimension or not len(value):
                raise ValueError(f"{name} must be nonempty with {dimension} columns")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains nonfinite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if len(set(self.feature_ids)) != dimension:
            raise ValueError("Feature identities must be unique")
        if any(j < 0 or j >= dimension for j in self.signal_indices):
            raise ValueError("Signal indices must refer to retained columns")
        for label in (0, 1):
            subjects = getattr(self, f"subjects{label}")
            if subjects is not None:
                subjects = np.asarray(subjects, dtype=int)
                if subjects.shape != (len(getattr(self, f"test{label}")),):
                    raise ValueError("Test subjects must align with test rows")
                subjects.setflags(write=False)
                object.__setattr__(self, f"subjects{label}", subjects)

    @property
    def dimension(self):
        return len(self.feature_ids)

    @property
    def train(self):
        return self.train0, self.train1

    @property
    def validation(self):
        return self.val0, self.val1

    @property
    def test(self):
        return self.test0, self.test1

    def with_test(self, x0, x1):
        """Reuse fitted data without refitting preprocessing for a test shift."""
        return replace(self, test0=x0, test1=x1)

    def as_dict(self):
        return {
            "signal_indices": list(self.signal_indices),
            "feature_ids": list(self.feature_ids),
            "split_sizes": {
                name: len(getattr(self, name))
                for name in ("train0", "train1", "val0", "val1", "test0", "test1")
            },
            **self.metadata,
        }


def standardize_split(
    pairs, feature_ids, *, signal_ids=(), metadata=None, subjects0=None, subjects1=None
):
    """Fit pooled sample mean/SD once; remove only exactly constant train columns."""
    train = np.vstack(pairs[0])
    if len(train) < 2 or train.ndim != 2 or not np.isfinite(train).all():
        raise ValueError("At least two finite training rows are required")
    keep = np.ptp(train, axis=0) > 0
    if not keep.any():
        raise ValueError("All training features are constant")
    center, scale = train.mean(axis=0)[keep], train.std(axis=0, ddof=1)[keep]
    if not np.all(scale > 0):
        raise ValueError("Nonpositive scale after constant-column removal")
    arrays = [(np.asarray(x, float)[:, keep] - center) / scale for pair in pairs for x in pair]
    retained = tuple(int(j) for j in np.asarray(feature_ids)[keep])
    signal_ids = set(signal_ids)
    signals = tuple(j for j, original in enumerate(retained) if original in signal_ids)
    transform = {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "ddof": 1,
        "original_dimension": len(feature_ids),
        "dropped_feature_ids": np.asarray(feature_ids)[~keep].astype(int).tolist(),
    }
    return DataSplit(
        *arrays,
        signals,
        retained,
        {**(metadata or {}), "preprocessing": transform},
        subjects0,
        subjects1,
    )


def bayes_error(model, signal_size, parameter):
    """Exact equal-prior Bayes error, before training standardization."""
    s, value = int(signal_size), float(parameter)
    if model == "mean":
        return float(norm.cdf(-value * np.sqrt(s)))
    if model == "variance":
        if value <= 1:
            return 0.5
        threshold = 2 * s * np.log(value) / (1 - value**-2)
        return float((chi2.sf(threshold, s) + chi2.cdf(threshold / value**2, s)) / 2)
    if model == "correlation":
        return float(f.cdf((1 - value) / (1 + value), s / 2, s / 2))
    if model == "mixture":
        p = norm.cdf(-value * np.sqrt(s / 2))
        return float(2 * p * (1 - p))
    raise ValueError(f"Unknown signal model: {model}")


@lru_cache(maxsize=None)
def calibrated_parameter(model, signal_size, target_error=0.2):
    s = int(signal_size)
    if s < 1 or not 0 < target_error < 0.5:
        raise ValueError("Need positive signal size and target error in (0, .5)")
    if model in ("correlation", "mixture") and s % 2:
        raise ValueError("Correlation and mixture signals require an even signal size")
    if model == "mean":
        return float(norm.isf(target_error) / np.sqrt(s))
    if model == "mixture":
        p = (1 - np.sqrt(1 - 2 * target_error)) / 2
        return float(np.sqrt(2 / s) * norm.isf(p))
    if model == "variance":
        return float(brentq(lambda r: bayes_error(model, s, r) - target_error, 1 + 1e-10, 100))
    if model == "correlation":
        return float(brentq(lambda rho: bayes_error(model, s, rho) - target_error, 0, 1 - 1e-12))
    raise ValueError(f"Unknown signal model: {model}")


def draw_population(model, D, s, parameter, rng, rows, label):
    """Generate signals first and independent standard-normal noise afterward."""
    x = rng.standard_normal((rows, D))
    if model == "mean":
        x[:, :s] += (2 * label - 1) * parameter
    elif model == "variance":
        if label:
            x[:, :s] *= parameter
    elif model == "correlation":
        rho = parameter
        x[:, 1:s:2] = (1 - 2 * label) * rho * x[:, :s:2] + np.sqrt(1 - rho**2) * x[:, 1:s:2]
    elif model == "mixture":
        direction = np.ones(s)
        if label:
            direction[s // 2 :] = -1
        signs = rng.choice((-1.0, 1.0), size=(rows, 1))
        x[:, :s] += parameter * signs * direction
    else:
        raise ValueError(f"Unknown signal model: {model}")
    return x


def _generated_split(
    model, D, n, s, seed, validation_per_class, test_per_class, parameter, target_error
):
    if not 1 <= s <= D or n < 2 or min(validation_per_class, test_per_class) < 1:
        raise ValueError("Invalid dimensions or split sizes")
    if model in ("correlation", "mixture") and s % 2:
        raise ValueError("Signal size must be even for this model")
    permutation = random_stream(seed, "column_order").permutation(D)
    pairs = []
    for role, rows in (
        ("train", n),
        ("validation", validation_per_class),
        ("test", test_per_class),
    ):
        pairs.append(
            tuple(
                draw_population(
                    model, D, s, parameter, random_stream(seed, f"{role}_{label}"), rows, label
                )[:, permutation]
                for label in (0, 1)
            )
        )
    metadata = {
        "kind": "synthetic",
        "model": model,
        "D": D,
        "n": n,
        "s": s,
        "seed": int(seed),
        "parameter": parameter,
        "target_bayes_error": target_error,
        "column_order": permutation.tolist(),
        "data_sha256": sha256(b"".join(x.tobytes() for pair in pairs for x in pair)).hexdigest(),
    }
    return standardize_split(pairs, permutation, signal_ids=range(s), metadata=metadata)


def synthetic_split(
    model, D, n, s, seed, validation_per_class=20, test_per_class=10000, target_error=0.2
):
    parameter = calibrated_parameter(model, s, target_error)
    return _generated_split(
        model, D, n, s, seed, validation_per_class, test_per_class, parameter, target_error
    )


def original_anchor_split(seed):
    """The reference's 100D mixture, with a separate 20/class validation sample."""
    split = _generated_split("mixture", 100, 10, 100, seed, 20, 1000, 0.4, None)
    return replace(
        split,
        signal_indices=(),
        metadata={
            **split.metadata,
            "kind": "anchor",
            "recovery_defined": False,
            "source_section": "5.1",
            "source_url": "https://arxiv.org/pdf/2105.14348#page=19",
        },
    )
