"""Frozen-protocol data mechanisms and development-only preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x0, x1):
        pooled = np.vstack((x0, x1))
        scale = pooled.std(axis=0, ddof=1)
        return cls(pooled.mean(axis=0), np.maximum(scale, 1e-8))

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.center) / self.scale

    def as_dict(self):
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class Population:
    """Column order maps observed positions to stable semantic coordinates."""

    mechanism: str
    dimension: int
    signal_size: int
    column_order: np.ndarray

    def draw(self, rng, n, label):
        x = rng.normal(size=(n, self.dimension))
        s = self.signal_size
        if label == 1:
            if self.mechanism == "mean":
                x[:, :s] += 0.9
            elif self.mechanism == "variance":
                x[:, :s] *= 1.8
            elif self.mechanism == "dependence":
                covariance = np.full((s, s), 0.65)
                np.fill_diagonal(covariance, 1.0)
                x[:, :s] = x[:, :s] @ np.linalg.cholesky(covariance).T
            elif self.mechanism == "mixture":
                x[:, :s] += rng.choice([-1.5, 1.5], size=(n, 1))
            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")
        return x[:, self.column_order]

    @property
    def true_support(self):
        return tuple(int(j) for j in np.flatnonzero(self.column_order < self.signal_size))


@dataclass
class Trial:
    name: str
    seed: int
    selection: tuple[np.ndarray, np.ndarray]
    head: tuple[np.ndarray, np.ndarray]
    testing: tuple[np.ndarray, np.ndarray]
    standardizer: Standardizer
    column_order: np.ndarray
    true_support: tuple[int, ...] | None
    population: Population | None = None
    split_indices: dict | None = None

    @property
    def dimension(self):
        return self.selection[0].shape[1]

    def semantic_support(self, support):
        return sorted(int(self.column_order[j]) for j in support)

    def sample_testing_pair(self, replicate, n, *, null=False):
        if self.population is None:
            raise ValueError("Fresh population draws are available only for synthetic trials")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 10001, replicate, int(null)]))
        raw0 = self.population.draw(rng, n, 0)
        raw1 = self.population.draw(rng, n, 0 if null else 1)
        return self.standardizer.transform(raw0), self.standardizer.transform(raw1)

    def as_dict(self):
        return {
            "dataset": self.name,
            "seed": self.seed,
            "dimension": self.dimension,
            "column_order": self.column_order.tolist(),
            "true_support": self.true_support,
            "preprocessing": self.standardizer.as_dict(),
            "split_sizes": {"selection": [len(x) for x in self.selection],
                            "head": [len(x) for x in self.head],
                            "testing": [len(x) for x in self.testing]},
            "split_indices": self.split_indices,
        }


def synthetic_trial(mechanism, seed, *, dimension=12, signal_size=3,
                    n_select=20, n_head=20, n_classification=1000):
    if mechanism not in ("mean", "variance", "dependence", "mixture"):
        raise ValueError(f"Unknown mechanism: {mechanism}")
    if not 1 <= signal_size <= dimension:
        raise ValueError("signal_size must lie between one and dimension")
    if min(n_select, n_head, n_classification) < 2:
        raise ValueError("Each split needs at least two observations per class")
    streams = np.random.SeedSequence([seed, 31003]).spawn(4)
    order = np.random.default_rng(streams[0]).permutation(dimension)
    population = Population(mechanism, dimension, signal_size, order)
    splits = []
    for stream, n in zip(streams[1:], (n_select, n_head, n_classification)):
        rng = np.random.default_rng(stream)
        splits.append(tuple(population.draw(rng, n, k) for k in (0, 1)))
    standardizer = Standardizer.fit(*splits[0])
    transformed = [tuple(standardizer.transform(x) for x in pair) for pair in splits]
    return Trial(mechanism, seed, *transformed, standardizer, order,
                 population.true_support, population)


def real_trial(path, seed, *, n_select=20, n_head=20):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        x, y = np.asarray(source["x"], float), np.asarray(source["y"])
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Real dataset must contain finite 2D x and aligned y")
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError("Real dataset must have exactly two classes")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 41003]))
    splits = [[], [], []]
    indices = {"selection": [], "head": [], "testing": [],
               "label_mapping": [str(v) for v in labels]}
    for label in labels:
        order = rng.permutation(np.flatnonzero(y == label))
        if len(order) < n_select + n_head + 2:
            raise ValueError("Insufficient records for disjoint development and final test splits")
        groups = np.split(order, [n_select, n_select + n_head])
        for pair, key, group in zip(splits, ("selection", "head", "testing"), groups):
            pair.append(x[group])
            indices[key].append(group.tolist())
    standardizer = Standardizer.fit(*splits[0])
    transformed = [tuple(standardizer.transform(x) for x in pair) for pair in splits]
    return Trial(path.stem, seed, *transformed, standardizer,
                 np.arange(x.shape[1]), None, split_indices=indices)
