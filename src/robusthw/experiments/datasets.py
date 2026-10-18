"""Attributed UCI downloads, immutable source hashes, and leakage-free splits."""

import io
import json
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from .data import random_stream, standardize_split

SOURCES = (
    {
        "name": "wdbc",
        "uci_id": 17,
        "rows": 569,
        "features": 30,
        "url": "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "sha256": "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        "doi": "https://doi.org/10.24432/C5DW2B",
        "license": "CC BY 4.0",
        "labels": {"0": "benign", "1": "malignant"},
        "excluded": ["ID", "diagnosis"],
    },
    {
        "name": "ionosphere",
        "uci_id": 52,
        "rows": 351,
        "features": 34,
        "url": "https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        "sha256": "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860",
        "doi": "https://doi.org/10.24432/C5W01B",
        "license": "CC BY 4.0",
        "labels": {"0": "good", "1": "bad"},
        "excluded": ["target"],
    },
    {
        "name": "sonar",
        "uci_id": 151,
        "rows": 208,
        "features": 60,
        "url": "https://archive.ics.uci.edu/static/public/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks.zip",
        "sha256": "088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d",
        "doi": "https://doi.org/10.24432/C5T01Q",
        "license": "CC BY 4.0",
        "labels": {"0": "rock", "1": "metal cylinder"},
        "excluded": ["target"],
    },
    {
        "name": "har",
        "uci_id": 240,
        "rows": 10299,
        "features": 561,
        "url": "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip",
        "sha256": "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
        "doi": "https://doi.org/10.24432/C54S4K",
        "license": "CC BY 4.0",
        "labels": {"0": "sitting (official label 4)", "1": "standing (official label 5)"},
        "excluded": ["activity label", "subject ID"],
        "official_subject_split": [21, 9],
    },
)


def _parse_archive(name, blob):
    """Read only known members; never extract archive-supplied filesystem paths."""
    with ZipFile(io.BytesIO(blob)) as archive:
        if name == "har":
            with ZipFile(io.BytesIO(archive.read("UCI HAR Dataset.zip"))) as har:
                arrays = {}
                for part in ("train", "test"):
                    for field, filename in (
                        ("x", f"X_{part}.txt"),
                        ("y", f"y_{part}.txt"),
                        ("subjects", f"subject_{part}.txt"),
                    ):
                        path = f"UCI HAR Dataset/{part}/{filename}"
                        arrays[f"{part}_{field}"] = np.loadtxt(
                            io.BytesIO(har.read(path)), dtype=float if field == "x" else int
                        )
                return arrays
        filename = {
            "wdbc": "wdbc.data",
            "ionosphere": "ionosphere.data",
            "sonar": "sonar.all-data",
        }[name]
        path = next(member for member in archive.namelist() if member.endswith(filename))
        rows = [line.split(",") for line in archive.read(path).decode().strip().splitlines()]
    if name == "wdbc":
        return {
            "x": np.array([row[2:] for row in rows], float),
            "y": np.array([int(row[1] == "M") for row in rows]),
        }
    target = "b" if name == "ionosphere" else "M"
    return {
        "x": np.array([row[:-1] for row in rows], float),
        "y": np.array([int(row[-1] == target) for row in rows]),
    }


def fetch_datasets(data_root, cache_roots=()):
    """Download four pinned archives, verify before parsing, and save NPZ caches.

    cache_roots optionally supplies existing archives, avoiding redundant network
    transfer without accepting an unverified parsed array.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in SOURCES:
        archive = root / (source["name"] + ".zip")
        candidates = (archive, *(Path(cache) / archive.name for cache in cache_roots))
        existing = next((path for path in candidates if path.is_file()), None)
        blob = existing.read_bytes() if existing else urlopen(source["url"], timeout=180).read()
        if sha256(blob).hexdigest() != source["sha256"]:
            raise ValueError(
                f"Source checksum changed for {source['name']}; inspect upstream before updating"
            )
        arrays = _parse_archive(source["name"], blob)
        shape = (
            (len(arrays["train_x"]) + len(arrays["test_x"]), arrays["train_x"].shape[1])
            if source["name"] == "har"
            else arrays["x"].shape
        )
        if shape != (source["rows"], source["features"]):
            raise ValueError(f"Unexpected source shape for {source['name']}: {shape}")
        for field, values in arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite source values: {source['name']}/{field}")
        if not archive.exists():
            archive.write_bytes(blob)
        destination = root / (source["name"] + ".npz")
        np.savez_compressed(destination, **arrays)
        manifest.append({**source, "parsed_sha256": sha256(destination.read_bytes()).hexdigest()})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load(name, data_root):
    root = Path(data_root)
    path = root / (name + ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run robusthw fetch-data first")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next((row for row in entries if row["name"] == name), None)
        if (
            record
            and "parsed_sha256" in record
            and sha256(path.read_bytes()).hexdigest() != record["parsed_sha256"]
        ):
            raise ValueError(f"Parsed dataset checksum changed: {name}")
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def tabular_split(name, data_root, n, split_seed):
    if name not in ("wdbc", "ionosphere", "sonar"):
        raise ValueError("Unknown tabular benchmark")
    arrays = _load(name, data_root)
    x, y = arrays["x"], arrays["y"]
    pairs = [[], [], []]
    indices = {"train": [], "validation": [], "test": [], "train_pool": []}
    for label in (0, 1):
        order = random_stream(split_seed, f"{name}_class_{label}").permutation(
            np.flatnonzero(y == label)
        )
        train_end, val_end = int(np.floor(0.6 * len(order))), int(np.floor(0.2 * len(order)))
        if n < 2 or n > train_end:
            raise ValueError("Training request exceeds the 60 percent class pool")
        groups = (order[:n], order[train_end : train_end + val_end], order[train_end + val_end :])
        indices["train_pool"].append(order[:train_end].astype(int).tolist())
        for pair, key, group in zip(pairs, ("train", "validation", "test"), groups):
            pair.append(x[group])
            indices[key].append(group.astype(int).tolist())
    return standardize_split(
        pairs,
        np.arange(x.shape[1]),
        metadata={
            "kind": "tabular",
            "dataset": name,
            "n": n,
            "seed": int(split_seed),
            "split_indices": indices,
            "source_sha256": next(s["sha256"] for s in SOURCES if s["name"] == name),
        },
    )


def alternating_run_indices(labels, subjects):
    """Retain window 0,2,4,... in each original contiguous subject/activity run."""
    labels, subjects = np.asarray(labels), np.asarray(subjects)
    if labels.shape != subjects.shape or labels.ndim != 1:
        raise ValueError("Activity labels and subjects must align")
    if not len(labels):
        return np.array([], dtype=int)
    starts = np.r_[True, (labels[1:] != labels[:-1]) | (subjects[1:] != subjects[:-1])]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(len(labels)), 0))
    return np.flatnonzero(((np.arange(len(labels)) - run_start) % 2 == 0) & np.isin(labels, (4, 5)))


def _balanced_subject_sample(candidates, subjects, people, n, rng):
    """Round-robin random subject order: per-person counts differ by at most one."""
    people = rng.permutation(people)
    queues = {
        int(person): list(rng.permutation(candidates[subjects[candidates] == person]))
        for person in people
    }
    selected = []
    while len(selected) < n:
        progressed = False
        for person in people:
            queue = queues[int(person)]
            if queue and len(selected) < n:
                selected.append(int(queue.pop()))
                progressed = True
        if not progressed:
            raise ValueError("Insufficient retained HAR windows")
    return np.array(selected, int)


def har_split(data_root, seed):
    arrays = _load("har", data_root)
    tx, ty, ts = (arrays["train_" + field] for field in ("x", "y", "subjects"))
    ex, ey, es = (arrays["test_" + field] for field in ("x", "y", "subjects"))
    train_people, test_people = np.unique(ts), np.unique(es)
    if len(train_people) != 21 or len(test_people) != 9 or set(train_people) & set(test_people):
        raise ValueError("HAR must preserve the official 21/9 subject split")
    rng = random_stream(seed, "har_subject_partition")
    val_people = np.sort(rng.choice(train_people, 6, replace=False))
    fit_people = np.setdiff1d(train_people, val_people)
    retained_train, retained_test = alternating_run_indices(ty, ts), alternating_run_indices(ey, es)
    pairs = [[], [], []]
    indices = {"train": [], "validation": [], "test": []}
    test_subjects = []
    for label in (4, 5):
        fitting = retained_train[
            (ty[retained_train] == label) & np.isin(ts[retained_train], fit_people)
        ]
        fit = _balanced_subject_sample(
            fitting, ts, fit_people, 40, random_stream(seed, f"har_class_{label}")
        )
        val = retained_train[
            (ty[retained_train] == label) & np.isin(ts[retained_train], val_people)
        ]
        test = retained_test[ey[retained_test] == label]
        for pair, key, values, matrix in zip(
            pairs, ("train", "validation", "test"), (fit, val, test), (tx, tx, ex)
        ):
            pair.append(matrix[values])
            indices[key].append(values.tolist())
        test_subjects.append(es[test])
    metadata = {
        "kind": "har",
        "dataset": "har",
        "n": 40,
        "seed": int(seed),
        "training_subjects": fit_people.tolist(),
        "validation_subjects": val_people.tolist(),
        "test_subjects": test_people.tolist(),
        "split_indices": indices,
        "window_policy": "original contiguous subject/activity run, retain offsets 0,2,4,...",
        "source_sha256": next(s["sha256"] for s in SOURCES if s["name"] == "har"),
    }
    return standardize_split(
        pairs,
        np.arange(tx.shape[1]),
        metadata=metadata,
        subjects0=test_subjects[0],
        subjects1=test_subjects[1],
    )
