import io
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import pytest

from robusthw.experiments import datasets
from robusthw.experiments.datasets import alternating_run_indices, har_split, tabular_split


def test_tabular_nested_training_and_disjoint_class_stratified_splits(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(160, 5))
    x[:, 2] = 1
    np.savez(tmp_path / "sonar.npz", x=x, y=np.repeat([0, 1], 80))
    small, large = [tabular_split("sonar", tmp_path, n, 2) for n in (10, 40)]
    for label in (0, 1):
        indices = large.metadata["split_indices"]
        assert set(small.metadata["split_indices"]["train"][label]) <= set(indices["train"][label])
        assert not set(indices["train"][label]) & set(indices["validation"][label])
        assert not set(indices["validation"][label]) & set(indices["test"][label])
        assert len(indices["validation"][label]) == 16 and len(indices["test"][label]) == 16
    assert small.feature_ids == (0, 1, 3, 4)


def test_run_thinning_resets_before_activity_filtering():
    labels = np.array([4, 4, 4, 1, 4, 4, 5, 5, 5, 4, 4])
    subjects = np.array([1] * 9 + [2] * 2)
    assert alternating_run_indices(labels, subjects).tolist() == [0, 2, 4, 6, 8, 9]


def test_har_preserves_official_people_and_balances_training_windows(tmp_path):
    rng = np.random.default_rng(7)
    arrays = {}
    for part, people in (("train", range(1, 22)), ("test", range(22, 31))):
        subjects, labels = [], []
        for person in people:
            subjects.extend([person] * 40)
            labels.extend([4] * 20 + [5] * 20)
        arrays[part + "x"] = rng.normal(size=(len(subjects), 4))
        arrays[part + "y"] = np.array(labels)
        arrays[part + "subjects"] = np.array(subjects)
    np.savez(
        tmp_path / "har.npz",
        **{
            key.replace("train", "train_").replace("test", "test_"): value
            for key, value in arrays.items()
        },
    )
    split = har_split(tmp_path, 12)
    metadata = split.metadata
    assert len(metadata["training_subjects"]) == 15 and len(metadata["validation_subjects"]) == 6
    assert not set(metadata["training_subjects"]) & set(metadata["validation_subjects"])
    assert not set(metadata["test_subjects"]) & set(metadata["training_subjects"])
    assert split.train0.shape == (40, 4) and len(np.unique(split.subjects0)) == 9
    people = arrays["trainsubjects"]
    for rows in metadata["split_indices"]["train"]:
        counts = np.unique(people[rows], return_counts=True)[1]
        assert counts.max() - counts.min() == 1
    assert all(index % 2 == 0 for index in metadata["split_indices"]["test"][0])


def test_download_checks_source_bytes_before_parsing(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("wdbc.data", "1,B,1,2\n2,M,3,4\n")
    blob = stream.getvalue()
    source = {
        "name": "wdbc",
        "url": "https://example.invalid/data",
        "rows": 2,
        "features": 2,
        "sha256": sha256(blob).hexdigest(),
    }
    monkeypatch.setattr(datasets, "SOURCES", (source,))
    monkeypatch.setattr(datasets, "urlopen", lambda *args, **kwargs: io.BytesIO(blob))
    manifest = datasets.fetch_datasets(tmp_path)
    assert manifest[0]["parsed_sha256"] == sha256((tmp_path / "wdbc.npz").read_bytes()).hexdigest()
    (tmp_path / "wdbc.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        datasets.fetch_datasets(tmp_path)
