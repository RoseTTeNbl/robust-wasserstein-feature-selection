"""Explain the saved anchor prediction rule without fitting or tuning any model.

Run only after all 500 frozen anchor repetitions are complete. The script imports
exactly the source snapshot used by the run, verifies stored records and samples,
and checks that diagnostic scores equal FitResult.predict_score.
"""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def diagnostic_values(fitted, queries):
    source = fitted.problem.x[:, list(fitted.support)]
    targets = queries[:, list(fitted.support)]
    labels = fitted.problem.y
    distances = np.abs(source[:, None, :] - targets[None, :, :]).sum(axis=2)
    anchors = []
    for label in (0, 1):
        rows = np.flatnonzero(labels == label)
        envelopes = fitted.source_bounds[rows, None] + fitted.prices[label] * distances[rows]
        anchors.append(rows[np.argmin(envelopes, axis=0)])
    i0, i1 = anchors
    r0 = distances[i0, np.arange(len(targets))]
    rho = np.abs(source[i0] - source[i1]).sum(axis=1)
    ratio = np.divide(r0, rho, out=np.zeros_like(r0), where=rho > 0)
    gamma = np.minimum(1.0, ratio)
    score = np.clip((1 - gamma) * fitted.scores[i0] + gamma * fitted.scores[i1], -1, 1)
    np.testing.assert_allclose(score, fitted.predict_score(queries), rtol=0, atol=1e-12)
    return dict(r0=r0, rho=rho, ratio=ratio, gamma=gamma, score=score)


def summarize(values):
    return {
        "samples": len(values["score"]),
        "gamma_equal_one_rate": float(np.mean(values["gamma"] == 1.0)),
        "hard_prediction_class1_rate": float(np.mean(values["score"] < 0)),
        "randomized_prediction_class1_probability": float(np.mean((1 - values["score"]) / 2)),
        "quantiles": {
            key: np.quantile(values[key], QUANTILES).tolist()
            for key in ("r0", "rho", "ratio", "gamma", "score")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study_root.resolve()
    root = study / "anchor"
    snapshot = study / "frozen-source"
    manifest_path, execution_path = root / "protocol.json", root / "execution.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if (
        len(manifest["specs"]) != 500
        or execution["planned"] != 500
        or execution["completed"] != 500
        or execution["failed"] != 0
        or execution.get("partial_execution", True)
    ):
        raise RuntimeError("Final diagnostic requires all 500 completed frozen anchor records")
    sys.path.insert(0, str(snapshot / "src"))
    from robusthw.core import FitResult, Problem
    from robusthw.experiments.metrics import classification_metrics
    from robusthw.experiments.storage import array_hash, fingerprint, write_json

    for relative, expected in manifest["source_hashes"].items():
        if file_hash(snapshot / relative) != expected:
            raise ValueError(f"Frozen source differs from protocol: {relative}")
    planned = {item["independent_data_id"] for item in manifest["specs"]}
    paths = sorted((root / "records").rglob("*.json"))
    if len(paths) != 500:
        raise ValueError("Anchor record count differs from frozen plan")
    records, inputs, observed = [], [], set()
    pooled = [{key: [] for key in ("r0", "rho", "ratio", "gamma", "score")} for _ in range(2)]
    maximum_metric_difference = 0.0
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        data_id = record["spec"]["independent_data_id"]
        observed.add(data_id)
        if (
            record["manifest_hash"] != manifest["manifest_hash"]
            or fingerprint({k: v for k, v in record.items() if k != "record_hash"})
            != record["record_hash"]
        ):
            raise ValueError(f"Anchor record failed integrity checks: {data_id}")
        npz_path = root / "data" / (data_id + ".npz")
        if file_hash(npz_path) != record["data"]["npz_sha256"]:
            raise ValueError(f"Anchor sample archive changed: {data_id}")
        with np.load(npz_path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        for key, expected in record["data"]["array_hashes"].items():
            if array_hash(arrays[key]) != expected:
                raise ValueError(f"Anchor matrix changed: {data_id}/{key}")
        (result,) = record["results"]
        payload = result["fit"]
        if payload is None:
            raise ValueError(f"Saved anchor fit unavailable: {data_id}")
        fitted = FitResult(
            Problem(arrays["train0"], arrays["train1"], np.asarray(record["radii"])),
            tuple(payload["support"]),
            np.asarray(payload["scores"]),
            np.asarray(payload["source_bounds"]),
            np.asarray(payload["prices"]),
            payload["upper"],
            payload["lower"],
            payload["status"],
            payload["runtime"],
            None,
            payload["diagnostics"],
        )
        values = [diagnostic_values(fitted, arrays[f"test{label}"]) for label in (0, 1)]
        recomputed = classification_metrics(values[0]["score"], values[1]["score"])
        for key, value in recomputed.items():
            difference = abs(value - result["metrics"][key])
            maximum_metric_difference = max(maximum_metric_difference, difference)
            if difference > 1e-12:
                raise ValueError(f"Restored prediction metrics differ: {data_id}/{key}")
        for label in (0, 1):
            for key, value in values[label].items():
                pooled[label][key].append(value)
        records.append(
            {
                "data_id": data_id,
                "status": result["status"],
                "training_robust_risk": result["upper"],
                "test_metrics": result["metrics"],
                "by_true_class": {str(k): summarize(values[k]) for k in (0, 1)},
            }
        )
        inputs.append(
            {
                "data_id": data_id,
                "record": path.relative_to(study).as_posix(),
                "record_file_sha256": file_hash(path),
                "record_hash": record["record_hash"],
                "sample_archive_sha256": file_hash(npz_path),
            }
        )
    if observed != planned:
        raise ValueError("Saved anchor identities differ from the frozen plan")
    aggregate = {
        str(k): summarize({key: np.concatenate(parts) for key, parts in pooled[k].items()})
        for k in (0, 1)
    }
    output = {
        "purpose": "Explain the saved distance-interpolation prediction; no fitting or tuning",
        "complete": True,
        "independent_repetitions": len(records),
        "quantile_probabilities": QUANTILES,
        "definitions": {
            "r0": "L1 distance from test point to its selected class-0 source anchor",
            "rho": "L1 distance between the selected class-0 and class-1 source anchors",
            "ratio": "r0/rho; zero when rho=0, matching the saved prediction implementation",
            "gamma": "min(1, ratio)",
            "score": "(1-gamma)*u_class0_anchor + gamma*u_class1_anchor, clipped to [-1,1]",
            "hard_prediction_class1": "score < 0; score=0 predicts class 0",
            "randomized_prediction_class1_probability": "(1-score)/2",
        },
        "maximum_recomputed_metric_difference": maximum_metric_difference,
        "aggregate_by_true_class": aggregate,
        "records": records,
        "provenance": {
            "manifest_hash": manifest["manifest_hash"],
            "protocol_sha256": file_hash(manifest_path),
            "execution_sha256": file_hash(execution_path),
            "prediction_source_sha256": file_hash(snapshot / "src/robusthw/core/model.py"),
            "diagnostic_script_sha256": file_hash(__file__),
            "inputs": inputs,
        },
    }
    output["diagnostic_hash"] = fingerprint(output)
    destination = args.output or study / "diagnostics" / "anchor_prediction.json"
    write_json(destination, output)
    print(
        json.dumps(
            {
                "output": str(destination),
                "repetitions": len(records),
                "maximum_metric_difference": maximum_metric_difference,
                "aggregate_by_true_class": aggregate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
