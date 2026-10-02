"""Summarize saved records; uncertainty is clustered at the training seed."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def bootstrap_interval(values, seed=20261003):
    values = np.array(values, float)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in records)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/research_20261003"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("records/*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        sel = r.get("selection", {})
        diagnostics = sel.get("diagnostics", {})
        row = dict(
            suite=path.relative_to(args.root).parts[0],
            dataset=r["dataset"],
            seed=r["seed"],
            dimension=r["dimension"],
            budget=r["budget"],
            theta=float(r.get("theta", 0.25)),
            method=r["method"],
            run_status=r["run_status"],
            selection_status=sel.get("status", "failure"),
            record_file=str(path.relative_to(args.root)),
            config_sha256=r["config_sha256"],
        )
        for field in ["lower", "upper", "gap", "seconds", "oracle_calls", "visited"]:
            if field in sel:
                row["selection_" + field] = sel[field]
        for field in ["root_bound", "oracle_seconds", "master_seconds"]:
            if field in diagnostics:
                row[field] = diagnostics[field]
        row.update(r.get("metrics", {}))
        for s in r.get("stress", []):
            row[f"stress_ber_{s['rho']:.2f}"] = s["balanced_error"]
            row[f"stress_randomized_ber_{s['rho']:.2f}"] = s[
                "randomized_balanced_error"
            ]
        row["support"] = json.dumps(sel.get("support", []))
        row["semantic_support"] = json.dumps(r.get("semantic_support", []))
        rows.append(row)
    if not rows:
        raise RuntimeError("No saved run records")
    write_csv(args.root / "per_trial.csv", rows)
    groups = defaultdict(list)
    fields = ["suite", "dataset", "dimension", "budget", "theta", "method"]
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    summary = []
    excluded = {"dimension", "budget", "theta", "seed"}
    for key, group in sorted(groups.items()):
        out = dict(zip(fields, key))
        out.update(
            n_total=len(group),
            n_complete=sum(r["run_status"] == "complete" for r in group),
            status_counts=json.dumps(
                dict(Counter(r["selection_status"] for r in group))
            ),
        )
        if key[-1] in (
            "risk",
            "risk_ratio",
            "risk_no_root",
            "box",
            "triangle",
            "enumeration",
        ):
            out["solved_fraction"] = sum(
                r["selection_status"] == "optimal" for r in group
            ) / len(group)
        numeric = sorted(set().union(*(r.keys() for r in group)) - excluded)
        for metric in numeric:
            vals = [
                r[metric]
                for r in group
                if isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])
            ]
            if not vals:
                continue
            out[metric + "_mean"] = float(np.mean(vals))
            out[metric + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[metric + "_n"] = len(vals)
            if key[0] != "real":
                lo, hi = bootstrap_interval(vals)
                out[metric + "_lo"], out[metric + "_hi"] = lo, hi
        supports = [
            set(json.loads(r["semantic_support"]))
            for r in group
            if r["run_status"] == "complete"
        ]
        pairs = [
            len(a & b) / len(a | b) if a | b else 1.0
            for a, b in combinations(supports, 2)
        ]
        out["stability_jaccard"] = float(np.mean(pairs)) if pairs else None
        summary.append(out)
    write_csv(args.root / "summary.csv", summary)
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    count = dict(
        total_records=len(rows),
        failed_records=sum(r["run_status"] != "complete" for r in rows),
        failed_records_definition="runner/evaluation failures, not optimizer stopping statuses",
        selection_failure_records=sum(
            r["selection_status"] in {
                "master_failure", "oracle_failure", "numerical_failure",
                "numerical_inconsistency", "failure",
            }
            for r in rows
        ),
        selection_time_limit_records=sum("time_limit" in r["selection_status"] for r in rows),
        suites=dict(Counter(r["suite"] for r in rows)),
        statuses=dict(Counter(r["selection_status"] for r in rows)),
    )
    (args.root / "result_inventory.json").write_text(
        json.dumps(count, indent=2), encoding="utf-8"
    )
    print(json.dumps(count))


if __name__ == "__main__":
    main()
