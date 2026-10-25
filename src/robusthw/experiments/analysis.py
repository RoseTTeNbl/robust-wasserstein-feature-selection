"""Aggregate immutable experiment records without changing their sampling unit.

Every condition retains its data family, dimensions, sample size, signal count,
budget, radius, initialization and manifest. Timing repetitions are collapsed
within an instance. HAR prediction statistics first average training repetitions
within each held-out person; nine people never become ninety independent tests.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap_ci
from .storage import fingerprint, read_json, write_json

GROUP_FIELDS = (
    "manifest_hash",
    "suite",
    "kind",
    "model",
    "D",
    "n",
    "s",
    "budget",
    "method",
    "role",
    "radius_factor",
    "initialization",
)
PREDICTION_METRICS = {
    "ber_prob",
    "ber_hard",
    "error0_prob",
    "error1_prob",
    "error0_hard",
    "error1_hard",
    "half_hinge_risk",
    "hinge_risk",
}
INVALID_CERTIFICATE_STATUSES = {
    "verification_failure",
    "numerical_failure",
    "instance_failure",
    "solver_error",
    "failed",
    "invalid_certificate",
}


def _finite(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _key(group, excluded=()):
    return tuple((name, group[name]) for name in GROUP_FIELDS if name not in excluded)


def _stats(values):
    values = np.asarray(values, float)
    return {
        "n": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def _ci(values, identity):
    if len(values) < 2:
        return None
    seed = int(fingerprint(identity)[:8], 16)
    return list(paired_bootstrap_ci(values, seed=seed, repeats=2000))


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group(record, method):
    spec = record["spec"]
    return {
        "manifest_hash": record["manifest_hash"],
        "suite": spec.get("metadata", {}).get("suite", spec["kind"]),
        **{name: spec[name] for name in ("kind", "model", "D", "n", "s")},
        "budget": method["budget"],
        "method": method["method"],
        "role": method.get("role", "comparison"),
        "radius_factor": method.get("radius_factor", 1),
        "initialization": method.get("initialization", "marginal_mmd"),
    }


def _optimum(record, method):
    for container in (
        method.get("validation", {}),
        record.get("verification", {}),
        record.get("validation", {}),
    ):
        if isinstance(container, dict) and _finite(container.get("independent_optimum")):
            return float(container["independent_optimum"])
    return None


def _protocols(root):
    base = root.parent if root.name == "records" else root
    paths = (
        [base / "protocol.json"]
        if (base / "protocol.json").exists()
        else sorted(base.rglob("protocol.json"))
    )
    manifests, errors = [], []
    for path in paths:
        try:
            manifest = read_json(path)
            payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
            if fingerprint(payload) != manifest.get("manifest_hash"):
                raise ValueError("Frozen protocol contents changed")
            manifests.append({"path": str(path), **manifest})
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return manifests, errors


def _load(root, manifests):
    directories = (
        [root]
        if root.name == "records"
        else (
            [root / "records"]
            if (root / "records").exists()
            else [path for path in root.rglob("records") if path.is_dir()]
        )
    )
    paths = sorted({path for directory in directories for path in directory.rglob("*.json")})
    frozen = {
        manifest["manifest_hash"]: {spec["independent_data_id"]: spec for spec in manifest["specs"]}
        for manifest in manifests
    }
    attempts, failures, errors, excluded, seen = [], [], [], [], {}
    for path in paths:
        try:
            record = read_json(path)
            if record.get("schema_version") != 1:
                raise ValueError("Unsupported or missing record schema_version")
            payload = {key: value for key, value in record.items() if key != "record_hash"}
            if record.get("record_hash") != fingerprint(payload):
                raise ValueError("Saved record contents changed or record_hash is missing")
            spec = record["spec"]
            identity = spec["independent_data_id"]
            if frozen:
                expected = frozen.get(record["manifest_hash"], {}).get(identity)
                if expected is None or fingerprint(spec) != fingerprint(expected):
                    raise ValueError("Record setting does not match its frozen protocol")
            if (
                spec.get("metadata", {}).get("excluded_from_final_statistics", False)
                or spec["kind"] == "development"
            ):
                excluded.append({"path": str(path), "independent_data_id": identity})
                continue
            seen[(record["manifest_hash"], identity)] = record.get("status", "missing")
            if record.get("status") != "completed":
                failures.append(
                    {
                        "path": str(path),
                        "independent_data_id": identity,
                        "status": record.get("status", "missing"),
                        "error": str(record.get("error", "")),
                        "spec": spec,
                    }
                )
            for index, method in enumerate(record.get("results", [])):
                try:
                    group = _group(record, method)
                    attempts.append(
                        {
                            "group": group,
                            "independent_data_id": identity,
                            "record_status": record.get("status", "missing"),
                            "source_path": str(path),
                            "timing_repeat": int(method.get("timing_repeat", 0)),
                            "result": method,
                            "independent_optimum": _optimum(record, method),
                        }
                    )
                except (KeyError, TypeError, ValueError) as error:
                    errors.append({"path": str(path), "result_index": index, "error": str(error)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return paths, attempts, failures, errors, excluded, seen


def _planned(manifests, rows, seen):
    """Declare missing outcomes from the protocol, without fabricating metrics."""
    expected, missing_data, protocols = {}, [], []
    for manifest in manifests:
        if manifest.get("stage") == "development":
            continue
        observed = 0
        for spec in manifest["specs"]:
            identity, kind = spec["independent_data_id"], spec["kind"]
            if (manifest["manifest_hash"], identity) in seen:
                observed += 1
            else:
                missing_data.append(
                    {
                        "manifest_hash": manifest["manifest_hash"],
                        "independent_data_id": identity,
                        "spec": spec,
                    }
                )
            methods = (
                ("global", "enumeration", "forward", "forward_swap")
                if kind == "small"
                else (
                    ("global", "forward", "forward_swap")
                    if kind == "scaling"
                    else manifest["methods"]
                )
            )
            settings = [
                (method, budget, 1, "marginal_mmd", "comparison")
                for budget in spec["budgets"]
                for method in methods
            ]
            if kind == "anchor":
                settings = [("all_variables", spec["D"], 1, "marginal_mmd", "reference")]
            elif kind == "ablation":
                settings = [
                    (method, 4, factor, "marginal_mmd", "comparison")
                    for factor in spec["metadata"]["radius_factors"]
                    for method in spec["metadata"]["radius_methods"]
                ] + [("global", 4, 1, "random", "comparison")]
            for method, budget, radius, initialization, role in settings:
                group = _group(
                    {"spec": spec, "manifest_hash": manifest["manifest_hash"]},
                    {
                        "method": method,
                        "budget": budget,
                        "radius_factor": radius,
                        "initialization": initialization,
                        "role": role,
                    },
                )
                expected[(_key(group), identity)] = {
                    "group": group,
                    "independent_data_id": identity,
                    "timing_runs": spec.get("metadata", {}).get("timing_repeats", 1),
                }
        protocols.append(
            {
                "path": manifest["path"],
                "manifest_hash": manifest["manifest_hash"],
                "planned_instances": len(manifest["specs"]),
                "valid_record_instances": observed,
                "missing_record_instances": len(manifest["specs"]) - observed,
                "planned_budget_tasks": manifest.get("budget_tasks"),
            }
        )
    actual = {(_key(row["group"]), row["independent_data_id"]): row for row in rows}
    missing_outcomes = []
    grouped = defaultdict(list)
    for key, item in expected.items():
        row = actual.get(key)
        present = row["timing_runs"] if row else 0
        if present < item["timing_runs"]:
            missing_outcomes.append(
                {
                    **item,
                    "observed_timing_runs": present,
                    "missing_timing_runs": item["timing_runs"] - present,
                }
            )
        grouped[_key(item["group"])].append(item)
    return protocols, grouped, missing_data, missing_outcomes


def _trace(result):
    entries = [
        entry
        for entry in result.get("trace", [])
        if _finite(entry.get("seconds")) and entry["seconds"] >= 0
    ]
    if (
        _finite(result.get("runtime"))
        and result["runtime"] >= 0
        and _finite(result.get("upper"))
        and _finite(result.get("lower"))
    ):
        entries.append(
            {
                "seconds": float(result["runtime"]),
                "upper": result["upper"],
                "lower": result["lower"],
                "phase": "stored_final_bounds",
            }
        )
    return sorted(entries, key=lambda entry: entry["seconds"])


def _zero_risk(result):
    """Old records use half-hinge; newly tagged records use standard hinge."""
    return float(result.get("zero_score_risk", 1.0))


def _valid_interval(low, high, zero_risk=1.0):
    return _finite(low) and _finite(high) and 0 <= low <= high and low <= zero_risk + 1e-12


def _search_times(result, kind):
    """Separate obtaining the returned risk from proving its global gap.

    A full-budget fit can establish the final incumbent long before a master
    proves it. An unresolved or inconsistent interval supplies no certificate
    time; it is never converted into a zero-duration observation.
    """
    default = 1e-6 if kind == "small" else 1e-3
    tolerance = result.get(
        "target_tolerance", result.get("options", {}).get("absolute_tolerance", default)
    )
    if not _finite(tolerance) or tolerance <= 0:
        tolerance = default
    entries = _trace(result)
    upper = result.get("upper")
    first = None
    if _finite(upper):
        if abs(upper - _zero_risk(result)) <= 1e-12:
            first = 0.0
        else:
            for entry in entries:
                support = entry.get("support")
                own_fit = (
                    isinstance(support, (list, tuple))
                    and len(support) == result["budget"]
                    and _finite(entry.get("fit_upper"))
                    and entry["fit_upper"] <= upper + 1e-12
                )
                incumbent = _finite(entry.get("upper")) and entry["upper"] <= upper + 1e-12
                if own_fit or incumbent:
                    first = float(entry["seconds"])
                    break
    certified = None
    if (
        result["method"] in ("global", "enumeration")
        and result.get("status") not in INVALID_CERTIFICATE_STATUSES
    ):
        for entry in entries:
            low, high = entry.get("lower"), entry.get("upper")
            if _valid_interval(low, high, _zero_risk(result)) and high - low <= tolerance:
                certified = float(entry["seconds"])
                break
    return {"first_best_time": first, "time_to_gap": certified}, float(tolerance)


def _collapse(attempts):
    by_instance = defaultdict(dict)
    duplicates = []
    for attempt in attempts:
        identity = (_key(attempt["group"]), attempt["independent_data_id"])
        repeat = attempt["timing_repeat"]
        if repeat in by_instance[identity]:
            duplicates.append(
                {
                    "source_path": attempt["source_path"],
                    "independent_data_id": attempt["independent_data_id"],
                    "group": attempt["group"],
                    "timing_repeat": repeat,
                    "reason": "Duplicate attempt; first path in sorted order is used",
                }
            )
        else:
            by_instance[identity][repeat] = attempt
    rows = []
    for repetitions in by_instance.values():
        primary = repetitions[min(repetitions)]
        result = primary["result"]
        metrics = {
            name: float(value)
            for name, value in result.get("metrics", {}).items()
            if _finite(value)
        }
        values = dict(metrics)
        for name in ("upper", "lower", "gap"):
            values[name] = float(result[name]) if _finite(result.get(name)) else None
        for name in (
            "runtime",
            "total_runtime",
            "radius_selection_seconds",
            "initialization_seconds",
            "fit_count",
            "master_count",
            "master_seconds",
            "oracle_seconds",
        ):
            measured = [item["result"].get(name) for item in repetitions.values()]
            measured = [float(value) for value in measured if _finite(value)]
            values[name] = float(np.median(measured)) if measured else None
        times = [
            _search_times(item["result"], item["group"]["kind"])[0] for item in repetitions.values()
        ]
        timing_valid_counts = {}
        for name in ("first_best_time", "time_to_gap"):
            measured = [item[name] for item in times if _finite(item[name])]
            values[name] = float(np.median(measured)) if measured else None
            timing_valid_counts[name] = len(measured)
        _, target_tolerance = _search_times(result, primary["group"]["kind"])
        if primary["independent_optimum"] is not None and _finite(values.get("upper")):
            values["excess_risk"] = values["upper"] - primary["independent_optimum"]
        trace = _trace(result)
        if primary["group"]["method"] == "global":
            for seconds in (10, 60, 300):
                if result.get("status") in INVALID_CERTIFICATE_STATUSES:
                    for bound in ("upper", "lower", "gap"):
                        values[f"{bound}_at_{seconds}s"] = None
                    continue
                before = [entry for entry in trace if entry["seconds"] <= seconds]
                last = before[-1] if before else {"upper": _zero_risk(result), "lower": 0.0}
                if not _valid_interval(last.get("lower"), last.get("upper"), _zero_risk(result)):
                    for bound in ("upper", "lower", "gap"):
                        values[f"{bound}_at_{seconds}s"] = None
                    continue
                for bound in ("upper", "lower"):
                    values[f"{bound}_at_{seconds}s"] = (
                        float(last[bound]) if _finite(last.get(bound)) else None
                    )
                if _finite(values.get(f"upper_at_{seconds}s")) and _finite(
                    values.get(f"lower_at_{seconds}s")
                ):
                    values[f"gap_at_{seconds}s"] = (
                        values[f"upper_at_{seconds}s"] - values[f"lower_at_{seconds}s"]
                    )
        rows.append(
            {
                "group": primary["group"],
                "zero_score_risk": _zero_risk(result),
                "independent_data_id": primary["independent_data_id"],
                "source_path": primary["source_path"],
                "record_status": primary["record_status"],
                "status": result.get("status", "missing"),
                "timing_runs": len(repetitions),
                "timing_valid_counts": timing_valid_counts,
                "target_tolerance": target_tolerance,
                "timing_status_counts": dict(
                    Counter(
                        item["result"].get("status", "missing") for item in repetitions.values()
                    )
                ),
                "values": values,
                "by_subject": result.get("metrics", {}).get("by_subject", {}),
                "trace": trace,
                "shifts": result.get("shifts", []),
            }
        )
    return sorted(
        rows, key=lambda row: (str(_key(row["group"])), row["independent_data_id"])
    ), duplicates


def _subject_values(rows, metric):
    people = defaultdict(list)
    missing = 0
    for row in rows:
        seen = False
        for person, measures in row.get("by_subject", {}).items():
            value = measures.get(metric)
            if _finite(value):
                people[str(person)].append(float(value))
                seen = True
        missing += not seen
    return {person: float(np.mean(values)) for person, values in people.items()}, missing


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[_key(row["group"])].append(row)
    groups = []
    for members in grouped.values():
        group = members[0]["group"]
        metrics = {}
        for name in sorted({name for row in members for name in row["values"]}):
            if group["kind"] == "har" and name in PREDICTION_METRICS:
                values, missing = _subject_values(members, name)
                summary = _stats(list(values.values()))
                summary.update(
                    unit="held_out_person",
                    ci95=_ci(list(values.values()), [group, name]),
                    person_values=values,
                    missing_instances=missing,
                )
            else:
                values = [
                    row["values"][name] for row in members if _finite(row["values"].get(name))
                ]
                unit = {"tabular": "reused_dataset_split", "har": "training_repetition"}.get(
                    group["kind"], "independent_data_instance"
                )
                summary = {
                    **_stats(values),
                    "unit": unit,
                    "ci95": None,
                    "missing_instances": len(members) - len(values),
                }
            if name in ("first_best_time", "time_to_gap"):
                valid = sum(row["timing_valid_counts"][name] for row in members)
                summary["valid_timing_runs"] = valid
                summary["missing_timing_runs"] = sum(row["timing_runs"] for row in members) - valid
                if name == "time_to_gap":
                    summary["target_tolerances"] = sorted(
                        {row["target_tolerance"] for row in members}
                    )
            metrics[name] = summary
        groups.append(
            {
                "group": group,
                "n_instances": len(members),
                "n_timing_runs": sum(row["timing_runs"] for row in members),
                "status_counts": dict(Counter(row["status"] for row in members)),
                "timing_status_counts": dict(
                    sum((Counter(row["timing_status_counts"]) for row in members), Counter())
                ),
                "failed_record_instances": sum(
                    row["record_status"] != "completed" for row in members
                ),
                "metrics": metrics,
            }
        )
    return groups


def _paired(rows):
    conditions = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["group"]["role"] == "comparison":
            conditions[_key(row["group"], ("method",))][row["group"]["method"]][
                row["independent_data_id"]
            ] = row
    comparisons = []
    for methods in conditions.values():
        if "global" not in methods:
            continue
        target = methods["global"]
        for method, baseline in methods.items():
            if method == "global":
                continue
            shared = sorted(set(target) & set(baseline))
            group = next(iter(target.values()))["group"]
            for name in ("ber_prob", "ber_hard", "recall", "precision", "runtime", "excess_risk"):
                if group["kind"] == "har" and name in PREDICTION_METRICS:
                    people = defaultdict(list)
                    contributing = 0
                    for identity in shared:
                        left, right = (
                            baseline[identity]["by_subject"],
                            target[identity]["by_subject"],
                        )
                        used = False
                        for person in set(left) & set(right):
                            a, b = left[person].get(name), right[person].get(name)
                            if _finite(a) and _finite(b):
                                people[str(person)].append(float(a - b))
                                used = True
                        contributing += used
                    differences = [float(np.mean(values)) for values in people.values()]
                    unit = "held_out_person"
                else:
                    differences = [
                        baseline[identity]["values"][name] - target[identity]["values"][name]
                        for identity in shared
                        if _finite(baseline[identity]["values"].get(name))
                        and _finite(target[identity]["values"].get(name))
                    ]
                    contributing = len(differences)
                    unit = {"tabular": "reused_dataset_split", "har": "training_repetition"}.get(
                        group["kind"], "independent_data_instance"
                    )
                if not differences:
                    continue
                # Ten random splits reuse observations. Their differences and
                # SD are descriptive; no independent-split bootstrap is added.
                descriptive = group["kind"] == "tabular" or (
                    group["kind"] == "har" and name not in PREDICTION_METRICS
                )
                interval = None if descriptive else _ci(differences, [group, method, name])
                comparisons.append(
                    {
                        "group": group,
                        "baseline": method,
                        "target": "global",
                        "metric": name,
                        "difference": "baseline minus global",
                        **_stats(differences),
                        "ci95": interval,
                        "unit": unit,
                        "paired_training_instances": contributing,
                        "available_global_instances": len(target),
                        "available_baseline_instances": len(baseline),
                        "shared_instance_ids": len(shared),
                        "missing_metric_pairs": len(shared) - contributing,
                    }
                )
    return comparisons


def _shift_groups(rows):
    grouped = defaultdict(list)
    condition_counts = Counter(_key(row["group"]) for row in rows)
    for row in rows:
        for shift in row["shifts"]:
            group = {
                **row["group"],
                "shift_kind": shift["kind"],
                "delta_factor": shift["delta_factor"],
            }
            key = (_key(row["group"]), shift["kind"], shift["delta_factor"])
            grouped[key].append(
                {
                    "group": group,
                    "metrics": shift.get("metrics", {}),
                    "independent_data_id": row["independent_data_id"],
                }
            )
    result = []
    for members in grouped.values():
        denominator = condition_counts[_key(members[0]["group"])]
        measures = {}
        for name in sorted(
            {
                key
                for member in members
                for key, value in member["metrics"].items()
                if _finite(value)
            }
        ):
            values = [
                float(member["metrics"][name])
                for member in members
                if _finite(member["metrics"].get(name))
            ]
            measures[name] = {**_stats(values), "missing_instances": denominator - len(values)}
        result.append(
            {
                "group": members[0]["group"],
                "n_instances": denominator,
                "n_shift_records": len(members),
                "missing_shift_instances": denominator - len(members),
                "metrics": measures,
            }
        )
    return result


def _title(group, excluded=()):
    labels = {"D": "D", "n": "n/class", "s": "s", "budget": "d"}
    items = [
        f"{labels[key]}={group[key]}" for key in labels if key in group and key not in excluded
    ]
    if group.get("radius_factor", 1) != 1:
        items.append(f"radius ×{group['radius_factor']}")
    if group.get("initialization") == "random":
        items.append("random start")
    return f"{group.get('suite', '')}: {group.get('model', '')}\n" + ", ".join(items)


def _figures(destination, groups, paired, shifts, rows):
    if not groups:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = destination / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.dpi": 160,
        }
    )

    def save(fig, name, identity):
        path = figure_root / f"{name}-{fingerprint(identity)[:12]}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    def curves(collection, x_name, metric, prefix, extra_excluded=(), ylabel=None):
        panels = defaultdict(list)
        for item in collection:
            group = item["group"]
            if (
                group["role"] != "comparison"
                or metric not in item["metrics"]
                or item["metrics"][metric]["mean"] is None
            ):
                continue
            excluded = {"method", x_name, *extra_excluded}
            panel = tuple((key, value) for key, value in group.items() if key not in excluded)
            panels[panel].append(item)
        for panel, members in panels.items():
            if len({item["group"][x_name] for item in members}) < 2:
                continue
            fig, axis = plt.subplots(figsize=(6.5, 4.2))
            for method in sorted({item["group"]["method"] for item in members}):
                selected = sorted(
                    (item for item in members if item["group"]["method"] == method),
                    key=lambda item: item["group"][x_name],
                )
                axis.errorbar(
                    [item["group"][x_name] for item in selected],
                    [item["metrics"][metric]["mean"] for item in selected],
                    yerr=[item["metrics"][metric]["sd"] or 0 for item in selected],
                    marker="o",
                    capsize=2,
                    linewidth=1.3,
                    label=method,
                )
            labels = {
                "budget": "Variables selected (d)",
                "D": "Candidate variables (D)",
                "n": "Training samples per class (n)",
                "delta_factor": "Shift size / mean training radius",
            }
            axis.set(
                xlabel=labels.get(x_name, x_name),
                ylabel=ylabel or metric,
                title=_title(dict(panel), excluded=(x_name,)),
            )
            axis.set_xticks(sorted({item["group"][x_name] for item in members}))
            axis.set_ylim(bottom=0)
            if metric == "ber_prob":
                axis.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Chance (0.5)")
            axis.legend(fontsize=7, ncol=2)
            save(fig, prefix, [panel, metric])

    main = [
        group
        for group in groups
        if group["group"]["kind"] in ("synthetic", "tabular", "har")
        and group["group"]["suite"] not in ("shift", "ablation")
    ]
    curves(main, "budget", "ber_prob", "E1-error", ylabel="Balanced randomized error (mean ± SD)")
    curves(main, "budget", "recall", "E1-recall", ylabel="Signal recall (mean ± SD)")
    curves(
        shifts,
        "delta_factor",
        "ber_prob",
        "E2-shift",
        ylabel="Balanced randomized error (mean ± SD)",
    )
    scaling = [group for group in groups if group["group"]["kind"] == "scaling"]
    for axis in ("D", "n", "budget"):
        curves(scaling, axis, "runtime", "E3-time", ylabel="Runtime in seconds (mean ± SD)")

    panels = defaultdict(list)
    for comparison in paired:
        if comparison["metric"] == "ber_prob":
            panels[_key(comparison["group"], ("method", "budget"))].append(comparison)
    for panel, comparisons in panels.items():
        comparisons.sort(key=lambda item: (item["group"]["budget"], item["baseline"]))
        fig, axis = plt.subplots(figsize=(7, max(3, 0.3 * len(comparisons))))
        y = np.arange(len(comparisons))
        means = np.array([item["mean"] for item in comparisons])
        intervals = [item["ci95"] for item in comparisons]
        axis.scatter(means, y, s=20)
        for i, interval in enumerate(intervals):
            half_width = comparisons[i]["sd"] or 0
            low, high = interval if interval else (means[i] - half_width, means[i] + half_width)
            axis.hlines(y[i], low, high, linewidth=1.2)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(
            y, [f"{item['baseline']} (d={item['group']['budget']})" for item in comparisons]
        )
        kind = dict(panel)["kind"]
        interval_label = (
            "Bars: paired split SD" if kind == "tabular" else "Bars: 95% paired bootstrap interval"
        )
        if kind == "har":
            interval_label += " over held-out people"
        axis.set(
            xlabel="Baseline BER minus global BER; positive favors global",
            title=_title(dict(panel), excluded=("budget",)) + "\n" + interval_label,
        )
        save(fig, "E1-paired-baselines", panel)

    traced = defaultdict(list)
    for row in rows:
        if (
            row["group"]["method"] == "global"
            and row["group"]["kind"] in ("small", "scaling")
            and row["status"] not in INVALID_CERTIFICATE_STATUSES
        ):
            traced[_key(row["group"])].append(row)
    for panel, members in traced.items():
        row = min(members, key=lambda member: member["independent_data_id"])
        entries = [
            {"seconds": 0, "upper": _zero_risk(row), "lower": 0.0},
            *[
                entry
                for entry in row["trace"]
                if _valid_interval(entry.get("lower"), entry.get("upper"), _zero_risk(row))
            ],
        ]
        fig, axis = plt.subplots(figsize=(6.5, 4))
        for bound in ("upper", "lower"):
            axis.step(
                [entry["seconds"] for entry in entries],
                [entry[bound] for entry in entries],
                where="post",
                label=bound,
            )
        axis.set(
            xlabel="Elapsed seconds",
            ylabel="Certified risk bound",
            title=_title(row["group"]) + "\nFirst data ID: " + row["independent_data_id"],
        )
        axis.legend()
        save(fig, "E3-bound-trace", panel)

    exact = defaultdict(list)
    for group in groups:
        if (
            group["group"]["kind"] == "small"
            and group["group"]["role"] == "comparison"
            and "excess_risk" in group["metrics"]
        ):
            exact[_key(group["group"], ("method",))].append(group)
    for panel, members in exact.items():
        fig, axis = plt.subplots(figsize=(6.5, 4))
        for item in members:
            x, y = item["metrics"].get("runtime", {}), item["metrics"]["excess_risk"]
            if x.get("mean") is not None and y["mean"] is not None:
                axis.errorbar(
                    x["mean"],
                    y["mean"],
                    xerr=x.get("sd") or 0,
                    yerr=y.get("sd") or 0,
                    marker="o",
                    capsize=2,
                    label=item["group"]["method"],
                )
        axis.set(
            xlabel="Runtime in seconds (mean ± SD)",
            ylabel="Feasible risk minus independent optimum",
            title=_title(dict(panel)),
        )
        axis.legend(fontsize=8)
        save(fig, "E3-excess-risk-time", panel)
    return paths


def summarize(output_root, destination=None):
    """Write summaries, complete denominator tables and condition-specific figures.

    The function reads records only; it never reruns a fit, edits a manifest or
    treats missing records as completed experiments. Malformed/failed records
    and duplicate attempts are explicitly listed in summary.json.
    """
    root = Path(output_root).resolve()
    base = root.parent if root.name == "records" else root
    destination = Path(destination).resolve() if destination is not None else base / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    manifests, protocol_errors = _protocols(root)
    paths, attempts, failures, errors, excluded, seen = _load(root, manifests)
    errors = protocol_errors + errors
    rows, duplicates = _collapse(attempts)
    groups, paired, shifts = _aggregate(rows), _paired(rows), _shift_groups(rows)
    protocols, planned_groups, missing_data, missing_outcomes = _planned(manifests, rows, seen)
    observed_groups = {_key(item["group"]): item for item in groups}
    actual = {(_key(row["group"]), row["independent_data_id"]): row for row in rows}
    for key, planned in planned_groups.items():
        missing = sum((key, item["independent_data_id"]) not in actual for item in planned)
        if key not in observed_groups:
            empty = {
                "group": planned[0]["group"],
                "n_instances": 0,
                "n_timing_runs": 0,
                "status_counts": {},
                "timing_status_counts": {},
                "failed_record_instances": 0,
                "metrics": {
                    name: {
                        "n": 0,
                        "mean": None,
                        "sd": None,
                        "unit": "unobserved",
                        "ci95": None,
                        "missing_instances": 0,
                    }
                    for name in ("ber_prob", "runtime", "first_best_time", "time_to_gap")
                },
            }
            groups.append(empty)
            observed_groups[key] = empty
        group = observed_groups[key]
        group.update(
            planned_instances=len(planned),
            missing_outcome_instances=missing,
            planned_timing_runs=sum(item["timing_runs"] for item in planned),
            missing_timing_runs=sum(
                max(
                    0,
                    item["timing_runs"]
                    - actual.get((key, item["independent_data_id"]), {}).get("timing_runs", 0),
                )
                for item in planned
            ),
        )
        if missing:
            group["status_counts"]["missing_outcome"] = missing
        for name, measure in group["metrics"].items():
            measure["missing_planned_instances"] = measure["missing_instances"] + missing
            if name in ("first_best_time", "time_to_gap"):
                measure.setdefault("valid_timing_runs", 0)
                measure.setdefault("missing_timing_runs", 0)
    for shift in shifts:
        key = _key(shift["group"])
        if key in observed_groups and "planned_instances" in observed_groups[key]:
            counts = observed_groups[key]
            shift["planned_instances"] = counts["planned_instances"]
            for measure in shift["metrics"].values():
                measure["missing_planned_instances"] = (
                    measure["missing_instances"] + counts["missing_outcome_instances"]
                )
    for comparison in paired:
        key = _key(comparison["group"])
        if key in observed_groups and "planned_instances" in observed_groups[key]:
            comparison["planned_instances"] = observed_groups[key]["planned_instances"]
            comparison["missing_planned_training_pairs"] = max(
                0, comparison["planned_instances"] - comparison["paired_training_instances"]
            )
    summary = {
        "schema_version": 1,
        "source_root": str(root),
        "record_files": len(paths),
        "observed_method_attempts": len(attempts),
        "collapsed_instance_method_rows": len(rows),
        "failed_instances": failures,
        "load_errors": errors,
        "duplicate_attempts": duplicates,
        "excluded_development_instances": excluded,
        "protocol_coverage": protocols,
        "planned_missing_records": missing_data,
        "planned_missing_outcomes": missing_outcomes,
        "sampling_rules": {
            "timing": "Prediction metrics use the lowest timing_repeat (normally 0); costs use per-instance medians over timing attempts.",
            "search_times": "first_best_time is the earliest known full-budget final-risk incumbent (zero for the initial constant-score test at the recorded loss scale). time_to_gap is the earliest valid global/enumeration interval meeting its declared tolerance. Each time uses the median of reached timing runs; unreached runs remain null and valid-run counts are reported. Explicit certificate failures suppress proof-time and checkpoint-interval statistics while retaining raw traces.",
            "synthetic": "Means and sample SD across data IDs; baseline-minus-global differences use 2000 paired bootstrap resamples.",
            "tabular": "Paired split differences and SD only; random splits reuse observations and are not independent test populations.",
            "har": "Average training repetitions within each held-out person, then weight people equally; bootstrap people, not windows or training repetitions.",
            "failures": "Available finite metrics include timeouts and fallback tests. Missing metrics and failed records remain in denominator tables.",
            "scope": "Every summary retains manifest, suite, data kind/model, D, n, s, budget, radius, role and initialization. Frozen plans declare missing outcomes; no missing metric is imputed.",
        },
        "groups": groups,
        "paired_comparisons": paired,
        "shift_groups": shifts,
    }
    files = {}
    instance_rows = [
        {
            **row["group"],
            "independent_data_id": row["independent_data_id"],
            "record_status": row["record_status"],
            "status": row["status"],
            "timing_runs": row["timing_runs"],
            "target_tolerance": row["target_tolerance"],
            "first_best_time_valid_timing_runs": row["timing_valid_counts"]["first_best_time"],
            "time_to_gap_valid_timing_runs": row["timing_valid_counts"]["time_to_gap"],
            **row["values"],
        }
        for row in rows
    ]
    tables = {
        "instances": instance_rows,
        "timing_runs": [
            {
                **item["group"],
                "independent_data_id": item["independent_data_id"],
                "timing_repeat": item["timing_repeat"],
                "record_status": item["record_status"],
                "status": item["result"].get("status", "missing"),
                "runtime": item["result"].get("runtime"),
                **_search_times(item["result"], item["group"]["kind"])[0],
                "source_path": item["source_path"],
            }
            for item in attempts
        ],
        "groups": [
            {
                **item["group"],
                "n_instances": item["n_instances"],
                "n_timing_runs": item["n_timing_runs"],
                "failed_record_instances": item["failed_record_instances"],
                "planned_instances": item.get("planned_instances"),
                "missing_outcome_instances": item.get("missing_outcome_instances"),
                "planned_timing_runs": item.get("planned_timing_runs"),
                "missing_timing_runs": item.get("missing_timing_runs"),
                "status_counts": str(item["status_counts"]),
                "metric": name,
                **{key: value for key, value in measure.items() if key != "person_values"},
            }
            for item in groups
            for name, measure in item["metrics"].items()
        ],
        "paired_comparisons": [
            {**item["group"], **{key: value for key, value in item.items() if key != "group"}}
            for item in paired
        ],
        "shifts": [
            {
                **item["group"],
                "n_instances": item["n_instances"],
                "n_shift_records": item["n_shift_records"],
                "missing_shift_instances": item["missing_shift_instances"],
                "metric": name,
                **measure,
            }
            for item in shifts
            for name, measure in item["metrics"].items()
        ],
        "failed_instances": [
            {
                "source_path": item["path"],
                "independent_data_id": item["independent_data_id"],
                "status": item["status"],
                "error": item["error"],
            }
            for item in failures
        ],
        "planned_missing_outcomes": [
            {
                **item["group"],
                "independent_data_id": item["independent_data_id"],
                "planned_timing_runs": item["timing_runs"],
                "observed_timing_runs": item["observed_timing_runs"],
                "missing_timing_runs": item["missing_timing_runs"],
            }
            for item in missing_outcomes
        ],
    }
    for name, table in tables.items():
        path = destination / f"{name}.csv"
        _write_csv(path, table)
        files[name] = str(path)
    summary["figure_errors"] = []
    try:
        summary["figures"] = _figures(destination, groups, paired, shifts, rows)
    except (ImportError, OSError, ValueError) as error:
        summary["figures"] = []
        summary["figure_errors"].append(str(error))
    summary["files"] = files
    summary_path = destination / "summary.json"
    write_json(summary_path, summary)
    return {
        "summary": str(summary_path),
        "tables": files,
        "figures": summary["figures"],
        "record_files": len(paths),
        "groups": len(groups),
        "failed_records": len(failures),
        "load_errors": len(errors),
        "planned_missing_records": len(missing_data),
        "planned_missing_outcomes": len(missing_outcomes),
    }
