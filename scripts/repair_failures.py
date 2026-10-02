"""Repair the complete numerical-failure cohort with a shared selection budget.

Run only after both original runners have stopped. Original records/traces are
archived before any replacement; compact original records remain published.
The restart rebuilds the same problem, rather than pretending to resume cuts.
"""
import argparse
import copy
import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from robusthw.experiments.data import synthetic_trial
from robusthw.experiments.runner import (
    _global_selection,
    _problem,
    _write_json,
    evaluate,
)
from robusthw.problem import json_value

POLICY = "shared-budget-presolve-retry-v1"
FAILURES = {"master_failure", "oracle_failure", "numerical_failure", "numerical_inconsistency"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore_trial(record, config, suite):
    """Use the frozen generator and require exact agreement with stored metadata."""
    current = copy.deepcopy(config)
    if suite == "computational":
        settings = [s for s in config["settings"]
                    if s["dimension"] == record["dimension"]
                    and s["budget"] == record["budget"]
                    and s.get("n_select", 20) == record["split_sizes"]["selection"][0]
                    and s.get("mechanism") == record["dataset"]]
        if len(settings) != 1:
            raise ValueError("Cannot uniquely identify the original computational setting")
        current.update(settings[0])
        current.pop("settings", None)
    if record["seed"] not in current["seeds"]:
        raise ValueError("Original seed is outside the frozen protocol")
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if record["config_sha256"] != digest:
        raise ValueError("Original record/configuration hash mismatch")
    arguments = {key: int(current[key]) for key in
                 ("dimension", "signal_size", "n_select", "n_head", "n_classification")
                 if key in current}
    trial = synthetic_trial(record["dataset"], record["seed"], **arguments)
    metadata = json_value(trial.as_dict())
    for key, value in metadata.items():
        if record[key] != value:
            raise ValueError(f"Restored trial disagrees with original {key}")
    if current.get("theta", .25) != record["theta"]:
        raise ValueError("Restored radius differs from original")
    return trial, current


def binary_supports(trace):
    return {tuple(int(j) for j in row["support"]) for row in trace
            if row.get("support") is not None}


def merge_selection(initial, restart, initial_trace, restart_trace, *, budget):
    """Combine two valid intervals and retain the better feasible incumbent.

    The original oracle's feasible upper bound remains valid after a master
    failure. Common evaluation heads are rebuilt separately for the selected
    support, so no discarded selection test is deployed.
    """
    for attempt in (initial, restart):
        if not np.isfinite(attempt["lower"]) or not np.isfinite(attempt["upper"]):
            raise ValueError("Cannot merge nonfinite selection bounds")
        if attempt["lower"] > attempt["upper"] + 1e-7:
            raise ValueError("Cannot retain an inconsistent initial/restart certificate")
    best = initial if initial["upper"] <= restart["upper"] else restart
    lower = max(initial["lower"], restart["lower"])
    upper = best["upper"]
    gap = max(0., upper - lower)
    status = restart["status"]
    if lower > upper + 1e-7:
        status = "numerical_inconsistency"
    elif gap <= 1e-6 + 1e-5 * abs(upper):
        status = "optimal"
    elif status == "optimal":
        status = "oracle_accuracy_limit"
    diagnostics = copy.deepcopy(restart.get("diagnostics", {}))
    old_diag, new_diag = initial.get("diagnostics", {}), restart.get("diagnostics", {})
    for component in ("oracle_seconds", "master_seconds"):
        old, new = float(old_diag.get(component, 0.)), float(new_diag.get(component, 0.))
        diagnostics[component] = old + new
        diagnostics["initial_" + component] = old
        diagnostics["restart_" + component] = new
    diagnostics["root_bound"] = max(old_diag.get("root_bound", 0.),
                                    new_diag.get("root_bound", 0.))
    diagnostics["master_retries"] = (old_diag.get("master_retries", 0)
                                       + new_diag.get("master_retries", 0))
    diagnostics["master_retry_log"] = [dict(attempt=label, **row)
        for label, diag in (("initial", old_diag), ("restart", new_diag))
        for row in diag.get("master_retry_log", [])]
    old_supports, new_supports = binary_supports(initial_trace), binary_supports(restart_trace)
    # Do not report the sum of visits as distinct visits across a restart.
    if len(old_supports) != initial["visited"] or len(new_supports) != restart["visited"]:
        raise ValueError("Trace does not certify the distinct visited-support count")
    diagnostics.update(initial_visited=initial["visited"], restart_visited=restart["visited"],
                       effort_visited=initial["visited"] + restart["visited"],
                       visited_definition="distinct binary supports across both attempts",
                       initial_oracle_calls=initial["oracle_calls"],
                       restart_oracle_calls=restart["oracle_calls"],
                       initial_selection_seconds=initial["seconds"],
                       restart_selection_seconds=restart["seconds"],
                       selection_time_budget=budget)
    merged = {key: copy.deepcopy(value) for key, value in restart.items() if key != "trace"}
    merged.update(support=best["support"], lower=lower, upper=upper, gap=gap, status=status,
                  seconds=initial["seconds"] + restart["seconds"],
                  oracle_calls=initial["oracle_calls"] + restart["oracle_calls"],
                  visited=len(old_supports | new_supports), diagnostics=diagnostics)
    trace = [dict(row, attempt="initial") for row in initial_trace]
    for row in restart_trace:
        copied = dict(row, attempt="restart", seconds=initial["seconds"] + row["seconds"])
        if "lower" in copied:
            copied["lower"] = max(initial["lower"], copied["lower"])
        if "upper" in copied:
            copied["upper"] = min(initial["upper"], copied["upper"])
        trace.append(copied)
    return merged, trace, "initial" if best is initial else "restart"


def freeze_cohort(results, project, archive):
    """Freeze ALL initial failures once; public metadata survives a clean clone."""
    public = results / "remediation" / "initial_failures.json"
    if public.exists():
        return read_json(public)
    entries = []
    for suite in ("computational", "statistical"):
        for path in sorted((results / suite / "records").glob("*.json")):
            record = read_json(path)
            if record.get("selection", {}).get("status") not in FAILURES:
                continue
            relative = path.relative_to(results)
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != path.read_bytes():
                raise ValueError(f"Conflicting prior archive: {destination}")
            shutil.copy2(path, destination)
            trace_source = results / suite / record["trace_file"]
            trace_destination = archive / suite / record["trace_file"]
            trace_destination.parent.mkdir(parents=True, exist_ok=True)
            if trace_destination.exists() and trace_destination.read_bytes() != trace_source.read_bytes():
                raise ValueError(f"Conflicting prior trace archive: {trace_destination}")
            shutil.copy2(trace_source, trace_destination)
            entry = {**record, "suite": suite, "record_relative_path": relative.as_posix(),
                     "archive_record_path": destination.relative_to(project).as_posix(),
                     "archive_trace_path": trace_destination.relative_to(project).as_posix(),
                     "original_record_sha256": sha256(path.read_bytes()).hexdigest()}
            entries.append(entry)
    _write_json(public, entries)
    return entries


def repair_one(entry, results, project):
    path = results / entry["record_relative_path"]
    existing = read_json(path)
    if existing.get("numerical_remediation", {}).get("policy") == POLICY:
        return existing, True
    archived = project / entry["archive_record_path"]
    if sha256(archived.read_bytes()).hexdigest() != entry["original_record_sha256"]:
        raise ValueError("Archived original record checksum mismatch")
    record = read_json(archived)
    config = read_json(results / entry["suite"] / "config.json")
    trial, current = restore_trial(record, config, entry["suite"])
    initial = record["selection"]
    initial_trace = read_json(project / entry["archive_trace_path"])
    budget = float(current.get("time_limit", 30.))
    remaining = max(0., budget - float(initial["seconds"]))
    tick = perf_counter()
    if remaining > 0:
        restart = _global_selection(_problem(trial.selection, record["budget"], record["theta"]),
                                    {**current, "time_limit": remaining}, record["method"])
        restart_trace = restart.pop("trace")
    else:
        restart = dict(support=initial["support"], status="time_limit", seconds=0.,
                       oracle_calls=0, visited=0, lower=0., upper=1., gap=1.,
                       weights=None, diagnostics={"message": "No shared selection budget remains"})
        restart_trace = []
    merged, trace, incumbent = merge_selection(initial, restart, initial_trace, restart_trace,
                                                budget=budget)
    record["selection"] = merged
    record["semantic_support"] = trial.semantic_support(merged["support"])
    record["selection_timed_out"] = "time_limit" in merged["status"]
    record["selection_failed"] = merged["status"] in FAILURES
    initial_evaluation_seconds = float(record.get("evaluation_seconds", 0.))
    restart_evaluation_seconds = 0.
    if entry["suite"] == "statistical":
        # Always refit every common head/test for the final selected support.
        evaluation = evaluate(trial, merged, current)
        restart_evaluation_seconds = float(evaluation["evaluation_seconds"])
        record.update(evaluation)
        record["evaluation_seconds"] = initial_evaluation_seconds + restart_evaluation_seconds
    restart_total = perf_counter() - tick
    initial_total = float(record["total_seconds"])
    record["total_seconds"] = initial_total + restart_total
    record["numerical_remediation"] = dict(
        policy=POLICY, initial_status=initial["status"], restart_status=restart["status"],
        final_status=merged["status"], initial_selection_seconds=initial["seconds"],
        restart_selection_seconds=restart["seconds"], selection_time_budget=budget,
        restart_time_budget=remaining, initial_total_seconds=initial_total,
        restart_total_seconds=restart_total, initial_evaluation_seconds=initial_evaluation_seconds,
        restart_evaluation_seconds=restart_evaluation_seconds, incumbent_source=incumbent,
        initial_archive_record=entry["archive_record_path"],
        initial_archive_trace=entry["archive_trace_path"],
        original_record_sha256=entry["original_record_sha256"],
        initial_message=initial.get("diagnostics", {}).get("message", ""),
        restart_resumes_cuts=False,
        restored_selection_sha256=sha256(np.vstack(trial.selection).tobytes()).hexdigest())
    _write_json(results / entry["suite"] / record["trace_file"], trace)
    _write_json(path, record)
    return record, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, default=Path("results/research_20261003"))
    parser.add_argument("--confirm-original-runners-finished", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_original_runners_finished:
        parser.error("Both original runners must have stopped before freezing the failure cohort")
    project = args.project_root.resolve()
    results = args.results if args.results.is_absolute() else project / args.results
    archive = project / "_archive/numerical_failures_20261003"
    entries = freeze_cohort(results, project, archive)
    statuses = Counter()
    rerun, skipped = 0, 0
    with threadpool_limits(limits=1):
        for entry in entries:
            record, was_skipped = repair_one(entry, results, project)
            skipped += int(was_skipped)
            rerun += int(not was_skipped)
            selection = record["selection"]
            statuses[selection["status"]] += 1
            print(f"{entry['record_relative_path']}: {selection['status']} "
                  f"({selection['seconds']:.3f}s charged selection)", flush=True)
    summary = dict(policy=POLICY, initial_failure_count=len(entries), repaired_this_run=rerun,
                   previously_repaired=skipped, final_status_counts=dict(statuses),
                   shared_budget=True, initial_provenance="remediation/initial_failures.json")
    _write_json(results / "remediation" / "repair_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
