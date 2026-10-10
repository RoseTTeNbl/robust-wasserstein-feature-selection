"""Matched optimization validation; never overwrites the frozen study.

Run with .venv-research/Scripts/python.exe scripts/validate_adaptive_search.py.
Results concern optimization on generated data, not population testing power.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import cvxpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from robusthw import GateOracle, Problem
import robusthw.solver as outer


class MeasuredOracle(GateOracle):
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MeasuredOracle.last = self
        self.full_calls = self.reprice_calls = 0

    def solve(self, *args, **kwargs):
        self.full_calls += 1
        return super().solve(*args, **kwargs)

    def reprice(self, *args, **kwargs):
        self.reprice_calls += 1
        return super().reprice(*args, **kwargs)


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def generated_case(loss, dimension, n, seed, budget=2, group="primary"):
    rng = np.random.default_rng(seed)
    scales = np.r_[1., 1., np.geomspace(.3, 3., dimension - 2)]
    x0 = rng.normal(size=(n, dimension)) * scales
    x1 = rng.normal(size=(n, dimension)) * scales
    x1[:, :2] += [1.4, .9]
    problem = Problem(x0, x1, budget, .18, .27, loss)
    case_id = f"{group}_{loss}_d{dimension}_n{n}_s{seed}_b{budget}"
    parameters = dict(id=case_id, group=group, loss=loss, dimension=dimension,
                      per_class_n=n, seed=seed, budget=budget, signal_coordinates=[0, 1],
                      mean_shift=[1.4, .9], coordinate_scales=scales.tolist(),
                      theta=problem.theta.tolist(),
                      warm_support=list(range(dimension - budget, dimension)))
    arrays = dict(x0=x0.tolist(), x1=x1.tolist(),
                  q0=problem.q0.tolist(), q1=problem.q1.tolist())
    parameters["data_sha256"] = hashlib.sha256(
        np.r_[x0.ravel(), x1.ravel(), problem.q0, problem.q1].tobytes()).hexdigest()
    return problem, parameters, arrays


def run_method(problem, method, settings, seed, warm_support):
    started = perf_counter()
    try:
        if method == "enumeration":
            result = outer.enumerate_supports(problem, time_limit=settings["time_limit"],
                                               tolerance=settings["tolerance"])
        else:
            result = outer.solve(problem, search=method, random_state=seed,
                                 warm_support=warm_support, **settings)
        oracle = MeasuredOracle.last
        record = result.to_dict()
        record["test_certificate"] = result.test.as_dict() if hasattr(result.test, "as_dict") else None
        record["measured"] = dict(full_calls=oracle.full_calls, repricings=oracle.reprice_calls,
                                  model_reuse=oracle.reuse_statistics,
                                  external_seconds=perf_counter() - started)
        return record
    except Exception as error:
        return dict(status="exception", exception_type=type(error).__name__,
                    message=str(error), seconds=perf_counter() - started)


def comparison(record, tolerance):
    reference = record["methods"]["enumeration"]
    ref_complete = reference["status"] == "optimal" and reference.get("gap", np.inf) <= tolerance
    rows = []
    for method in ("master", "adaptive"):
        result = record["methods"][method]
        row = {**record["case"], "method": method, "status": result["status"],
               "reference_complete": ref_complete}
        for key in ("lower", "upper", "gap", "seconds", "oracle_calls", "visited"):
            row[key] = result.get(key)
        measured = result.get("measured", {})
        diagnostics = result.get("diagnostics", {})
        row["repricings"] = measured.get("repricings")
        row["reused_fraction_cuts"] = diagnostics.get("reused_fraction_cuts", 0)
        row["model_reuse"] = measured.get("model_reuse", {})
        row["refinements"] = diagnostics.get("refinements", 0)
        row["accurate_support_evaluations"] = diagnostics.get("accurate_support_evaluations")
        row["anchor_target"] = diagnostics.get("anchor_target")
        gaps = diagnostics.get("anchor_gaps", [])
        row["anchor_condition_met"] = (all(g <= diagnostics["anchor_target"] + 1e-10 for g in gaps)
                                        if gaps and "anchor_target" in diagnostics else None)
        row["epsilon_terminated"] = result["status"] == "optimal" and result.get("gap", np.inf) <= tolerance
        row["reference_interval_consistent"] = (result["lower"] <= reference["upper"] + tolerance
                                                and result["upper"] >= reference["lower"] - tolerance
                                                if "lower" in result and "lower" in reference else None)
        row["upper_excess_over_reference_lower"] = (result["upper"] - reference["lower"]
                                                     if "upper" in result and "lower" in reference else None)
        row["master_guard_calls"] = diagnostics.get("master_calls", sum(
            entry["phase"] == "master" for entry in result.get("trace", [])))
        rows.append(row)
    return rows


def aggregate(rows, group):
    selected = [row for row in rows if row["group"] == group]
    output = {}
    for method in ("master", "adaptive"):
        subset = [row for row in selected if row["method"] == method]
        output[method] = dict(cases=len(subset), optimal=sum(row["epsilon_terminated"] for row in subset),
                              reference_consistent=sum(row["reference_interval_consistent"] is True for row in subset),
                              statuses={status: sum(row["status"] == status for row in subset)
                                        for status in sorted({row["status"] for row in subset})})
        for field in ("seconds", "oracle_calls", "repricings", "reused_fraction_cuts", "master_guard_calls"):
            values = [row[field] for row in subset if row.get(field) is not None]
            output[method][field] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                          total=float(np.sum(values))) if values else None
    paired = []
    for case in sorted({row["id"] for row in selected}):
        by_method = {row["method"]: row for row in selected if row["id"] == case}
        a, m = by_method["adaptive"], by_method["master"]
        if a["oracle_calls"] is not None and m["oracle_calls"] is not None:
            paired.append(dict(id=case, adaptive_calls=a["oracle_calls"], master_calls=m["oracle_calls"],
                               adaptive_seconds=a["seconds"], master_seconds=m["seconds"],
                               call_ratio=a["oracle_calls"] / m["oracle_calls"],
                               runtime_ratio=a["seconds"] / m["seconds"],
                               both_optimal=a["epsilon_terminated"] and m["epsilon_terminated"]))
    output["pairs"] = paired
    output["paired_counts"] = dict(
        fewer_calls=sum(p["adaptive_calls"] < p["master_calls"] for p in paired),
        equal_calls=sum(p["adaptive_calls"] == p["master_calls"] for p in paired),
        more_calls=sum(p["adaptive_calls"] > p["master_calls"] for p in paired),
        faster=sum(p["adaptive_seconds"] < p["master_seconds"] for p in paired),
        slower=sum(p["adaptive_seconds"] >= p["master_seconds"] for p in paired),
        median_call_ratio=float(np.median([p["call_ratio"] for p in paired])) if paired else None,
        median_runtime_ratio=float(np.median([p["runtime_ratio"] for p in paired])) if paired else None)
    return output


def plot(rows, output):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")
    primary = [row for row in rows if row["group"] == "primary"]
    for loss, color, marker in (("overlap", "#4C79A5", "o"), ("logistic", "#C88645", "s")):
        pairs = []
        for case in sorted({row["id"] for row in primary if row["loss"] == loss}):
            selected = {row["method"]: row for row in primary if row["id"] == case}
            if all(selected[method]["seconds"] is not None for method in ("master", "adaptive")):
                pairs.append(selected)
        label = "Half hinge" if loss == "overlap" else "Logistic"
        for ax, field in zip(axes, ("oracle_calls", "seconds")):
            ax.scatter([pair["master"][field] for pair in pairs],
                       [pair["adaptive"][field] for pair in pairs], color=color,
                       marker=marker, s=37, alpha=.78, label=label, edgecolor="white", linewidth=.5)
    for ax, field, title in zip(axes, ("oracle_calls", "seconds"),
                                ("Full oracle calls", "Total runtime (seconds)")):
        values = [row[field] for row in primary if row[field] is not None]
        if field == "seconds":
            limits = (min(values) * .72, max(values) * 1.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            limits = (0, max(values) + 1)
        ax.plot(limits, limits, color=".4", linewidth=1., linestyle="--", zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Master-only search")
        ax.set_ylabel("History-guided search")
        ax.text(.97, .04, "Below line: adaptive uses less", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        ax.grid(alpha=.12, linewidth=.5)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"New matched validation: {len(primary)//2} cases, 60-second limit", fontsize=11)
    fig.savefig(output / "calls_and_runtime.svg")
    fig.savefig(output / "calls_and_runtime.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/verification/20261010/adaptive_search")
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=60.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    settings = dict(time_limit=args.time_limit, root_rounds=0, tolerance=2e-5,
                    relative_tolerance=0., proposal_count=args.proposal_count,
                    core_point=False, gate="risk", cut_mode="tangent")
    sources = [ROOT / "code/robusthw" / f"{name}.py" for name in
               ("problem", "oracle", "solver", "history", "adaptive")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = dict(scope="New optimization validation; separate from frozen 1640-study records",
                    settings=settings, sources=hashes,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, cvxpy=cvxpy.__version__, matplotlib=matplotlib.__version__),
                    protocol="Both methods use the identical explicit tail-coordinate initial support. "
                             "Method order alternates; all failures and censored runs are retained. "
                             "Enumeration is an independent exhaustive search using the same fixed-support oracle.")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Output already exists or source/protocol changed; choose a fresh directory")
    else:
        save(manifest_path, manifest)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for source in sources:
            shutil.copy2(source, snapshot / source.name)
    cases = [(loss, d, n, seed, 2, "primary") for loss, d, n, seed in
             itertools.product(("overlap", "logistic"), (6, 8), (4, 8), (3, 7, 11))]
    if not args.small_only:
        cases.extend((loss, 12, 4, 3, 3, "larger") for loss in ("overlap", "logistic"))
    records = []
    with threadpool_limits(limits=1), patch.object(outer, "GateOracle", MeasuredOracle):
        for index, spec in enumerate(cases):
            problem, parameters, arrays = generated_case(*spec)
            path = raw / f"{parameters['id']}.json"
            if path.exists():
                if not args.resume:
                    raise SystemExit(f"Refusing to overwrite {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                methods = {"enumeration": run_method(problem, "enumeration", settings,
                                                     parameters["seed"], parameters["warm_support"])}
                order = ("master", "adaptive") if index % 2 == 0 else ("adaptive", "master")
                for method in order:
                    methods[method] = run_method(problem, method, settings,
                                                 parameters["seed"], parameters["warm_support"])
                record = dict(case=parameters, generated_data=arrays, method_order=list(order), methods=methods)
                save(path, record)
            records.append(record)
            statuses = {method: value["status"] for method, value in record["methods"].items()}
            print(json.dumps(dict(case=parameters["id"], statuses=statuses), ensure_ascii=False), flush=True)
    rows = list(itertools.chain.from_iterable(comparison(record, settings["tolerance"]) for record in records))
    summary = dict(manifest=manifest, groups={group: aggregate(rows, group) for group in ("primary", "larger")},
                   rows=rows, references_complete=sum(record["methods"]["enumeration"]["status"] == "optimal"
                                                      for record in records))
    save(output / "summary.json", summary)
    plot(rows, output)
    sections = ["# Matched validation of history-guided search", "",
                "This is a new algorithm validation, separate from the frozen historical study. "
                "It measures robust optimization accuracy and computational effort; no testing-power or "
                "variable-recovery conclusion is inferred from it.", "",
                "Independent exhaustive search supplies the reference interval. Methods share generated data, "
                "fixed radii, loss, initial support, numerical tolerance and wall-time limit. Both use the same "
                "current oracle, including its model reuse, so the comparison isolates the search policy.", ""]
    for group, data in summary["groups"].items():
        if not data["pairs"]:
            continue
        sections.extend([f"## {group.capitalize()} cases", "",
                         "| Method | Certified stops | Median full calls | Median seconds |", "|---|---:|---:|---:|"])
        for method in ("master", "adaptive"):
            m = data[method]
            sections.append(f"| {method} | {m['optimal']}/{m['cases']} | "
                            f"{m['oracle_calls']['median']:.1f} | {m['seconds']['median']:.3f} |")
        p = data["paired_counts"]
        sections.extend(["", f"Adaptive uses fewer/equal/more full calls in {p['fewer_calls']}/"
                         f"{p['equal_calls']}/{p['more_calls']} matched cases. It is faster in {p['faster']} "
                         f"and slower in {p['slower']}. The median paired call ratio is {p['median_call_ratio']:.3f}; "
                         f"the median runtime ratio is {p['median_runtime_ratio']:.3f}.", ""])
    sections.extend(["The extra repricing and cut-densification work is included in runtime. "
                     "A reduction in expensive oracle calls does not imply a runtime reduction. "
                     "This seeded validation is too small to establish universal performance superiority.", "",
                     "`summary.json` retains every status, reference consistency check, attained gap, "
                     "repricing/refinement count and model/workspace reuse count. Individual raw JSON files "
                     "include the generated data, full traces and returned test certificates. "
                     "The source snapshot and manifest identify the code used."])
    (output / "README.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    changed = [str(path.relative_to(ROOT)) for path in sources if
               hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))]]
    if changed:
        raise SystemExit(f"Source changed during validation; records retained, inspect snapshots: {changed}")
    print(json.dumps(dict(output=str(output), references_complete=summary["references_complete"]), indent=2))


if __name__ == "__main__":
    main()
