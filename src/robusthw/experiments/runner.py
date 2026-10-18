"""Freeze a study, then execute its shared-data comparisons without test tuning.

The orchestration is deliberately separate from the mathematical oracle. Every
method receives the same prepared samples, radii and fitted prediction rule.
Raw method failures remain records with the feasible constant-score fallback.
"""

import hashlib
import platform
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial.distance import pdist

from robusthw.core import FitResult, Problem, SolveOptions, constant_fit, fit_fixed
from robusthw.search import select

from .baselines import SelectionTimeout, rank_features, select_features
from .data import original_anchor_split, random_stream, synthetic_split
from .datasets import har_split, tabular_split
from .metrics import classification_metrics, recovery_metrics, subject_metrics
from .protocol import ExperimentSpec, build_plan
from .storage import array_hash, file_hash, fingerprint, fit_payload, read_json, write_json
from .verification import verify_small

METHODS = (
    "global",
    "forward",
    "forward_swap",
    "marginal_mmd",
    "l1_logistic",
    "mrmr",
    "joint_mmd",
    "random",
)
SEARCH_METHODS = ("global", "enumeration", "forward", "forward_swap")
PROJECT = Path(__file__).resolve().parents[3]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sources():
    if not (PROJECT / "src/robusthw").is_dir() or not (PROJECT / "configs").is_dir():
        raise RuntimeError("Freezing a research run requires the source checkout")
    paths = list((PROJECT / "src").rglob("*.py")) + list((PROJECT / "configs").glob("*.json"))
    paths += [
        PROJECT / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "note/main.tex",
            "note/main_zh.tex",
            "reports/experiment_report.tex",
            "reports/experiment_report_zh.tex",
        )
    ]
    return {
        path.relative_to(PROJECT).as_posix(): file_hash(path)
        for path in sorted(paths)
        if path.is_file()
    }


def freeze(suite, destination, seed=20261018, methods=METHODS):
    """Write an immutable setting list and exact source snapshot before execution."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a new protocol path; an existing freeze is immutable")
    methods = tuple(methods)
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"Methods must be selected from {METHODS}")
    specs = build_plan(suite, seed)
    sources = _sources()
    manifest = {
        "schema_version": 1,
        "created_at_utc": _now(),
        "suite": suite,
        "seed": seed,
        "methods": methods,
        "stage": "development" if suite == "development" else "confirmation",
        "source_hashes": sources,
        "specs": [spec.as_dict() for spec in specs],
        "data_settings": len(specs),
        "budget_tasks": sum(len(s.budgets) for s in specs),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "execution_state": "not_run",
    }
    manifest["manifest_hash"] = fingerprint(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot = destination.with_suffix(".sources.zip")
    with ZipFile(snapshot, "x", compression=ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(PROJECT / name, name)
    if _sources() != sources:
        raise RuntimeError("Sources changed while freezing; discard this snapshot and freeze again")
    write_json(destination, manifest)
    return manifest


def _verify_manifest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if fingerprint(payload) != manifest["manifest_hash"]:
        raise ValueError("Frozen protocol contents changed")
    for name, expected in manifest["source_hashes"].items():
        if not (PROJECT / name).is_file() or file_hash(PROJECT / name) != expected:
            raise ValueError(f"Source changed after freezing: {name}; freeze a new run")


def _verify_snapshot(path, manifest):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["source_hashes"]):
            raise ValueError("Frozen source snapshot members changed")
        for name, expected in manifest["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Frozen source snapshot changed: {name}")


def _verify_record(record):
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    if record.get("record_hash") != fingerprint(payload):
        raise ValueError("Saved record contents changed")


def _verify_data_file(root, record):
    expected = record.get("data", {}).get("npz_sha256")
    if expected:
        path = Path(root) / "data" / (record["spec"]["independent_data_id"] + ".npz")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Saved data file changed")
        with np.load(path, allow_pickle=False) as stored:
            for name, digest in record["data"]["array_hashes"].items():
                if name not in stored or array_hash(stored[name]) != digest:
                    raise ValueError("Saved sample matrix changed")


def _prepare(spec, data_root):
    if spec.kind == "anchor":
        return original_anchor_split(spec.seed)
    if spec.kind == "tabular":
        return tabular_split(spec.model, data_root, spec.n, spec.seed)
    if spec.kind == "har":
        return har_split(data_root, spec.seed)
    return synthetic_split(
        spec.model,
        spec.D,
        spec.n,
        spec.s,
        spec.seed,
        validation_per_class=spec.metadata.get("validation_per_class", 20),
        test_per_class=spec.metadata.get("test_per_class", 10000),
        target_error=spec.metadata.get("target_error", 0.2),
    )


def _options(spec):
    if spec.kind == "small":
        return SolveOptions(
            absolute_tolerance=spec.metadata.get("outer_tolerance", 1e-6),
            inner_tolerance=spec.metadata.get("validation_tolerance", 1e-8),
            constraint_tolerance=1e-9,
        )
    return SolveOptions(
        time_limit=spec.metadata.get("time_limit", 600 if spec.kind == "har" else 300)
    )


def _scales(data):
    scales = np.array(
        [np.median(pdist(x, metric="cityblock")) / data.dimension for x in data.train]
    )
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("The declared within-class median scale is zero; no radius was invented")
    return scales


def _restore_fit(data, radii, payload):
    return FitResult(
        Problem(*data.train, radii),
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


def _validation(fit, data):
    return classification_metrics(fit.predict_score(data.val0), fit.predict_score(data.val1))[
        "ber_prob"
    ]


def _radius_selection(data, spec, options):
    started = perf_counter()
    scales = _scales(data)
    if spec.kind in ("small", "scaling"):
        factor = float(spec.metadata.get("radius_factor", 0.25))
        return scales * factor, {
            "rule": "fixed_scale_quarter",
            "scales": scales,
            "factor": factor,
            "runtime": perf_counter() - started,
            "candidates": [],
        }
    reference_budget = min(16 if spec.kind == "har" else 4, data.dimension)
    references = {}
    reference_errors = {}
    for method in ("marginal_mmd", "mrmr", "joint_mmd"):
        try:
            references[method] = select_features(
                method, *data.train, reference_budget, seed=spec.seed, time_limit=options.time_limit
            )
        except Exception:
            references[method] = tuple(range(reference_budget))
            reference_errors[method] = traceback.format_exc()
    candidates = []
    for factor in (1 / 8, 1 / 4, 1 / 2, 1, 2):
        problem = Problem(*data.train, scales * factor)
        checks = []
        for method, support in references.items():
            failure = reference_errors.get(method)
            try:
                fitted = (
                    constant_fit(problem, support, "reference_selection_failure")
                    if failure
                    else fit_fixed(problem, support, options, with_cut=False)
                )
            except Exception:
                failure = traceback.format_exc()
                fitted = constant_fit(problem, support, "reference_fit_failure")
            checks.append(
                {
                    "method": method,
                    "validation_error": _validation(fitted, data),
                    "fit": fit_payload(fitted),
                    "error": failure,
                }
            )
        candidates.append(
            {
                "factor": factor,
                "validation_error": float(np.mean([item["validation_error"] for item in checks])),
                "references": checks,
            }
        )
    best = min(candidates, key=lambda row: (row["validation_error"], -row["factor"]))
    return scales * best["factor"], {
        "rule": "shared_validation",
        "scales": scales,
        "factor": best["factor"],
        "reference_budget": reference_budget,
        "candidates": candidates,
        "runtime": perf_counter() - started,
    }


def _fit_method(data, radii, budget, method, options, seed, *, C=1.0, random_initial=False):
    """Fit a method without touching any test observations."""
    problem = Problem(*data.train, radii)
    started = perf_counter()
    selection_status = "complete"
    initialization_seconds = 0.0
    if not 0 <= budget <= data.dimension:
        raise ValueError("Budget exceeds retained feature count")
    if method == "all_variables":
        fitted = fit_fixed(problem, tuple(range(data.dimension)), options, with_cut=False)
        record = {
            "method": method,
            "budget": budget,
            "status": fitted.status,
            "upper": fitted.upper,
            "lower": None,
            "gap": None,
            "fit_count": 1,
            "master_count": 0,
            "master_seconds": 0.0,
            "oracle_seconds": fitted.runtime,
            "trace": [],
            "cuts": [],
            "diagnostics": {},
        }
        result = None
    elif method in SEARCH_METHODS:
        ranking = rank_features(*data.train)
        initial = (
            tuple(
                sorted(
                    random_stream(seed, "random_initialization").choice(
                        data.dimension, budget, replace=False
                    )
                )
            )
            if random_initial
            else tuple(ranking[:budget])
        )
        initialization_seconds = perf_counter() - started
        remaining = max(0.0, options.time_limit - initialization_seconds)
        result = select(
            problem,
            budget,
            method,
            replace(options, time_limit=remaining),
            initial_support=initial if method == "global" else None,
            ranked_features=ranking,
        )
        record = {
            "method": method,
            "budget": budget,
            "status": result.status,
            "upper": result.upper,
            "lower": result.lower,
            "gap": result.gap,
            "fit_count": result.fit_count,
            "master_count": result.master_count,
            "master_seconds": result.master_seconds,
            "oracle_seconds": result.oracle_seconds,
            "trace": [
                {**event, "seconds": event["seconds"] + initialization_seconds}
                for event in result.trace
            ],
            "diagnostics": result.diagnostics,
            "cuts": [{"intercept": cut.intercept, "slope": cut.slope} for cut in result.cuts],
        }
        fitted = result.fit
    else:
        try:
            support = select_features(
                method,
                *data.train,
                budget,
                seed=seed,
                C=C,
                swap_limit=options.swap_limit,
                min_improvement=options.swap_improvement,
                time_limit=options.time_limit,
            )
        except SelectionTimeout as error:
            partial = list(error.partial_support)
            ranking = rank_features(*data.train)
            support = tuple(sorted((partial + [j for j in ranking if j not in partial])[:budget]))
            selection_status = "incomplete"
        remaining = max(0.0, options.time_limit - (perf_counter() - started))
        fitted = fit_fixed(problem, support, replace(options, time_limit=remaining), with_cut=False)
        record = {
            "method": method,
            "budget": budget,
            "status": selection_status
            if selection_status != "complete"
            else ("complete" if fitted.status == "optimal" else fitted.status),
            "upper": fitted.upper,
            "lower": None,
            "gap": None,
            "fit_count": int(remaining > 0),
            "master_count": 0,
            "master_seconds": 0.0,
            "oracle_seconds": fitted.runtime,
            "trace": [],
            "diagnostics": {},
            "cuts": [],
        }
        result = None
    record.update(
        runtime=perf_counter() - started,
        initialization_seconds=initialization_seconds,
        support=fitted.support,
        feature_ids=[data.feature_ids[j] for j in fitted.support],
        fit=fit_payload(fitted),
        radius_factor=1.0,
        initialization="random" if random_initial else "marginal_mmd",
        timing_repeat=0,
        role="comparison",
    )
    return record, fitted, result


def _fit_l1(data, radii, budget, options, seed):
    started = perf_counter()
    candidates = []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        candidate_started = perf_counter()
        try:
            record, fit, _ = _fit_method(data, radii, budget, "l1_logistic", options, seed, C=C)
        except Exception:
            fit = constant_fit(Problem(*data.train, radii), tuple(range(budget)), "solver_error")
            record = {
                "method": "l1_logistic",
                "budget": budget,
                "status": "solver_error",
                "runtime": perf_counter() - candidate_started,
                "upper": 1.0,
                "lower": None,
                "gap": None,
                "fit_count": 0,
                "oracle_seconds": 0.0,
                "master_count": 0,
                "master_seconds": 0.0,
                "trace": [],
                "cuts": [],
                "support": fit.support,
                "feature_ids": [data.feature_ids[j] for j in fit.support],
                "fit": fit_payload(fit),
                "error": traceback.format_exc(),
                "role": "comparison",
            }
        candidates.append((C, _validation(fit, data), record, fit))
    C, error, selected, fit = min(candidates, key=lambda row: (row[1], row[0]))
    record = dict(selected)
    record["runtime"] = perf_counter() - started
    record["validation"] = {
        "selected_C": C,
        "validation_error": error,
        "candidates": [
            {
                "C": row[0],
                "validation_error": row[1],
                "status": row[2]["status"],
                "runtime": row[2]["runtime"],
                "fit": fit_payload(row[3]),
                "error": row[2].get("error"),
            }
            for row in candidates
        ],
    }
    record["fit_count"] = sum(row[2]["fit_count"] for row in candidates)
    record["oracle_seconds"] = sum(row[2]["oracle_seconds"] for row in candidates)
    return record, fit, None


def _evaluate(fit, data, budget, recovery=True):
    scores0, scores1 = fit.predict_score(data.test0), fit.predict_score(data.test1)
    metrics = classification_metrics(scores0, scores1)
    if recovery and data.signal_indices and budget > 0:
        metrics.update(recovery_metrics(fit.support, data.signal_indices, budget))
    if data.subjects0 is not None:
        metrics["by_subject"] = subject_metrics(scores0, scores1, data.subjects0, data.subjects1)
        for key in (
            "ber_prob",
            "ber_hard",
            "error0_prob",
            "error1_prob",
            "error0_hard",
            "error1_hard",
        ):
            metrics[key] = float(np.mean([row[key] for row in metrics["by_subject"].values()]))
        metrics["half_hinge_risk"] = 2 * metrics["ber_prob"]
    return metrics


def _persist_data(data, path):
    fields = ("train0", "train1", "val0", "val1", "test0", "test1")
    arrays = {name: getattr(data, name) for name in fields}
    for name in ("subjects0", "subjects1"):
        if getattr(data, name) is not None:
            arrays[name] = getattr(data, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        **data.as_dict(),
        "array_hashes": {name: array_hash(value) for name, value in arrays.items()},
        "npz_sha256": file_hash(path),
    }


def _single_record(
    spec,
    data,
    radii,
    method,
    budget,
    options,
    *,
    factor=1.0,
    random_initial=False,
    timing_repeat=0,
    recovery=True,
):
    started = perf_counter()
    try:
        if method == "l1_logistic":
            record, fitted, search_result = _fit_l1(
                data, radii * factor, budget, options, spec.seed
            )
        else:
            record, fitted, search_result = _fit_method(
                data,
                radii * factor,
                budget,
                method,
                options,
                spec.seed,
                random_initial=random_initial,
            )
    except Exception:
        problem = Problem(*data.train, radii * factor)
        fitted = constant_fit(problem, tuple(range(min(budget, data.dimension))), "solver_error")
        record = {
            "method": method,
            "budget": budget,
            "status": "solver_error",
            "runtime": perf_counter() - started,
            "upper": 1.0,
            "lower": 0.0 if method in ("global", "enumeration") else None,
            "gap": 1.0 if method in ("global", "enumeration") else None,
            "fit_count": 0,
            "master_count": 0,
            "oracle_seconds": 0.0,
            "master_seconds": 0.0,
            "support": fitted.support,
            "feature_ids": [data.feature_ids[j] for j in fitted.support],
            "trace": [],
            "cuts": [],
            "error": traceback.format_exc(),
            "fit": fit_payload(fitted),
            "role": "comparison",
        }
        search_result = None
    record.update(
        radius_factor=factor,
        initialization="random" if random_initial else "marginal_mmd",
        timing_repeat=timing_repeat,
        target_tolerance=options.absolute_tolerance,
    )
    record["metrics"] = _evaluate(fitted, data, budget, recovery=recovery and spec.kind != "anchor")
    return record, search_result


def _failed_instance(spec, methods, error, data_root, runtime):
    """Keep the full planned denominator and explicit zero-score predictions."""
    data = None
    try:
        data = _prepare(spec, data_root)
    except Exception:
        pass
    compared = (
        SEARCH_METHODS
        if spec.kind == "small"
        else (("global", "forward", "forward_swap") if spec.kind == "scaling" else methods)
    )
    settings = [(method, budget, 1.0, False) for budget in spec.budgets for method in compared]
    if spec.kind == "anchor":
        settings = [("all_variables", spec.D, 1.0, False)]
    elif spec.kind == "ablation":
        settings = [
            (method, 4, factor, False)
            for factor in spec.metadata["radius_factors"]
            for method in spec.metadata["radius_methods"]
        ] + [("global", 4, 1.0, True)]
    results = []
    for method, budget, factor, random_initial in settings:
        dimension = data.dimension if data is not None else spec.D
        support = tuple(range(min(budget, dimension)))
        metrics = classification_metrics(np.zeros(1), np.zeros(1))
        if (
            data is not None
            and data.signal_indices
            and len(support) == budget
            and spec.kind != "anchor"
        ):
            metrics.update(recovery_metrics(support, data.signal_indices, budget))
        if data is not None and data.subjects0 is not None:
            metrics["by_subject"] = subject_metrics(
                np.zeros(len(data.test0)), np.zeros(len(data.test1)), data.subjects0, data.subjects1
            )
        result = {
            "method": method,
            "budget": budget,
            "role": "comparison" if method != "all_variables" else "reference",
            "status": "instance_failure",
            "metrics": metrics,
            "support": support,
            "feature_ids": [data.feature_ids[j] for j in support] if data else support,
            "upper": 1.0,
            "lower": 0.0 if method in ("global", "enumeration") else None,
            "gap": 1.0 if method in ("global", "enumeration") else None,
            "runtime": runtime,
            "fit_count": 0,
            "master_count": 0,
            "master_seconds": 0.0,
            "oracle_seconds": 0.0,
            "trace": [],
            "fit": None,
            "cuts": [],
            "radius_factor": factor,
            "initialization": "random" if random_initial else "marginal_mmd",
            "timing_repeat": 0,
            "fallback": "constant_zero_score",
            "error": error,
        }
        if spec.kind == "shift":
            result["shifts"] = [
                {
                    "kind": kind,
                    "delta_factor": factor,
                    "delta": None,
                    "metrics": dict(metrics),
                    "fallback": "constant_zero_score",
                }
                for kind in spec.metadata["shift_types"]
                for factor in spec.metadata["delta_factors"]
            ]
        results.append(result)
    return {
        "schema_version": 1,
        "spec": spec.as_dict(),
        "status": "failed",
        "results": results,
        "data": data.as_dict() if data is not None else {},
        "radii": None,
        "radius_selection": {"status": "failed"},
        "runtime": runtime,
        "error": error,
    }


def _execute(spec, root, data_root, methods):
    started = perf_counter()
    data = _prepare(spec, data_root)
    options = _options(spec)
    data_path = root / "data" / (spec.independent_data_id + ".npz")
    metadata = _persist_data(data, data_path)
    radii, selection = _radius_selection(data, spec, options)
    results, search_results = [], []
    if spec.kind == "anchor":
        record, _ = _single_record(
            spec, data, radii, "all_variables", data.dimension, options, recovery=False
        )
        record["role"] = "reference"
        results.append(record)
    elif spec.kind == "ablation":
        for factor in spec.metadata["radius_factors"]:
            for method in spec.metadata["radius_methods"]:
                record, _ = _single_record(spec, data, radii, method, 4, options, factor=factor)
                results.append(record)
        record, _ = _single_record(spec, data, radii, "global", 4, options, random_initial=True)
        results.append(record)
    else:
        compared = (
            SEARCH_METHODS
            if spec.kind == "small"
            else (("global", "forward", "forward_swap") if spec.kind == "scaling" else methods)
        )
        for budget in spec.budgets:
            for timing_repeat in range(spec.metadata.get("timing_repeats", 1)):
                for method in compared:
                    record, search_result = _single_record(
                        spec, data, radii, method, budget, options, timing_repeat=timing_repeat
                    )
                    results.append(record)
                    if search_result is not None:
                        search_results.append(search_result)
        if spec.kind not in ("small", "scaling"):
            supports = [("all_variables", tuple(range(data.dimension)))]
            if data.signal_indices:
                supports.insert(0, ("true_signals", data.signal_indices))
            for method, support in supports:
                reference_started = perf_counter()
                error = None
                try:
                    fitted = fit_fixed(
                        Problem(*data.train, radii), support, options, with_cut=False
                    )
                except Exception:
                    fitted = constant_fit(Problem(*data.train, radii), support, "solver_error")
                    error = traceback.format_exc()
                results.append(
                    {
                        "method": method,
                        "budget": len(support),
                        "support": support,
                        "feature_ids": [data.feature_ids[j] for j in support],
                        "role": "reference",
                        "status": fitted.status,
                        "upper": fitted.upper,
                        "lower": None,
                        "gap": None,
                        "runtime": perf_counter() - reference_started,
                        "fit_count": 1,
                        "master_count": 0,
                        "oracle_seconds": fitted.runtime,
                        "master_seconds": 0.0,
                        "trace": [],
                        "radius_factor": 1.0,
                        "initialization": "known_reference",
                        "timing_repeat": 0,
                        "metrics": _evaluate(fitted, data, len(support)),
                        "fit": fit_payload(fitted),
                        "error": error,
                    }
                )
    verification = None
    execution_status = "completed"
    if spec.kind == "small":
        try:
            verification = verify_small(
                Problem(*data.train, radii),
                spec.budgets[0],
                search_results,
                spec.metadata["validation_tolerance"],
            )
            for record in results:
                record["excess_risk"] = record["upper"] - verification["independent_optimum"]
            if any(row["status"] != "optimal" for row in results if row["method"] == "enumeration"):
                raise RuntimeError(
                    "Complete exhaustive reference did not attain requested precision"
                )
        except Exception:
            verification = {"status": "failed", "error": traceback.format_exc()}
            execution_status = "failed"
            for record in results:
                record["uncertified_status"] = record["status"]
                record["status"] = "verification_failure"
    for row in results:
        row["radius_selection_seconds"] = selection["runtime"]
        row["total_runtime"] = row["runtime"] + selection["runtime"]
    return {
        "schema_version": 1,
        "spec": spec.as_dict(),
        "status": execution_status,
        "data": metadata,
        "radii": radii,
        "radius_selection": selection,
        "results": results,
        "verification": verification,
        "runtime": perf_counter() - started,
    }


def _shift(spec, source_root, methods=METHODS):
    source_root = Path(source_root)
    source = read_json(source_root / "records" / (spec.metadata["source_data_id"] + ".json"))
    _verify_record(source)
    for name in ("independent_data_id", "model", "D", "n", "s", "seed"):
        if source["spec"][name] != spec.as_dict()[name]:
            raise ValueError(f"E2 source setting mismatch: {name}")
    if source["status"] != "completed":
        raise ValueError("E2 requires the completed matching E1 record; it never refits")
    _verify_data_file(source_root, source)
    data = _prepare(spec, None)
    if any(
        array_hash(getattr(data, name)) != expected
        for name, expected in source["data"]["array_hashes"].items()
    ):
        raise ValueError("E2 samples do not match the stored E1 generation")
    radii = np.asarray(source["radii"])
    signals, results = list(data.signal_indices), []
    for original in source["results"]:
        if (
            original["budget"] != 4
            or original["role"] != "comparison"
            or original["method"] not in methods
        ):
            continue
        fitted = _restore_fit(data, radii, original["fit"])
        record = dict(original)
        record["shifts"] = []
        for kind in spec.metadata["shift_types"]:
            noise = [
                random_stream(spec.seed, f"E2_{kind}_{k}").standard_normal(x.shape)
                for k, x in enumerate(data.test)
            ]
            for factor in spec.metadata["delta_factors"]:
                delta = factor * float(np.mean(radii))
                changed = []
                for k, x in enumerate(data.test):
                    y = x.copy()
                    y[:, signals] += (
                        delta / 4
                        if kind == "translation"
                        else (delta / (4 * np.sqrt(2 / np.pi)) * noise[k][:, signals])
                    )
                    changed.append(y)
                record["shifts"].append(
                    {
                        "kind": kind,
                        "delta_factor": factor,
                        "delta": delta,
                        "metrics": _evaluate(fitted, data.with_test(*changed), 4),
                        "test_hashes": [array_hash(x) for x in changed],
                    }
                )
        results.append(record)
    if set(row["method"] for row in results) != set(methods):
        raise ValueError("E2 is missing requested fitted E1 methods")
    return {
        "schema_version": 1,
        "spec": spec.as_dict(),
        "status": "completed",
        "data": source["data"],
        "source_manifest_hash": source["manifest_hash"],
        "source_record_sha256": file_hash(
            source_root / "records" / (spec.metadata["source_data_id"] + ".json")
        ),
        "radii": radii,
        "radius_selection": source["radius_selection"],
        "results": results,
        "refitted": False,
    }


def run(manifest_path, data_root="data/uci", *, reuse_root=None, limit=None):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    _verify_manifest(manifest)
    _verify_snapshot(manifest_path.with_suffix(".sources.zip"), manifest)
    root = manifest_path.parent
    specs = manifest["specs"] if limit is None else manifest["specs"][:limit]
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    records = []
    for raw in specs:
        raw = {**raw, "budgets": tuple(raw["budgets"])}
        spec = ExperimentSpec(**raw)
        destination = root / "records" / (spec.independent_data_id + ".json")
        if destination.exists():
            record = read_json(destination)
            _verify_record(record)
            if spec.kind != "shift":
                _verify_data_file(root, record)
            if record.get("manifest_hash") != manifest["manifest_hash"] or fingerprint(
                record["spec"]
            ) != fingerprint(raw):
                raise ValueError(f"Conflicting resumed record: {destination}")
        else:
            started = perf_counter()
            try:
                if spec.kind == "shift":
                    if reuse_root is None:
                        raise ValueError(
                            "E2 requires --reuse-root pointing to its matching E1 output"
                        )
                    record = _shift(spec, reuse_root, tuple(manifest["methods"]))
                else:
                    record = _execute(spec, root, data_root, tuple(manifest["methods"]))
            except Exception:
                record = _failed_instance(
                    spec,
                    tuple(manifest["methods"]),
                    traceback.format_exc(),
                    data_root,
                    perf_counter() - started,
                )
            record.update(manifest_hash=manifest["manifest_hash"], saved_at_utc=_now())
            record["record_hash"] = fingerprint(record)
            write_json(destination, record)
        records.append(
            {
                "data_id": spec.independent_data_id,
                "status": record["status"],
                "sha256": file_hash(destination),
            }
        )
    summary = {
        "manifest_hash": manifest["manifest_hash"],
        "planned": len(manifest["specs"]),
        "attempted": len(records),
        "completed": sum(row["status"] == "completed" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "partial_execution": len(records) < len(manifest["specs"]),
        "records": records,
    }
    write_json(root / "execution.json", summary)
    return summary
