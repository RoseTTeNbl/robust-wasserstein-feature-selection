"""Run a frozen JSON protocol, retaining one resumable record per method/seed."""

import argparse
import json
import traceback
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ..problem import Problem, json_value
from .baselines import screen_mmd, select
from .data import real_trial, synthetic_trial
from .metrics import (
    permutation_mmd,
    probability_metrics,
    rbf_bandwidth_squared,
    rbf_gram,
    score_metrics,
    support_metrics,
    translated_pair,
    translation_direction,
)


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(record), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _problem(pair, budget, theta, loss="overlap"):
    radii = [theta, theta] if np.isscalar(theta) else list(theta)
    return Problem(*pair, budget=budget, theta0=radii[0], theta1=radii[1], loss=loss)


def _fixed_head(problem, support, *, time_limit=30.0):
    from ..oracle import GateOracle

    return GateOracle(problem).fixed_support(problem.mask(support), time_limit=time_limit)


def robust_greedy(problem, *, time_limit=30.0):
    """Fully refit every candidate; a time limit yields a labeled fallback."""
    from ..oracle import GateOracle

    started = perf_counter()
    oracle = GateOracle(problem)
    selected = ()
    calls = 0
    history = []
    initial = screen_mmd(problem.x0, problem.x1, problem.budget)
    fallback = (*initial, *[j for j in range(problem.p) if j not in initial])
    status = "greedy_complete"
    for stage in range(problem.budget):
        best = None
        for j in range(problem.p):
            if j in selected:
                continue
            remaining = time_limit - (perf_counter() - started)
            if remaining <= 0:
                status = "greedy_time_limit"
                break
            support = tuple(sorted((*selected, j)))
            calls += 1
            try:
                test = oracle.fixed_support(problem.mask(support), time_limit=remaining)
            except TimeoutError:
                status = "greedy_time_limit"
                break
            value = float(test.primal_objective)
            if best is None or value < best[0] - 1e-12:
                best = value, support
        if status != "greedy_complete":
            # Preserve the last complete stage, then fill deterministically.
            selected = tuple(sorted((*selected, *[j for j in fallback if j not in selected]
                                     [:problem.budget - len(selected)])))
            break
        selected = best[1]
        history.append({"stage": stage + 1, "support": selected, "upper": best[0]})
    return {"support": selected, "status": status, "seconds": perf_counter() - started,
            "oracle_calls": calls, "weights": None, "trace": history, "diagnostics": {}}


def _global_selection(problem, config, method="risk"):
    from ..solver import solve

    initial = screen_mmd(problem.x0, problem.x1, problem.budget)
    gate = method.split("_", 1)[0]
    rounds = 0 if method.endswith("_no_root") else int(config.get("root_rounds", 5))
    result = solve(problem, gate=gate, time_limit=float(config.get("time_limit", 30)),
                   root_rounds=rounds, core_point=bool(config.get("core_point", False)),
                   warm_support=initial, cut_mode="ratio" if method.endswith("_ratio") else "tangent")
    return {"support": tuple(result.support), "status": result.status,
            "seconds": float(result.seconds), "oracle_calls": int(result.oracle_calls),
            "visited": int(result.visited), "lower": float(result.lower),
            "upper": float(result.upper), "gap": float(result.upper - result.lower),
            "weights": None, "trace": result.trace,
            "diagnostics": getattr(result, "diagnostics", {})}


def _select(trial, method, config):
    problem = _problem(trial.selection, int(config.get("budget", 3)), config.get("theta", 0.25))
    if method in ("risk", "risk_no_root", "box", "triangle"):
        return _global_selection(problem, config, method)
    if method == "greedy":
        return robust_greedy(problem, time_limit=float(config.get("time_limit", 30)))
    if method == "signal":
        if trial.true_support is None:
            raise ValueError("Signal-support diagnostic is undefined for real data")
        return {"support": trial.true_support, "seconds": 0.0, "status": "oracle_signal_diagnostic",
                "weights": None, "diagnostics": {"not_deployable": True}}
    if method == "all":
        return {"support": tuple(range(trial.dimension)), "seconds": 0.0,
                "status": "all_features_diagnostic", "weights": None,
                "diagnostics": {"not_budget_matched": True}}
    return select(method, *trial.selection, problem.budget, trial.seed,
                  wang_regularization=float(config.get("wang_regularization", 0.1)))


def _test_probability(test, x):
    return test.prob_class0(x)


def evaluate(trial, selection, config):
    """The same independent robust/SVM heads and MMD test for every support."""
    support = tuple(selection["support"])
    started = perf_counter()
    budget = len(support)
    head_problem = _problem(trial.head, budget, config.get("theta", 0.25))
    head = _fixed_head(head_problem, support, time_limit=float(config.get("head_time_limit", 30)))
    robust = probability_metrics(*[_test_probability(head, x) for x in trial.testing])
    metrics = {**support_metrics(support, trial.true_support),
               **{f"robust_{key}": value for key, value in robust.items()},
               "head_robust_upper": float(head.primal_objective)}
    points = np.vstack(trial.head)[:, list(support)]
    labels = np.r_[np.zeros(len(trial.head[0])), np.ones(len(trial.head[1]))]
    if support:
        head_width = rbf_bandwidth_squared(points)
        svm = SVC(C=1, gamma=1 / (2 * head_width)).fit(points, labels)
        svm_predictions = [svm.decision_function(x[:, list(support)]) for x in trial.testing]
    else:
        head_width, svm = 1.0, None
        svm_predictions = [np.zeros(len(x)) for x in trial.testing]
    metrics.update({f"svm_{key}": value for key, value in score_metrics(*svm_predictions).items()})
    direction = translation_direction(*trial.head)
    stress = []
    for rho in config.get("stress", [0.0, 0.25, 0.50]):
        shifted = translated_pair(trial.testing, direction, float(rho))
        values = probability_metrics(*[_test_probability(head, x) for x in shifted])
        stress.append({"rho": float(rho), **values})
    metrics["robust_worst_stress_balanced_error"] = max(x["balanced_error"] for x in stress)
    metrics["robust_worst_stress_randomized_balanced_error"] = max(
        x["randomized_balanced_error"] for x in stress)
    # This stress uses empirical source atoms: coupling inclusion is explicit.
    theta = config.get("theta", 0.25)
    rho_certificate = min([theta, theta] if np.isscalar(theta) else theta)
    shifted_atoms = translated_pair(trial.head, direction, float(rho_certificate))
    atom_values = probability_metrics(*[_test_probability(head, x) for x in shifted_atoms])
    metrics["empirical_shift_risk"] = atom_values["half_hinge_surrogate_sum"]
    metrics["empirical_certificate_violation"] = max(
        0.0, atom_values["half_hinge_surrogate_sum"] - head.primal_objective)
    tests = {"bandwidth_squared": rbf_bandwidth_squared(
        np.vstack(trial.selection)[:, list(support)]), "alternative": [], "null": []}
    if trial.population is not None:
        repeats = int(config.get("test_replicates", 20))
        n_testing = int(config.get("n_testing", 40))
        permutations = int(config.get("permutations", 199))
        for null, key in ((False, "alternative"), (True, "null")):
            for replicate in range(repeats):
                pair = trial.sample_testing_pair(replicate, n_testing, null=null)
                pooled = np.vstack(pair)
                permutation_seed = np.random.SeedSequence([trial.seed, 91003, replicate, int(null)])
                gram = rbf_gram(pooled[:, list(support)], tests["bandwidth_squared"])
                result = permutation_mmd(gram, len(pair[0]), permutations=permutations,
                                         seed=permutation_seed)
                result.update({"replicate": replicate,
                               "seed_entropy": permutation_seed.entropy})
                if selection.get("weights") is not None:
                    weights = np.asarray(selection["weights"])
                    widths = np.asarray(selection["diagnostics"]["coordinate_bandwidth_squared"])
                    distance = (pooled[:, None, :] - pooled[None, :, :]) ** 2
                    native_gram = np.einsum("ijr,r->ij", np.exp(-distance / (2 * widths)), weights)
                    result["native_additive"] = permutation_mmd(
                        native_gram, len(pair[0]), permutations=permutations, seed=permutation_seed)
                tests[key].append(result)
        metrics["power"] = float(np.mean([x["reject"] for x in tests["alternative"]]))
        metrics["null_rejection"] = float(np.mean([x["reject"] for x in tests["null"]]))
        if selection.get("weights") is not None:
            metrics["native_additive_power"] = float(np.mean(
                [x["native_additive"]["reject"] for x in tests["alternative"]]))
            metrics["native_additive_null_rejection"] = float(np.mean(
                [x["native_additive"]["reject"] for x in tests["null"]]))
    return {"metrics": metrics, "stress": stress, "tests": tests,
            "head": head.as_dict(), "head_svm_bandwidth_squared": head_width,
            "stress_direction": direction.tolist(), "evaluation_seconds": perf_counter() - started}


def _trial_jobs(config, project_root):
    seeds = config.get("seeds", list(range(30)))
    for mechanism, seed in product(config.get("mechanisms", []), seeds):
        settings = {key: int(config[key]) for key in
                    ("dimension", "signal_size", "n_select", "n_head", "n_classification")
                    if key in config}
        yield synthetic_trial(mechanism, int(seed), **settings)
    for source, seed in product(config.get("real_paths", []), config.get("real_seeds", seeds)):
        path = Path(source)
        if not path.is_absolute():
            path = project_root / path
        yield real_trial(path, int(seed), n_select=int(config.get("n_select", 20)),
                         n_head=int(config.get("n_head", 20)))


def run_statistical(config, output, project_root):
    output = Path(output)
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    methods = config.get("methods", ["risk", "greedy", "mmd", "wang", "l1", "random"])
    if config.get("diagnostic_references", False):
        methods = [*methods, "signal", "all"]
    completed, skipped, failures = 0, 0, 0
    for trial in _trial_jobs(config, project_root):
        key = f"{trial.name}_D{trial.dimension}_seed{trial.seed:03d}"
        _write_json(output / "trials" / f"{key}_data.json", trial.as_dict())
        for method in methods:
            if method == "signal" and trial.true_support is None:
                continue
            path = output / "records" / f"{key}_{method}.json"
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("config_sha256") != digest:
                    raise ValueError(f"Resume configuration does not match {path}")
                skipped += 1
                continue
            record = {**trial.as_dict(), "method": method, "config_sha256": digest,
                      "budget": int(config.get("budget", 3)), "theta": config.get("theta", 0.25),
                      "loss": "overlap", "run_status": "complete"}
            started = perf_counter()
            try:
                selection = _select(trial, method, config)
                trace = selection.pop("trace", [])
                record["selection"] = selection
                record["selection_timed_out"] = "time_limit" in selection["status"]
                record["selection_failed"] = selection["status"] in (
                    "master_failure", "oracle_failure", "numerical_failure", "numerical_inconsistency")
                record["semantic_support"] = trial.semantic_support(selection["support"])
                record.update(evaluate(trial, selection, config))
                if trace:
                    trace_path = output / "traces" / f"{key}_{method}.json"
                    _write_json(trace_path, trace)
                    record["trace_file"] = str(trace_path.relative_to(output))
            except Exception as error:
                failures += 1
                record.update({"run_status": "failure", "error_type": type(error).__name__,
                               "error": str(error), "traceback": traceback.format_exc()})
            record["total_seconds"] = perf_counter() - started
            _write_json(path, record)
            completed += 1
            print(f"{key} {method}: {record['run_status']} ({record['total_seconds']:.2f}s)", flush=True)
    return {"completed": completed, "skipped": skipped, "failures": failures}


def run_computational(config, output, project_root):
    """Common instances and warm starts for gate/root-cut comparisons."""
    from ..solver import enumerate_supports

    output = Path(output)
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    completed, skipped, failures = 0, 0, 0
    for setting in config["settings"]:
        current = {**config, **setting}
        current.pop("settings", None)
        if "mechanism" in current:
            current["mechanisms"] = [current["mechanism"]]
        if "gates" in current:
            current["methods"] = list(current["gates"])
        if current.get("enumeration", False) and int(current["dimension"]) <= 12:
            current["methods"] = [*current.get("methods", []), "enumeration"]
        for trial in _trial_jobs({**current, "real_paths": []}, project_root):
            key = (f"{trial.name}_D{trial.dimension}_d{current['budget']}"
                   f"_n{len(trial.selection[0])}_seed{trial.seed:03d}")
            problem = _problem(trial.selection, int(current["budget"]), current.get("theta", 0.25))
            methods = current.get("methods", ["box", "triangle", "risk_no_root", "risk"])
            for method in methods:
                path = output / "records" / f"{key}_{method}.json"
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("config_sha256") != digest:
                        raise ValueError(f"Resume configuration does not match {path}")
                    skipped += 1
                    continue
                record = {**trial.as_dict(), "method": method, "config_sha256": digest,
                          "budget": current["budget"], "theta": current.get("theta", 0.25),
                          "loss": "overlap", "run_status": "complete"}
                started = perf_counter()
                try:
                    if method == "enumeration":
                        solution = enumerate_supports(problem, time_limit=float(current.get("time_limit", 30)))
                        result = {"support": solution.support, "lower": solution.lower,
                                  "upper": solution.upper, "status": solution.status,
                                  "seconds": solution.seconds, "oracle_calls": solution.oracle_calls,
                                  "visited": solution.visited, "gap": solution.upper - solution.lower,
                                  "trace": solution.trace}
                    else:
                        result = _global_selection(problem, current, method)
                    trace = result.pop("trace", [])
                    trace_path = output / "traces" / f"{key}_{method}.json"
                    _write_json(trace_path, trace)
                    record.update({"selection": result, "trace_file": str(trace_path.relative_to(output)),
                                   "semantic_support": trial.semantic_support(result["support"]),
                                   "selection_timed_out": "time_limit" in result["status"],
                                   "selection_failed": result["status"] in (
                                       "master_failure", "oracle_failure", "numerical_failure",
                                       "numerical_inconsistency")})
                except Exception as error:
                    failures += 1
                    record.update({"run_status": "failure", "error_type": type(error).__name__,
                                   "error": str(error), "traceback": traceback.format_exc()})
                record["total_seconds"] = perf_counter() - started
                _write_json(path, record)
                completed += 1
                print(f"{key} {method}: {record['run_status']} ({record['total_seconds']:.2f}s)", flush=True)
    return {"completed": completed, "skipped": skipped, "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--only-seeds", type=int, nargs="+", help="Pilot subset; use a separate output folder")
    parser.add_argument("--limit", type=int, help="Pilot: keep only this many seeds per setting")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.only_seeds is not None or args.limit is not None:
        def pilot_seeds(values):
            result = [int(seed) for seed in values if args.only_seeds is None or seed in args.only_seeds]
            return result if args.limit is None else result[:args.limit]

        config["seeds"] = pilot_seeds(config.get("seeds", list(range(30))))
        if "real_seeds" in config:
            config["real_seeds"] = pilot_seeds(config["real_seeds"])
        for setting in config.get("settings", []):
            setting["seeds"] = pilot_seeds(setting.get("seeds", config["seeds"]))
        config["pilot_restriction"] = {"only_seeds": args.only_seeds, "limit": args.limit}
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output directory has a different configuration")
    _write_json(snapshot, config)
    with threadpool_limits(limits=1):
        suite = config.get("suite", "statistical")
        if suite in ("statistical", "real"):
            result = run_statistical(config, args.output, args.project_root)
        elif suite == "sensitivity":
            summaries = []
            for radius in config["radii"]:
                current = {**config, "theta": float(radius)}
                label = str(radius).replace(".", "p")
                summaries.append(run_statistical(current, args.output / f"radius_{label}", args.project_root))
            result = {key: sum(summary[key] for summary in summaries)
                      for key in ("completed", "skipped", "failures")}
        elif suite == "computational":
            result = run_computational(config, args.output, args.project_root)
        else:
            raise ValueError(f"Unknown suite: {suite}")
    _write_json(args.output / "execution_summary.json", result)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
