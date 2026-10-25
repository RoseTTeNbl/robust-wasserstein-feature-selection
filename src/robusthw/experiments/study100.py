"""A frozen, small 100-dimensional study of the current hinge OA formulation.

Run with ``python -m robusthw.experiments.study100 --output outputs/study100``.
This module owns sampling, calibration and recording, never optimization algebra.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn

from robusthw.core import MODEL_VERSION, ZERO_SCORE_RISK, Problem, SolveOptions, fit_fixed
from robusthw.core.selection import select

from .study100_baselines import (
    fisher_ranking,
    fit_logistic,
    l1_support,
    logistic_score,
    random_support,
)

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("outer_approximation", "fisher_robust", "l1_robust", "random_robust", "l1_logistic")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _stream(seed, purpose):
    tag = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, tag]))


def make_data(config, seed):
    """Known sparse Gaussian mean shift; fit centering/scaling on training only."""
    dimension, signal_count = config["dimension"], config["signal_count"]
    truth = tuple(
        sorted(
            int(j) for j in _stream(seed, "support").choice(dimension, signal_count, replace=False)
        )
    )
    shift = np.zeros(dimension)
    shift[list(truth)] = config["mean_amplitude"]
    data = {}
    for split, sizes in (
        ("train", config["train_sizes"]),
        ("validation", config["validation_sizes"]),
        ("test", config["test_sizes"]),
    ):
        for k, size in enumerate(sizes):
            noise = _stream(seed, f"{split}-{k}").normal(size=(size, dimension))
            mean = (1 if k == 0 else -1) * shift
            data[f"{split}{k}"] = mean + noise
            if split == "test":
                data[f"stress{k}"] = mean + config["stress_noise_scale"] * noise
    pooled = np.vstack((data["train0"], data["train1"]))
    center, scale = pooled.mean(axis=0), pooled.std(axis=0, ddof=1)
    scale = np.maximum(scale, 1e-12)
    data = {name: (values - center) / scale for name, values in data.items()}
    return data, truth, {"center": center, "scale": scale}


def prediction_metrics(score0, score1, *, hinge=True):
    """Equal class weight. A zero score is deterministically assigned to class 0."""
    scores = [np.asarray(score, float) for score in (score0, score1)]
    if any(score.ndim != 1 or not len(score) or not np.isfinite(score).all() for score in scores):
        raise ValueError("Prediction scores must be nonempty finite vectors")
    error0 = float(np.mean(scores[0] < 0))
    error1 = float(np.mean(scores[1] >= 0))
    return {
        "error0": error0,
        "error1": error1,
        "error_sum": error0 + error1,
        "balanced_error": (error0 + error1) / 2,
        "empirical_hinge": (
            float(np.maximum(0, 1 - scores[0]).mean() + np.maximum(0, 1 + scores[1]).mean())
            if hinge
            else None
        ),
    }


def _radii(config, scale):
    return float(scale) * np.sqrt(config["radius_reference_size"] / np.array(config["train_sizes"]))


def _options(config, *, search=False):
    return SolveOptions(
        time_limit=config["oa_time_limit"] if search else config["fit_time_limit"],
        fit_time_limit=config["fit_time_limit"],
        absolute_tolerance=config["absolute_tolerance"],
        inner_tolerance=config["inner_tolerance"],
        threads=config["threads"],
    )


def calibrate(config):
    """Choose shared radius on a fixed Fisher support and L1 C on development only."""
    seed = config["development_seed"]
    data, _, _ = make_data(config, seed)
    train = data["train0"], data["train1"]
    support = tuple(sorted(fisher_ranking(*train)[: config["budget"]]))
    radius_trials, logistic_trials = [], []
    for scale in config["radius_grid"]:
        fit = fit_fixed(
            Problem(*train, _radii(config, scale)), support, _options(config), with_cut=False
        )
        metrics = prediction_metrics(
            fit.predict_score(data["validation0"]), fit.predict_score(data["validation1"])
        )
        radius_trials.append(
            {"radius_scale": scale, "status": fit.status, "upper": fit.upper, "metrics": metrics}
        )
    valid_radii = [trial for trial in radius_trials if trial["status"] == "optimal"]
    if not valid_radii:
        raise RuntimeError("No development radius candidate completed successfully")
    chosen_radius = min(
        valid_radii,
        key=lambda trial: (
            trial["metrics"]["error_sum"],
            trial["metrics"]["empirical_hinge"],
            trial["radius_scale"],
        ),
    )["radius_scale"]
    for C in config["logistic_C_grid"]:
        support_l1, info = l1_support(*train, config["budget"], C, seed)
        model = fit_logistic(*train, support_l1, config["logistic_refit_C"], seed)
        metrics = prediction_metrics(
            logistic_score(model, data["validation0"], support_l1),
            logistic_score(model, data["validation1"], support_l1),
            hinge=False,
        )
        logistic_trials.append({"C": C, "support": support_l1, "metrics": metrics, **info})
    chosen_C = min(logistic_trials, key=lambda trial: (trial["metrics"]["error_sum"], trial["C"]))[
        "C"
    ]
    return {
        "development_seed": seed,
        "radius_scale": chosen_radius,
        "logistic_C": chosen_C,
        "radius_trials": radius_trials,
        "logistic_trials": logistic_trials,
        "radius_calibration_support": support,
        "selection_rule": "minimum validation error sum; radius ties: hinge then smaller radius; C ties: smaller C",
        "scope": "development only; shared fixed parameters for all confirmation seeds",
    }


def _fit_record(fit):
    return {
        "support": fit.support,
        "scores": fit.scores,
        "source_bounds": fit.source_bounds,
        "prices": fit.prices,
        "upper": fit.upper,
        "lower": fit.lower,
        "gap": fit.gap,
        "status": fit.status,
        "diagnostics": fit.diagnostics,
    }


def run_seed(config, calibration, seed, folder):
    data, truth, transform = make_data(config, seed)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "data.npz", **data, **transform, signal_indices=truth)
    train = data["train0"], data["train1"]
    problem = Problem(*train, _radii(config, calibration["radius_scale"]))
    started = perf_counter()
    ranking = fisher_ranking(*train)
    fisher = tuple(sorted(ranking[: config["budget"]]))
    fisher_seconds = perf_counter() - started
    started = perf_counter()
    l1, l1_info = l1_support(*train, config["budget"], calibration["logistic_C"], seed)
    l1_seconds = perf_counter() - started
    started = perf_counter()
    random = random_support(problem.dimension, config["budget"], seed)
    random_seconds = perf_counter() - started
    supports = {
        "fisher_robust": (fisher, fisher_seconds),
        "l1_robust": (l1, l1_seconds),
        "random_robust": (random, random_seconds),
        "l1_logistic": (l1, l1_seconds),
    }
    rows = []
    for method in METHODS:
        started = perf_counter()
        print(f"seed={seed} method={method}", flush=True)
        try:
            if method == "outer_approximation":
                result = select(
                    problem,
                    config["budget"],
                    "global",
                    _options(config, search=True),
                    initial_support=fisher,
                    ranked_features=ranking,
                )
                fit, support = result.fit, result.support
                training_seconds = perf_counter() - started + fisher_seconds
                detail = {
                    "fit": _fit_record(fit),
                    "trace": result.trace,
                    "diagnostics": result.diagnostics,
                }
                row = {
                    "status": result.status,
                    "robust_upper": result.upper,
                    "global_lower": result.lower,
                    "global_gap": result.gap,
                    "fit_count": result.fit_count,
                    "master_count": result.master_count,
                    "inner_gap": fit.gap,
                }
                predict = fit.predict_score
            elif method == "l1_logistic":
                support, selection_seconds = supports[method]
                model = fit_logistic(*train, support, config["logistic_refit_C"], seed)
                training_seconds = perf_counter() - started + selection_seconds
                detail = {
                    "selection": l1_info,
                    "coefficients": model.coef_,
                    "intercept": model.intercept_,
                    "C": config["logistic_refit_C"],
                }
                row = {
                    "status": "complete",
                    "robust_upper": None,
                    "global_lower": None,
                    "global_gap": None,
                    "fit_count": 1,
                    "master_count": 0,
                    "inner_gap": None,
                }

                def predict(x):
                    return logistic_score(model, x, support)
            else:
                support, selection_seconds = supports[method]
                fit = fit_fixed(problem, support, _options(config), with_cut=False)
                training_seconds = perf_counter() - started + selection_seconds
                detail = {"fit": _fit_record(fit)}
                if method == "l1_robust":
                    detail["selection"] = l1_info
                row = {
                    "status": fit.status,
                    "robust_upper": fit.upper,
                    "global_lower": None,
                    "global_gap": None,
                    "fit_count": 1,
                    "master_count": 0,
                    "inner_gap": fit.gap,
                }
                predict = fit.predict_score
            row.update(
                {
                    "seed": seed,
                    "method": method,
                    "support": support,
                    "signal_hits": len(set(support) & set(truth)),
                    "training_seconds": training_seconds,
                }
            )
            prediction_arrays = {}
            for setting in ("test", "stress"):
                score0, score1 = predict(data[f"{setting}0"]), predict(data[f"{setting}1"])
                prediction_arrays.update({f"{setting}0": score0, f"{setting}1": score1})
                row[setting] = prediction_metrics(score0, score1, hinge=method != "l1_logistic")
            np.savez_compressed(folder / f"{method}-predictions.npz", **prediction_arrays)
            _save(folder / f"{method}.json", {"summary": row, **detail})
        except Exception as error:
            row = {
                "seed": seed,
                "method": method,
                "status": "exception",
                "training_seconds": perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
            }
            _save(folder / f"{method}.json", {"summary": row})
        rows.append(row)
    record = {"seed": seed, "signal_indices": truth, "radii": problem.radii, "rows": rows}
    _save(folder / "result.json", record)
    return record


def _summarize(records, output):
    rows = [row for record in records for row in record["rows"]]
    fields = [
        "seed",
        "method",
        "status",
        "support",
        "signal_hits",
        "training_seconds",
        "robust_upper",
        "global_lower",
        "global_gap",
        "inner_gap",
        "fit_count",
        "master_count",
        "test_error0",
        "test_error1",
        "test_error_sum",
        "test_empirical_hinge",
        "stress_error_sum",
        "stress_empirical_hinge",
        "error",
    ]
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for setting in ("test", "stress"):
                for key, value in row.get(setting, {}).items():
                    if f"{setting}_{key}" in fields:
                        flat[f"{setting}_{key}"] = value
            writer.writerow(flat)
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        completed = [row for row in subset if "test" in row]
        summary[method] = {
            "runs": len(subset),
            "evaluated_runs": len(completed),
            "statuses": [row["status"] for row in subset],
            "mean_test_error_sum": float(np.mean([row["test"]["error_sum"] for row in completed]))
            if completed
            else None,
            "mean_stress_error_sum": float(
                np.mean([row["stress"]["error_sum"] for row in completed])
            )
            if completed
            else None,
            "mean_signal_hits": float(np.mean([row["signal_hits"] for row in completed]))
            if completed
            else None,
            "mean_training_seconds": float(np.mean([row["training_seconds"] for row in subset])),
            "global_gaps": [row.get("global_gap") for row in subset],
        }
    _save(output / "summary.json", summary)
    return summary


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if MODEL_VERSION != "full_support_standard_hinge_v2" or ZERO_SCORE_RISK != 2:
        raise ValueError("This protocol requires the current standard-hinge formulation")
    if config["development_seed"] in config["confirmation_seeds"]:
        raise ValueError("Development and confirmation seeds must be disjoint")
    if len(set(config["confirmation_seeds"])) != len(config["confirmation_seeds"]):
        raise ValueError("Confirmation seeds must be distinct")
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "src/robusthw/core").glob("*.py")) + [
        ROOT / "src/robusthw/search.py",
        Path(__file__),
        Path(__file__).with_name("study100_baselines.py"),
    ]
    provenance = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    identity = _digest({"config": config, "sources": provenance})
    plan_path = output / "protocol.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Configuration or code changed; choose a new output directory")
    else:
        _save(
            plan_path,
            {
                "identity": identity,
                "model_version": MODEL_VERSION,
                "zero_score_risk": ZERO_SCORE_RISK,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "source_hashes": provenance,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                },
            },
        )
    calibration_path = output / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        calibration = calibrate(config)
        _save(calibration_path, calibration)
    records = []
    for seed in config["confirmation_seeds"]:
        folder = output / f"seed-{seed}"
        result_path = folder / "result.json"
        record = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else run_seed(config, calibration, seed, folder)
        )
        records.append(record)
        _summarize(records, output)
    return _summarize(records, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study100.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
