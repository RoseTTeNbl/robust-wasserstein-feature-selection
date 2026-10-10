"""Frozen paired validation of complete-certificate candidate screening.

Development verifies the selected implementation; confirmation never tunes it.
Raw results retain all repetitions, failures, actual tests, and solver costs.
"""
# ruff: noqa: E402
import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

import robusthw.oracle as inner
import robusthw.solver as outer
from robusthw import GateOracle, Problem
from robusthw.problem import json_value

METHODS = ("master", "baseline", "curvature")
SETTINGS = dict(time_limit=60., root_rounds=0, tolerance=2e-5,
                relative_tolerance=0., proposal_count=8, certify_every=3,
                core_point=False, gate="risk", cut_mode="tangent",
                max_iterations=10000, max_refinements=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {key: finite(v) for key, v in item.items()}
        if isinstance(item, list):
            return [finite(v) for v in item]
        return None if isinstance(item, float) and not np.isfinite(item) else item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(json_value(value)), indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def data_hash(data):
    return hashlib.sha256(np.concatenate([
        np.asarray(data[key], dtype=float).ravel()
        for key in ("x0", "x1", "q0", "q1")]).tobytes()).hexdigest()


class MeasuredOracle(GateOracle):
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MeasuredOracle.last = self
        self.full_calls = self.continuous_calls = self.repricings = 0
        self.binary_only = True
        self.owned = []

    def solve(self, z, **kwargs):
        self.full_calls += 1
        self.binary_only &= bool(np.all(np.isin(z, (0., 1.))))
        result = super().solve(z, **kwargs)
        self.owned.append(dict(point=np.asarray(z).tolist(), lower=result.value,
                               upper=result.upper, intercept=result.intercept,
                               gradient=result.gradient.tolist(),
                               certificate=result.certificate))
        return result

    def _cvx_solve(self, model, deadline):
        if deadline is None or deadline > perf_counter():
            self.continuous_calls += 1
        return super()._cvx_solve(model, deadline)

    def reprice(self, *args, **kwargs):
        self.repricings += 1
        raise AssertionError("Test transfer is forbidden")


def cases(stage):
    if stage == "development":
        folder = ROOT / "reports/verification/20261010/full_inner_uncertainty/raw"
        for path in sorted(folder.glob("*.json")):
            original = load(path)
            case, data = original["case"], original["generated_data"]
            assert case["data_sha256"] == data_hash(data)
            yield case, data, original["methods"]["enumeration"]
        return
    for loss, (dimension, budget), seed, theta in itertools.product(
            ("overlap", "logistic"), ((7, 2), (10, 3)), (19, 23),
            ((.12, .20), (.30, .45))):
        rng = np.random.default_rng(seed)
        scales = np.r_[1., 1., np.geomspace(.3, 3., dimension-2)]
        x0 = rng.normal(size=(6, dimension)) * scales
        x1 = rng.normal(size=(6, dimension)) * scales
        x1[:, :2] += [1.4, .9]
        screening = np.abs(x0.mean(axis=0)-x1.mean(axis=0)) / (
            x0.std(axis=0)+x1.std(axis=0)+1e-12)
        warm = sorted(np.argsort(-screening, kind="stable")[:budget].tolist())
        data = dict(x0=x0.tolist(), x1=x1.tolist(), q0=[1/6]*6, q1=[1/6]*6)
        case = dict(id=f"confirm_{loss}_d{dimension}_n6_s{seed}_b{budget}_"
                       f"theta{theta[0]}_{theta[1]}",
                    loss=loss, dimension=dimension, budget=budget, seed=seed,
                    theta=list(theta), warm_support=warm, startup="data_screening",
                    screening_scores=screening.tolist(), data_sha256=data_hash(data))
        yield case, data, None


def checks(result, reference, method):
    epsilon = SETTINGS["tolerance"]
    measured, diag = result.get("measured", {}), result.get("diagnostics", {})
    ok = dict(optimal=result["status"] == "optimal",
              reference_complete=reference["status"] == "optimal",
              reference_accuracy=result.get("upper", np.inf)-reference["lower"] <= epsilon,
              reference_consistent=result.get("lower", np.inf) <= reference["upper"]+epsilon
              and result.get("upper", -np.inf) >= reference["lower"]-epsilon,
              binary_only=measured.get("binary_only", False),
              no_repricing=measured.get("repricings", 1) == 0)
    if method != "master" and "evaluations" in diag:
        rows = diag["evaluations"]
        ok.update(one_owned_cut=diag["cut_count"] == len(rows) == result["visited"],
                  unique_supports=len({tuple(r["support"]) for r in rows}) == len(rows),
                  anchor_accuracy=all(r["anchor_gap"] <= epsilon/4 for r in rows),
                  cut_filter=all(r["model_lower"] <= r["upper"]-epsilon/2
                                 for r in result["trace"] if r["phase"] == "acquisition"),
                  actual_only_upper=diag.get("predicted_upper_updates", 1) == 0
                  and diag.get("transferred_tests", 1) == 0)
        if method == "curvature":
            ok["one_owned_fraction"] = diag["fraction_count"] == len(rows)
            ok["fraction_same_result_as_cut"] = all(any(
                own["point"] == r["point"] and own["intercept"] == r["intercept"]
                and own["gradient"] == r["gradient"]
                and own["certificate"].get("tau") == r["owned_fraction"]["tau"]
                and np.array_equal(own["certificate"].get("mu"), r["owned_fraction"]["mu"])
                for own in measured["owned_evaluations"]) for r in rows)
    return ok


def metrics(result):
    measured, diag = result.get("measured", {}), result.get("diagnostics", {})
    return dict(status=result["status"], completed_evaluations=result.get("visited"),
                physical_calls=result.get("oracle_calls"), seconds=result.get("seconds"),
                masters=diag.get("master_calls", sum(r["phase"] == "master"
                                                    for r in result.get("trace", []))),
                master_seconds=sum(r["seconds"] for r in measured.get("master_measurements", [])),
                continuous_calls=measured.get("continuous_calls"),
                lower=result.get("lower"), upper=result.get("upper"), gap=result.get("gap"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "confirmation"))
    parser.add_argument("--root", type=Path,
                        default=ROOT / "reports/verification/20261010/next_selection")
    args = parser.parse_args()
    folder = args.root.resolve() / args.stage
    if (folder / "protocol.json").exists():
        raise SystemExit("Evidence exists; never overwrite a frozen run")
    generated = list(cases(args.stage))
    assert len(generated) == (26 if args.stage == "development" else 16)
    repeats = 1 if args.stage == "development" else 3
    files = sorted((ROOT / "code/robusthw").glob("*.py")) + [Path(__file__)]
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    protocol = dict(stage=args.stage, settings=SETTINGS, methods=list(METHODS),
                    timing_repeats=repeats, sources=sources,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__),
                    data_hash_fields=["x0", "x1", "q0", "q1"],
                    cases=[case for case, _, _ in generated],
                    selection="Locked before confirmation: phase4 complete-certificate screening; "
                              "same GP, pool of 8 and period of 3. No new acquisition mechanism. "
                              "Chosen for one mathematical change and fewer full evaluations with "
                              "limited master overhead; every-step ratio is a retained development control.",
                    scope="Development is a numerical/integration regression after the witnessed "
                          "presolve repair. Confirmation uses the predeclared 16 configurations, "
                          "four generated datasets, and shared screening starts; no retuning.")
    save(folder / "protocol.json", protocol)
    for source in files:
        target = folder / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    original_lp = inner._lp
    original_master, original_ratio, original_oracle = outer._master, outer._ratio_master, outer.GateOracle
    masters = []

    def counted_lp(*args, **kwargs):
        deadline = kwargs.get("deadline")
        if deadline is None or deadline > perf_counter():
            MeasuredOracle.last.continuous_calls += 1
        return original_lp(*args, **kwargs)

    def counted_master(function, kind):
        def wrapped(*args, **kwargs):
            started = perf_counter()
            answer = function(*args, **kwargs)
            masters.append(dict(kind=kind, seconds=perf_counter()-started,
                                status=int(answer.status), nodes=getattr(answer, "mip_node_count", None),
                                retries=getattr(answer, "master_retries", 0)))
            return answer
        return wrapped

    def run(problem, method, case):
        masters.clear()
        try:
            result = outer.solve(problem, **SETTINGS, warm_support=case["warm_support"],
                                 random_state=case["seed"],
                                 search="master" if method == "master" else "adaptive",
                                 use_fractions=method == "curvature")
            record = result.to_dict()
            record["test_certificate"] = (result.test.as_dict() if hasattr(result.test, "as_dict")
                                          else dict(kind="constant", support=list(result.support),
                                                    upper=result.upper, class0_probability=.5))
            oracle = MeasuredOracle.last
            record["measured"] = dict(binary_only=oracle.binary_only, full_calls=oracle.full_calls,
                                       continuous_calls=oracle.continuous_calls, repricings=oracle.repricings,
                                       owned_evaluations=oracle.owned, master_measurements=list(masters))
            return record
        except Exception as error:
            return dict(status="exception", message=str(error))

    inner._lp, outer.GateOracle = counted_lp, MeasuredOracle
    outer._master = counted_master(original_master, "tangent")
    outer._ratio_master = counted_master(original_ratio, "fraction")
    records, aggregate = [], []
    try:
        with threadpool_limits(limits=1):
            for index, (case, data, reference) in enumerate(generated):
                problem = Problem(data["x0"], data["x1"], case["budget"], *case["theta"],
                                  case["loss"], q0=data["q0"], q1=data["q1"])
                if reference is None:
                    reference = outer.enumerate_supports(problem, time_limit=60., tolerance=2e-5).to_dict()
                runs = []
                for repeat in range(repeats):
                    offset = (index + repeat) % len(METHODS)
                    order = METHODS[offset:] + METHODS[:offset]
                    results = {method: run(problem, method, case) for method in order}
                    verified = {method: checks(result, reference, method)
                                for method, result in results.items()}
                    runs.append(dict(repeat=repeat, method_order=list(order), methods=results,
                                     checks=verified))
                    aggregate.extend(dict(case=case["id"], repeat=repeat, method=method,
                                          **metrics(result), all_checks=all(verified[method].values()))
                                     for method, result in results.items())
                record = dict(case=case, data=data, reference=reference, repeats=runs)
                save(folder / "raw" / (case["id"] + ".json"), record)
                records.append(record)
                print(json.dumps(dict(stage=args.stage, case=case["id"],
                                      calls={m: [r["methods"][m].get("visited") for r in runs] for m in METHODS},
                                      passed=all(all(v.values()) for r in runs for v in r["checks"].values()))), flush=True)
    finally:
        inner._lp, outer._master, outer._ratio_master, outer.GateOracle = (
            original_lp, original_master, original_ratio, original_oracle)
    assert all(sha(ROOT / p) == h for p, h in sources.items()), "Source changed during validation"
    save(folder / "summary.json", dict(case_count=len(records), timing_repeats=repeats,
                                        rows=aggregate, all_pass=all(r["all_checks"] for r in aggregate)))
    seal_files = [folder / "protocol.json", folder / "summary.json", *sorted((folder / "raw").glob("*.json"))]
    save(folder / "seal.json", dict(files={p.relative_to(folder).as_posix(): sha(p) for p in seal_files},
                                     sources=sources, all_pass=all(r["all_checks"] for r in aggregate)))


if __name__ == "__main__":
    main()
