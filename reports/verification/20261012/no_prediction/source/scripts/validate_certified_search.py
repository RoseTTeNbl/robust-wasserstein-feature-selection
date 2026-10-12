"""Retrospective paired ablation on the sixteen frozen small configurations.

Both master-only controls and both historical GP controls solve the same binary
inner problems. Enumeration references and original data are verified by hash.
All repetitions, failures, certificates, timings, and source versions are kept.
"""
# ruff: noqa: E402
import argparse
import platform
import shutil
import traceback
from pathlib import Path
from time import perf_counter

import validate_next_selection as v

import robusthw.history as history

METHODS = ("cuts_tangent", "cuts_fraction", "gp_tangent", "gp_fraction")
SETTINGS = dict(v.SETTINGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=v.ROOT /
                        "reports/verification/20261012/no_prediction")
    args = parser.parse_args()
    folder = args.root.resolve()
    if (folder / "protocol.json").exists():
        raise SystemExit("Evidence exists; never overwrite a frozen run")
    generated = list(v.cases("confirmation"))
    reference_root = v.ROOT / "reports/verification/20261010/next_selection/current/confirmation"
    references, reference_hashes = {}, {}
    for case, data, _ in generated:
        path = reference_root / "raw" / (case["id"] + ".json")
        original = v.load(path)
        assert original["case"] == case
        assert v.data_hash(original["data"]) == v.data_hash(data) == case["data_sha256"]
        assert original["reference"]["status"] == "optimal"
        references[case["id"]] = original["reference"]
        reference_hashes[path.relative_to(v.ROOT).as_posix()] = v.sha(path)
    files = sorted((v.ROOT / "code/robusthw").glob("*.py")) + [Path(__file__), Path(v.__file__)]
    sources = {p.relative_to(v.ROOT).as_posix(): v.sha(p) for p in files}
    v.save(folder / "protocol.json", dict(stage="ablation", methods=list(METHODS),
        timing_repeats=3, settings=SETTINGS, sources=sources,
        versions=dict(python=platform.python_version(), numpy=v.np.__version__,
                      scipy=v.scipy.__version__),
        data_hash_fields=["x0", "x1", "q0", "q1"],
        cases=[case for case, _, _ in generated], reference_sources=reference_hashes,
        selection="Frozen before execution: master-only tangent, master-only tangent plus fraction; "
                  "historical GP tangent, historical GP tangent plus fraction. All start from "
                  "the same data screen and require complete binary inner evaluations.",
        scope="Retrospective comparison on the existing sixteen configurations (four base datasets). "
              "No retuning or independent statistical confirmation. Removing GP also removes "
              "its candidate pool and changes master frequency; this is a process comparison, "
              "not an isolated estimate of the effect of GP."))
    for source in files:
        target = folder / "source" / source.relative_to(v.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    original_lp = v.inner._lp
    original_master, original_ratio = v.outer._master, v.outer._ratio_master
    original_oracle, original_gp = v.outer.GateOracle, history.FixedKernelPredictor.__init__
    masters, gp_instances = [], [0]

    def counted_lp(*args, **kwargs):
        deadline = kwargs.get("deadline")
        if deadline is None or deadline > perf_counter():
            v.MeasuredOracle.last.continuous_calls += 1
        return original_lp(*args, **kwargs)

    def counted_master(function, kind):
        def wrapped(*args, **kwargs):
            started = perf_counter()
            result = function(*args, **kwargs)
            masters.append(dict(kind=kind, seconds=perf_counter()-started,
                status=int(result.status), nodes=getattr(result, "mip_node_count", None),
                retries=getattr(result, "master_retries", 0)))
            return result
        return wrapped

    def counted_gp(self, *args, **kwargs):
        gp_instances[0] += 1
        original_gp(self, *args, **kwargs)

    def run(problem, method, case):
        masters.clear()
        gp_instances[0] = 0
        started = perf_counter()
        v.MeasuredOracle.last = None
        try:
            result = v.outer.solve(problem, **SETTINGS,
                warm_support=case["warm_support"], random_state=case["seed"],
                search="cuts" if method.startswith("cuts_") else "adaptive",
                use_fractions=method.endswith("fraction"))
            record = result.to_dict()
            record["test_certificate"] = (result.test.as_dict()
                if hasattr(result.test, "as_dict") else
                dict(kind="constant", support=list(result.support), upper=result.upper,
                     class0_probability=.5))
        except Exception as error:
            record = dict(status="exception", message=str(error),
                          traceback=traceback.format_exc(), seconds=perf_counter()-started)
        oracle = v.MeasuredOracle.last
        record["measured"] = dict(
            binary_only=oracle.binary_only if oracle is not None else False,
            full_calls=oracle.full_calls if oracle is not None else 0,
            continuous_calls=oracle.continuous_calls if oracle is not None else 0,
            repricings=oracle.repricings if oracle is not None else 0,
            owned_evaluations=oracle.owned if oracle is not None else [],
            master_measurements=list(masters), gp_instances=gp_instances[0])
        return record

    def checks(record, reference, method):
        checked = v.checks(record, reference,
                          "curvature" if method.endswith("fraction") else "baseline")
        checked["outer_accuracy"] = record.get("gap", v.np.inf) <= SETTINGS["tolerance"]
        if method.startswith("cuts_"):
            diag, trace = record.get("diagnostics", {}), record.get("trace", [])
            checked.update(no_prediction=record["measured"]["gp_instances"] == 0,
                no_fractional_evaluations=diag.get("fractional_root_calls", -1) == 0,
                screened_candidates=all(row["model_lower"] <= row["pre_upper"]-
                    SETTINGS["tolerance"]/2 for row in trace if row["phase"] == "integer"),
                accurate_master_candidates=all(row["candidate_gap"] <= SETTINGS["tolerance"]/4
                    for row in trace if row["phase"] == "master" and "candidate_gap" in row))
        return checked

    v.inner._lp, v.outer.GateOracle = counted_lp, v.MeasuredOracle
    v.outer._master = counted_master(original_master, "tangent")
    v.outer._ratio_master = counted_master(original_ratio, "fraction")
    history.FixedKernelPredictor.__init__ = counted_gp
    aggregate = []
    try:
        with v.threadpool_limits(limits=1):
            for index, (case, data, _) in enumerate(generated):
                problem = v.Problem(data["x0"], data["x1"], case["budget"],
                    *case["theta"], case["loss"], q0=data["q0"], q1=data["q1"])
                reference, runs = references[case["id"]], []
                for repeat in range(3):
                    offset = (index+repeat) % len(METHODS)
                    order = METHODS[offset:] + METHODS[:offset]
                    results = {method: run(problem, method, case) for method in order}
                    verified = {method: checks(record, reference, method)
                                for method, record in results.items()}
                    runs.append(dict(repeat=repeat, method_order=list(order),
                                     methods=results, checks=verified))
                    aggregate.extend(dict(case=case["id"], repeat=repeat, method=method,
                        **v.metrics(record), all_checks=all(verified[method].values()))
                        for method, record in results.items())
                v.save(folder / "raw" / (case["id"] + ".json"),
                       dict(case=case, data=data, reference=reference, repeats=runs))
                print(v.json.dumps(dict(case=case["id"],
                    completed={method: [r["methods"][method].get("visited") for r in runs]
                               for method in METHODS},
                    passed=all(all(c.values()) for r in runs for c in r["checks"].values()))),
                    flush=True)
    finally:
        v.inner._lp, v.outer._master, v.outer._ratio_master, v.outer.GateOracle = (
            original_lp, original_master, original_ratio, original_oracle)
        history.FixedKernelPredictor.__init__ = original_gp
    assert all(v.sha(v.ROOT / p) == h for p, h in sources.items()), "Source changed during validation"
    assert all(v.sha(v.ROOT / p) == h for p, h in reference_hashes.items()), "Reference changed"
    all_pass = all(row["all_checks"] for row in aggregate)
    v.save(folder / "summary.json", dict(case_count=len(generated), timing_repeats=3,
                                       rows=aggregate, all_pass=all_pass))
    seal_files = [folder / "protocol.json", folder / "summary.json",
                  *sorted((folder / "raw").glob("*.json"))]
    v.save(folder / "seal.json", dict(files={p.relative_to(folder).as_posix(): v.sha(p)
                                            for p in seal_files},
                                     sources=sources, all_pass=all_pass))


if __name__ == "__main__":
    main()
