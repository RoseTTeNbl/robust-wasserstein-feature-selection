"""Publication repair preserves certificates, provenance, and a shared budget."""
import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from robusthw.experiments.data import synthetic_trial
from robusthw.problem import json_value

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/repair_failures.py"
spec = importlib.util.spec_from_file_location("repair_failures", SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def selection(support, lower, upper, seconds, visited, *, status="time_limit", calls=3):
    return dict(support=list(support), lower=lower, upper=upper, seconds=seconds,
                visited=visited, oracle_calls=calls, gap=upper-lower, status=status,
                weights=None, diagnostics={"oracle_seconds": seconds*.3,
                                           "master_seconds": seconds*.6})


def trace(supports):
    return [dict(phase="integer", support=list(s), seconds=i+1., lower=0., upper=1.)
            for i, s in enumerate(supports)]


def test_merge_keeps_best_initial_incumbent_and_strongest_lower():
    initial = selection((0,), .1, .25, 5., 2, calls=5)
    restart = selection((1,), .2, .4, 3., 2, calls=4)
    merged, combined, source = repair.merge_selection(
        initial, restart, trace([(0,), (1,)]), trace([(1,), (2,)]), budget=30.)
    assert source == "initial" and merged["support"] == [0]
    assert merged["lower"] == .2 and merged["upper"] == .25
    assert merged["seconds"] == 8. and merged["oracle_calls"] == 9
    assert merged["visited"] == 3
    assert merged["diagnostics"]["effort_visited"] == 4
    assert merged["diagnostics"]["oracle_seconds"] == pytest.approx(2.4)
    assert merged["diagnostics"]["master_seconds"] == pytest.approx(4.8)
    assert combined[2]["seconds"] == 6.
    assert combined[2]["lower"] == .1 and combined[2]["upper"] == .25
    assert merged["status"] == "time_limit"


def test_merge_keeps_best_restart_incumbent_and_original_lower():
    initial = selection((0,), .2, .5, 5., 1)
    restart = selection((1,), .1, .3, 3., 1)
    merged, _, source = repair.merge_selection(initial, restart, trace([(0,)]),
                                               trace([(1,)]), budget=30.)
    assert source == "restart" and merged["support"] == [1]
    assert merged["lower"] == .2 and merged["upper"] == .3


def test_complementary_valid_bounds_can_certify_optimality():
    initial = selection((0,), .2, .3, 5., 1)
    restart = selection((1,), .299999, .4, 3., 1)
    merged, _, _ = repair.merge_selection(initial, restart, trace([(0,)]),
                                         trace([(1,)]), budget=30.)
    assert merged["status"] == "optimal"


def test_union_visit_count_requires_complete_traces():
    with pytest.raises(ValueError, match="distinct visited"):
        repair.merge_selection(selection((0,), .1, .5, 5., 2),
                               selection((1,), .2, .4, 3., 1),
                               trace([(0,)]), trace([(1,)]), budget=30.)


def fixture_record(suite="statistical"):
    common = dict(theta=.25, time_limit=30., seeds=[3])
    setting = dict(dimension=3, signal_size=1, budget=1, n_select=2,
                   n_head=2, n_classification=2, mechanism="mean", seeds=[3])
    config = {**common, "suite": suite}
    config.update({"settings": [setting]} if suite == "computational" else setting)
    trial = synthetic_trial("mean", 3, dimension=3, signal_size=1, n_select=2,
                            n_head=2, n_classification=2)
    record = {**json_value(trial.as_dict()), "method": "risk", "budget": 1, "theta": .25,
              "loss": "overlap", "config_sha256": sha256(
                  json.dumps(config, sort_keys=True).encode()).hexdigest(),
              "selection": selection((0,), .05, .2, 5., 2, status="master_failure"),
              "trace_file": "traces/example.json", "total_seconds": 8.,
              "evaluation_seconds": 1., "run_status": "complete"}
    return config, record


@pytest.mark.parametrize("suite", ["statistical", "computational"])
def test_restored_trial_exactly_matches_original_metadata(suite):
    config, record = fixture_record(suite)
    trial, current = repair.restore_trial(record, config, suite)
    assert json_value(trial.as_dict())["preprocessing"] == record["preprocessing"]
    assert current["budget"] == 1
    damaged = copy.deepcopy(record)
    damaged["column_order"][0] = -1
    with pytest.raises(ValueError, match="column_order"):
        repair.restore_trial(damaged, config, suite)


def test_cohort_archival_published_provenance_and_remaining_budget(tmp_path, monkeypatch):
    config, record = fixture_record()
    results = tmp_path / "results"
    path = results / "statistical/records/example.json"
    repair._write_json(path, record)
    repair._write_json(results / "statistical/config.json", config)
    repair._write_json(results / "statistical/traces/example.json", trace([(0,), (1,)]))
    entries = repair.freeze_cohort(results, tmp_path, tmp_path / "_archive/numerical")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["selection"]["status"] == "master_failure"
    assert entry["suite"] == "statistical"
    assert entry["record_relative_path"] == "statistical/records/example.json"
    assert repair.read_json(results / "remediation/initial_failures.json") == entries
    assert repair.read_json(tmp_path / entry["archive_record_path"]) == record
    assert "trace" not in entry and "x0" not in entry
    calls, evaluated = [], []

    def restart(problem, current, method):
        calls.append(current["time_limit"])
        return {**selection((1,), .1, .4, 3., 1), "trace": trace([(1,)])}

    def evaluate(trial, chosen, current):
        evaluated.append(chosen["support"])
        return {"metrics": {"fresh_support": chosen["support"]}, "evaluation_seconds": 2.}

    ticks = iter([10., 15.])
    monkeypatch.setattr(repair, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(repair, "_global_selection", restart)
    monkeypatch.setattr(repair, "evaluate", evaluate)
    new, skipped = repair.repair_one(entry, results, tmp_path)
    assert not skipped
    assert calls == [25.]  # Restart receives 30 - the original 5 seconds.
    assert evaluated == [[0]]  # Refit heads for the strongest feasible incumbent.
    assert new["selection"]["seconds"] == 8.
    assert new["total_seconds"] == 13. and new["evaluation_seconds"] == 3.
    assert new["config_sha256"] == record["config_sha256"]
    assert new["numerical_remediation"]["initial_status"] == "master_failure"
    assert new["numerical_remediation"]["restart_time_budget"] == 25.
    assert new["selection"]["visited"] == 2
    assert repair.freeze_cohort(results, tmp_path, tmp_path / "_archive/numerical") == entries
    again, skipped = repair.repair_one(entry, results, tmp_path)
    assert skipped and again == new


def test_command_requires_runners_finished_confirmation(tmp_path):
    with pytest.raises(SystemExit) as exc:
        repair.main(["--project-root", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_exhausted_shared_budget_does_not_start_another_selection(tmp_path, monkeypatch):
    config, record = fixture_record()
    record["selection"]["seconds"] = 30.01
    record["total_seconds"] = 32.
    results = tmp_path / "results"
    repair._write_json(results / "statistical/records/example.json", record)
    repair._write_json(results / "statistical/config.json", config)
    repair._write_json(results / "statistical/traces/example.json", trace([(0,), (1,)]))
    entry = repair.freeze_cohort(results, tmp_path, tmp_path / "_archive/numerical")[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("No selection budget remains")

    ticks = iter([10., 12.])
    monkeypatch.setattr(repair, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(repair, "_global_selection", forbidden)
    monkeypatch.setattr(repair, "evaluate", lambda *args:
                        {"metrics": {}, "evaluation_seconds": 2.})
    new, _ = repair.repair_one(entry, results, tmp_path)
    assert new["selection"]["status"] == "time_limit"
    assert new["selection"]["seconds"] == 30.01
    assert new["numerical_remediation"]["restart_time_budget"] == 0.
    assert new["numerical_remediation"]["restart_selection_seconds"] == 0.
    assert new["selection"]["support"] == [0]
