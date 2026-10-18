"""Exercise the pipeline with a tiny subprocess that implements the run contract.

The subprocess never performs a research experiment; all files are pytest TEMP
fixtures. These tests check scheduling, integrity barriers and failure handling.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "execute_study.py"
SPEC = importlib.util.spec_from_file_location("execute_study", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)

FAKE_RUNNER = """
import hashlib
import json
import os
import sys
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

path = Path(sys.argv[2])
suite = path.parent.name
if os.environ.get("FAKE_CRASH_SUITE") == suite:
    print("deliberate worker crash", file=sys.stderr)
    raise SystemExit(7)
manifest = json.loads(path.read_text("utf-8"))
spec = manifest["specs"][0]
record_path = path.parent / "records" / (spec["independent_data_id"] + ".json")
if record_path.exists():
    record = json.loads(record_path.read_text("utf-8"))
else:
    record = {"spec": spec, "manifest_hash": manifest["manifest_hash"],
        "status": "failed" if os.environ.get("FAKE_FAILED_SUITE") == suite else "completed"}
    record["record_hash"] = digest(record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")
failed = int(record["status"] == "failed")
execution = {"manifest_hash": manifest["manifest_hash"], "planned": 1, "attempted": 1,
    "completed": 1-failed, "failed": failed, "partial_execution": False,
    "workers": int(sys.argv[sys.argv.index("--workers")+1]),
    "thread_env": {k: os.environ.get(k) for k in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    "source_file": __file__,
    "records": [{"data_id": spec["independent_data_id"], "status": record["status"],
        "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()}]}
(path.parent / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
if os.environ.get("FAKE_TAMPER_SUITE") == suite:
    record_path.write_text("{}", encoding="utf-8")
raise SystemExit(failed)
"""


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    source = root / "frozen-source" / "src" / "robusthw" / "__main__.py"
    source.parent.mkdir(parents=True)
    source.write_text(FAKE_RUNNER, encoding="utf-8")
    package = source.with_name("__init__.py")
    package.write_text("", encoding="utf-8")
    name = "src/robusthw/__main__.py"
    sources = {
        name: hashlib.sha256(source.read_bytes()).hexdigest(),
        "src/robusthw/__init__.py": hashlib.sha256(package.read_bytes()).hexdigest(),
    }
    for suite in pipeline.SUITES:
        path = root / suite / "protocol.json"
        path.parent.mkdir(parents=True)
        manifest = {
            "suite": suite,
            "source_hashes": sources,
            "specs": [{"independent_data_id": suite + "/fixture"}],
        }
        manifest["manifest_hash"] = pipeline._hash(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with ZipFile(path.with_suffix(".sources.zip"), "w") as archive:
            archive.write(source, name)
            archive.write(package, "src/robusthw/__init__.py")
    return root


def test_pipeline_orders_suites_uses_frozen_code_and_runs_report_callback(study, tmp_path):
    report = tmp_path / "refresh_report.py"
    report.write_text(
        "import sys\nfrom pathlib import Path\n"
        "root=Path(sys.argv[sys.argv.index('--study-root')+1])\n"
        "with (root/'report_calls.txt').open('a') as f: f.write('refreshed\\n')\n",
        encoding="utf-8",
    )
    assert pipeline.run_pipeline(study, tmp_path / "data", report_script=report) == 0
    state = pipeline._read(study / "pipeline.json")
    assert state["status"] == "completed" and state["phase"] is None
    experiments = [phase for phase in state["phases"] if phase["kind"] == "experiment"]
    assert [phase["suite"] for phase in experiments] == list(pipeline.DEFAULT_SUITES)
    assert len((study / "report_calls.txt").read_text().splitlines()) == 5
    for phase in state["phases"]:
        assert phase["status"] == "completed" and phase["exit_code"] == 0
        assert phase["pid"] > 0 and Path(phase["stdout"]).is_file()
        assert Path(phase["stderr"]).is_file()
    for suite in pipeline.DEFAULT_SUITES:
        execution = pipeline._read(study / suite / "execution.json")
        assert execution["workers"] == (1 if suite == "scaling" else 2)
        assert set(execution["thread_env"].values()) == {"1"}
        assert Path(execution["source_file"]).is_relative_to(study / "frozen-source")
    shifted = next(phase for phase in experiments if phase["suite"] == "shift")
    assert shifted["command"][-2:] == ["--reuse-root", str(study / "main")]
    assert not (study / "pipeline.lock").exists()


def test_complete_failed_outcomes_continue_and_resume_keeps_raw_records(
    study, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_FAILED_SUITE", "public")
    suites = ("main", "public", "ablation")
    assert pipeline.run_pipeline(study, tmp_path / "data", suites=suites) == 1
    state = pipeline._read(study / "pipeline.json")
    assert state["status"] == "completed_with_failures"
    assert [phase["status"] for phase in state["phases"]] == [
        "completed",
        "completed_with_failures",
        "completed",
    ]
    before = {path: path.read_bytes() for path in study.glob("*/records/**/*.json")}
    monkeypatch.delenv("FAKE_FAILED_SUITE")
    assert pipeline.run_pipeline(study, tmp_path / "data", suites=suites) == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    resumed = pipeline._read(study / "pipeline.json")
    assert len(resumed["history"]) == 1
    assert resumed["history"][0]["status"] == "completed_with_failures"


def test_unexpected_process_exit_stops_before_the_next_suite(study, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_SUITE", "public")
    assert pipeline.run_pipeline(study, tmp_path, suites=("main", "public", "ablation")) == 3
    state = pipeline._read(study / "pipeline.json")
    assert state["status"] == "stopped_unexpected_error"
    assert "code 7" in state["error"]
    assert state["phases"][-1]["status"] == "unexpected_error"
    assert not (study / "ablation" / "execution.json").exists()
    assert "deliberate worker crash" in Path(state["phases"][-1]["stderr"]).read_text()


def test_changed_frozen_checkout_stops_before_any_subprocess(study, tmp_path):
    source = study / "frozen-source" / "src" / "robusthw" / "__main__.py"
    source.write_text("changed", encoding="utf-8")
    assert pipeline.run_pipeline(study, tmp_path, suites=("main", "public")) == 2
    state = pipeline._read(study / "pipeline.json")
    assert state["status"] == "stopped_integrity_error"
    assert "Frozen checkout changed" in state["error"]
    assert state["phases"][-1]["status"] == "integrity_error"
    assert not (study / "main" / "execution.json").exists()


def test_changed_raw_record_stops_before_the_next_suite(study, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TAMPER_SUITE", "main")
    assert pipeline.run_pipeline(study, tmp_path, suites=("main", "public")) == 2
    state = pipeline._read(study / "pipeline.json")
    assert state["status"] == "stopped_integrity_error"
    assert not (study / "public" / "execution.json").exists()


def test_running_pipeline_lock_prevents_a_second_coordinator(study, tmp_path):
    (study / "pipeline.lock").write_text(str(os.getpid()), encoding="ascii")
    with pytest.raises(RuntimeError, match="already running"):
        pipeline.run_pipeline(study, tmp_path, suites=("main",))


def test_scaling_must_be_last_and_shift_must_follow_main(study, tmp_path):
    with pytest.raises(ValueError, match="Scaling must run last"):
        pipeline.run_pipeline(study, tmp_path, suites=("scaling", "main"))
    with pytest.raises(ValueError, match="Shift must follow main"):
        pipeline.run_pipeline(study, tmp_path, suites=("shift", "main"))
