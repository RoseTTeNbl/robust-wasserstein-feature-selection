"""Report integrity checks: no moving-data snapshot or unreviewed replacement."""

import importlib.util
import sys
from pathlib import Path

import pytest

from robusthw.experiments.storage import file_hash, write_json


@pytest.fixture
def report_tools(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "report_builder_test", scripts / "build_experiment_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_uses_only_raw_paths_in_its_summary_snapshot(tmp_path, report_tools):
    known, newly_arrived = tmp_path / "known.json", tmp_path / "later.json"
    write_json(known, {"identity": "summarized"})
    write_json(newly_arrived, {"identity": "arrived later"})
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "timing_runs.csv").write_text(
        "source_path\n" + str(known) + "\n" + str(known) + "\n", encoding="utf-8"
    )
    assert report_tools.observed_records(tmp_path) == [{"identity": "summarized"}]


def test_tables_do_not_share_mutable_rows(report_tools):
    from presentation_report import default_data

    data = default_data()
    data["tables"]["public"]["rows"][0][1] = 123
    assert data["tables"]["main"]["rows"][0][1] is None


def test_unobserved_planned_groups_do_not_generate_result_figures(report_tools):
    from plot_experiments import groups

    summary = {"groups": [{"n_instances": 0, "group": {"suite": "main"}}]}
    assert groups(summary, suite="main") == []


def test_publish_rejects_any_changed_reviewed_pdf(tmp_path, report_tools):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    sources, pdfs = {}, {}
    for stem in ("experiment_report", "experiment_report_zh"):
        (candidate / (stem + ".tex")).write_text("reviewed source", encoding="utf-8")
        (candidate / (stem + ".pdf")).write_bytes(b"reviewed PDF bytes")
        sources[stem] = file_hash(candidate / (stem + ".tex"))
        pdfs[stem] = file_hash(candidate / (stem + ".pdf"))
    write_json(candidate / "report.manifest.json", {"assets": {}, "sources": sources, "pdfs": pdfs})
    (candidate / "experiment_report_zh.pdf").write_bytes(b"changed after review")
    with pytest.raises(ValueError, match="changed since review"):
        report_tools.publish_candidate(candidate, "test reviewer")
