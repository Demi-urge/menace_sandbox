import importlib.util
import json
import sys
import types
from types import SimpleNamespace

from dynamic_path_router import resolve_path


def _load_scoring(monkeypatch):
    stub_logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    logging_utils = types.ModuleType("logging_utils")
    logging_utils.get_logger = lambda name: stub_logger
    logging_utils.log_record = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "logging_utils", logging_utils)

    results_logger = types.ModuleType("sandbox_results_logger")
    results_logger.record_run = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_results_logger", results_logger)

    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = [str(resolve_path("sandbox_runner"))]
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)

    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.scoring", resolve_path("sandbox_runner/scoring.py"),
    )
    scoring = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(scoring)
    monkeypatch.setitem(sys.modules, "sandbox_runner.scoring", scoring)
    return scoring


def _reset_logs(scoring, tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(scoring, "_RUN_LOG", tmp_path / "run_metrics.jsonl")
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", tmp_path / "run_summary.json")


def test_record_run_entropy_delta_positive(tmp_path, monkeypatch):
    scoring = _load_scoring(monkeypatch)
    _reset_logs(scoring, tmp_path, monkeypatch)
    scoring.record_run(
        SimpleNamespace(success=True, duration=1.0, failure=None),
        {"entropy_delta": 0.5},
    )
    data = json.loads(scoring._RUN_LOG.read_text().splitlines()[0])
    assert data["entropy_delta"] == 0.5
    summary = json.loads(scoring._SUMMARY_FILE.read_text())
    assert summary["entropy_total"] == 0.5


def test_record_run_entropy_delta_negative(tmp_path, monkeypatch):
    scoring = _load_scoring(monkeypatch)
    _reset_logs(scoring, tmp_path, monkeypatch)
    scoring.record_run(
        SimpleNamespace(success=True, duration=0.1, failure=None),
        {"entropy_delta": -0.25},
    )
    data = json.loads(scoring._RUN_LOG.read_text().splitlines()[0])
    assert data["entropy_delta"] == -0.25
    summary = json.loads(scoring._SUMMARY_FILE.read_text())
    assert summary["entropy_total"] == -0.25


def test_record_run_entropy_delta_zero(tmp_path, monkeypatch):
    scoring = _load_scoring(monkeypatch)
    _reset_logs(scoring, tmp_path, monkeypatch)
    scoring.record_run(
        SimpleNamespace(success=True, duration=0.0, failure=None),
        {"entropy_delta": 0.0},
    )
    data = json.loads(scoring._RUN_LOG.read_text().splitlines()[0])
    assert data["entropy_delta"] == 0.0
    summary = json.loads(scoring._SUMMARY_FILE.read_text())
    assert summary.get("entropy_total", 0.0) == 0.0
