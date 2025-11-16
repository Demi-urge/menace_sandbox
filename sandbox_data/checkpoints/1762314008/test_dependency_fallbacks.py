import importlib
import logging
import sys
import types
from pathlib import Path


def _reload(name, missing):
    for m in missing:
        sys.modules.pop(m, None)
        sys.modules[m] = None
    sys.modules.pop(name, None)
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return importlib.import_module(name)

def test_data_bot_missing_dependencies(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        mod = _reload(
            "menace_sandbox.data_bot", ["pandas", "psutil", "prometheus_client"]
        )
    assert mod.pd is None and mod.psutil is None
    assert any("pandas" in rec.message for rec in caplog.records)
    assert any("psutil" in rec.message for rec in caplog.records)
    mdb = mod.MetricsDB.__new__(mod.MetricsDB)
    mdb.router = types.SimpleNamespace(menace_id="x")

    class _Cur:
        description = [(c,) for c in [
            "bot","cpu","memory","response_time","disk_io","net_io","errors",
            "revenue","expense","security_score","safety_rating","adaptability",
            "antifragility","shannon_entropy","efficiency","flexibility","gpu_usage",
            "projected_lucrativity","profitability","patch_complexity","patch_entropy",
            "energy_consumption","resilience","network_latency","throughput","risk_index",
            "maintainability","code_quality","ts"
        ]]

        def fetchall(self):
            return []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_):
            return _Cur()

    mdb._connect = lambda: _Conn()
    result = mdb.fetch(limit=1)
    assert result == []

def test_performance_bot_fallback(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        mod = _reload("menace_sandbox.performance_assessment_bot", ["pandas"])
    assert any("pandas" in rec.message for rec in caplog.records)

    class DummyDB:
        def fetch(self, limit):
            return [{
                "bot": "b1", "cpu": 10.0, "memory": 10.0,
                "response_time": 1.0, "errors": 0,
            }]

    bot = mod.PerformanceAssessmentBot(metrics_db=DummyDB())
    score = bot.self_assess("b1")
    assert isinstance(score, float)


def test_adaptive_roi_predictor_top_level_import(monkeypatch):
    project_root = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(sys, "path", [str(project_root)])

    stored_modules = {}
    targets = {
        "adaptive_roi_predictor",
        "logging_utils",
        "adaptive_roi_dataset",
        "roi_tracker",
        "evaluation_history_db",
        "evolution_history_db",
        "truth_adapter",
    }
    for name in list(sys.modules):
        if name == "menace_sandbox" or name.startswith("menace_sandbox."):
            stored_modules[name] = sys.modules.pop(name)
    for name in targets:
        if name in sys.modules:
            stored_modules[name] = sys.modules.pop(name)

    try:
        mod = importlib.import_module("adaptive_roi_predictor")
    finally:
        for name in targets:
            sys.modules.pop(name, None)
        sys.modules.update(stored_modules)

    assert hasattr(mod, "AdaptiveROIPredictor")
    assert hasattr(mod, "build_dataset")
