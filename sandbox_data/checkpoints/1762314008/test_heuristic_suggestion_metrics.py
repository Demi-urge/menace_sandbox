import types
import sqlite3
import os
import sys
import types
import sqlite3

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _stub(name, **attrs):
    mod = types.SimpleNamespace(**attrs)
    sys.modules.setdefault(name, mod)
    return mod


_stub("memory_logging", log_with_tags=lambda *a, **k: None)
_stub("memory_aware_gpt_client", ask_with_memory=lambda *a, **k: "")
_stub("vector_service", Retriever=object, FallbackResult=object, EmbeddableDBMixin=object)
_stub("foresight_tracker", ForesightTracker=object)
_stub("db_router", DBRouter=object, GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None)
_stub("analytics", adaptive_roi_model=None)
_stub("adaptive_roi_predictor", load_training_data=lambda *a, **k: None)
_stub(
    "metrics_exporter",
    orphan_modules_reintroduced_total=None,
    orphan_modules_tested_total=None,
    orphan_modules_failed_total=None,
    orphan_modules_redundant_total=None,
    orphan_modules_legacy_total=None,
    orphan_modules_reclassified_total=None,
)
_stub(
    "relevancy_radar",
    RelevancyRadar=object,
    track_usage=lambda *a, **k: None,
    evaluate_final_contribution=lambda *a, **k: None,
    record_output_impact=lambda *a, **k: None,
    radar=types.SimpleNamespace(track=lambda fn: fn),
)
_stub("sandbox_runner.resource_tuner", ResourceTuner=object)
_stub(
    "sandbox_runner.orphan_discovery",
    append_orphan_cache=lambda *a, **k: None,
    append_orphan_classifications=lambda *a, **k: None,
    prune_orphan_cache=lambda *a, **k: None,
    load_orphan_cache=lambda *a, **k: None,
    load_orphan_traces=lambda *a, **k: None,
    append_orphan_traces=lambda *a, **k: None,
    discover_recursive_orphans=lambda *a, **k: None,
)

from menace.sandbox_runner import cycle


def _ctx(tmp_path, **kwargs):
    base = {"repo": tmp_path, "suggestion_db": None, "failure_db_path": str(tmp_path / "failures.db")}
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _write_module(tmp_path, name="mod.py"):  # path-ignore
    p = tmp_path / name
    p.write_text("def foo():\n    return 1\n")
    return p


def test_complexity_metric_influences_suggestion(monkeypatch, tmp_path):
    _write_module(tmp_path)
    ctx = _ctx(tmp_path)

    monkeypatch.setattr(cycle, "mi_visit", lambda code, _: 40.0)
    low = cycle._heuristic_suggestion(ctx, "mod.py")  # path-ignore
    assert "maintainability" in low

    monkeypatch.setattr(cycle, "mi_visit", lambda code, _: 90.0)
    high = cycle._heuristic_suggestion(ctx, "mod.py")  # path-ignore
    assert "consider simplifying" in high
    assert low != high


def test_failure_data_influences_suggestion(monkeypatch, tmp_path):
    _write_module(tmp_path)
    db_path = tmp_path / "failures.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE failures(model_id TEXT, cause TEXT, features TEXT, demographics TEXT, profitability REAL, retention REAL, cac REAL, roi REAL, ts TEXT)"
    )
    conn.execute(
        "INSERT INTO failures VALUES(?,?,?,?,?,?,?,?,datetime('now'))",
        ("mod.py", "", "", "", 0.0, 0.0, 0.0, 0.0),  # path-ignore
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(cycle, "mi_visit", lambda code, _: 90.0)
    ctx = _ctx(tmp_path)
    suggestion = cycle._heuristic_suggestion(ctx, "mod.py")  # path-ignore
    assert "recent failures" in suggestion
