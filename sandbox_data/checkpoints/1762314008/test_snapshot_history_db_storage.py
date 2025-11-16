import importlib
import sys
import types
from pathlib import Path
import db_router

from db_router import DBRouter


def test_snapshot_and_delta_persisted(tmp_path, monkeypatch):
    stub = types.SimpleNamespace(
        collect_snapshot_metrics=lambda *a, **k: (0.0, 0.0),
        compute_call_graph_complexity=lambda *a, **k: 0.0,
    )
    sys.modules["menace_sandbox.self_improvement.metrics"] = stub
    sys.modules["dynamic_path_router"] = types.SimpleNamespace(
        resolve_path=lambda p: p,
        resolve_dir=lambda p: Path(p),
        repo_root=lambda: Path("."),
    )
    sys.modules["config_discovery"] = types.ModuleType("config_discovery")
    menace_pkg = types.ModuleType("menace")
    menace_pkg.RAISE_ERRORS = False
    menace_pkg.auto_env_setup = types.SimpleNamespace(ensure_env=lambda *a, **k: None)
    menace_pkg.default_config_manager = types.SimpleNamespace(DefaultConfigManager=object)
    sys.modules["menace"] = menace_pkg
    sys.modules["menace.auto_env_setup"] = menace_pkg.auto_env_setup
    sys.modules["menace.default_config_manager"] = menace_pkg.default_config_manager
    sys.modules["menace_sandbox.audit_logger"] = types.SimpleNamespace(log_event=lambda *a, **k: None)
    db_router.init_db_router = lambda *a, **k: None

    class Settings:
        sandbox_data_dir = str(tmp_path)
        sandbox_score_db = ""
        sandbox_repo_path = str(tmp_path)

    sys.modules["sandbox_settings"] = types.SimpleNamespace(
        SandboxSettings=Settings,
        load_sandbox_settings=lambda: Settings(),
    )
    sys.modules["menace_sandbox.sandbox_settings"] = sys.modules["sandbox_settings"]

    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    sh = importlib.import_module("menace_sandbox.snapshot_history_db")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)
    monkeypatch.setattr(sh, "resolve_path", lambda p: p)
    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())
    monkeypatch.setattr(sh, "SandboxSettings", lambda: Settings())

    st._cycle_id = 0

    tracker = st.SnapshotTracker()
    tracker.capture("pre", {"files": [], "roi": 1.0, "sandbox_score": 0.0, "prompt": "a"})
    tracker.capture("post", {"files": [], "roi": 2.0, "sandbox_score": 0.0, "prompt": "a"})
    delta = tracker.delta()
    assert delta["regression"] is False

    path = tmp_path / "snapshot_history.db"
    router = DBRouter("snapshot_history", str(path), str(path))
    conn = router.get_connection("history")
    assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 2
    rows = conn.execute("SELECT regression FROM deltas").fetchall()
    assert rows == [(0,)]
    assert sh.last_successful_cycle() == 1
