import importlib
import sys
import types
from pathlib import Path
import db_router

from db_router import DBRouter


def test_delta_logs_regression(tmp_path, monkeypatch):
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

    sys.modules["sandbox_settings"] = types.SimpleNamespace(
        SandboxSettings=Settings,
        load_sandbox_settings=lambda: Settings(),
    )
    sys.modules["menace_sandbox.sandbox_settings"] = sys.modules["sandbox_settings"]

    pkg = types.ModuleType("menace_sandbox.self_improvement")
    pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace_sandbox.self_improvement"] = pkg
    sys.modules["self_improvement"] = pkg
    
    from dynamic_path_router import resolve_path
    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    sh = importlib.import_module("menace_sandbox.snapshot_history_db")
    pm = importlib.import_module("menace_sandbox.self_improvement.prompt_memory")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)
    monkeypatch.setattr(sh, "resolve_path", lambda p: p)
    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())
    monkeypatch.setattr(sh, "SandboxSettings", lambda: Settings())
    base = Path(resolve_path(str(tmp_path)))

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(st, "audit_log_event", lambda name, payload: events.append((name, payload)))

    tracker = st.SnapshotTracker()
    before = st.Snapshot(1.0, 0.0, 0.1, 0.0, 0.0)
    diff_file = Path(resolve_path(str(base / "diff.patch")))
    diff_file.write_text("diff-data", encoding="utf-8")
    after = st.Snapshot(0.5, 0.0, 0.2, 0.0, 0.0, prompt="p", diff=str(diff_file))
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    tracker._context["after"] = {"prompt": "p", "diff": str(diff_file)}

    delta = tracker.delta()
    assert delta["regression"] is True

    router = DBRouter(
        "snapshot_history",
        str(base / "snapshot_history.db"),
        str(base / "snapshot_history.db"),
    )
    conn = router.get_connection("regressions")
    rows = conn.execute(
        "SELECT prompt, diff, roi_delta, entropy_delta FROM regressions"
    ).fetchall()
    assert rows == [("p", "diff-data", -0.5, 0.1)]
    assert events and events[0][0] == "snapshot_regression"
    assert pm.load_prompt_penalties()["p"] == 1
