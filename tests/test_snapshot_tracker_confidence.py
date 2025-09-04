import importlib
import json
import sys
import types
from pathlib import Path
import db_router

def test_confidence_and_best_checkpoint(tmp_path, monkeypatch):
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

    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)
    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())

    tracker = st.SnapshotTracker()
    module = tmp_path / "mod.py"
    module.write_text("a = 1\n", encoding="utf-8")

    before = st.Snapshot(1.0, 0.0, 0.5, 0.0, 0.0)
    after = st.Snapshot(2.0, 1.0, 0.4, 0.0, 0.1, prompt="alpha")
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    tracker._context["after"] = {"prompt": "alpha", "files": [module]}

    delta = tracker.delta()
    assert delta["regression"] is False

    ckpt_base = tmp_path / "checkpoints"
    dirs = list(ckpt_base.iterdir())
    assert len(dirs) == 1
    ckpt_file = dirs[0] / module.name
    assert ckpt_file.exists()

    conf = json.loads((tmp_path / "strategy_confidence.json").read_text())
    assert conf["alpha"] == 1

    assert st.get_best_checkpoint(module) == ckpt_file
