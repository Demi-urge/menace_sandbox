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
        resolve_path=lambda p: Path(p),
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

    init_stub = types.ModuleType("init")
    init_stub._repo_path = lambda: tmp_path
    init_stub.init_self_improvement = lambda *a, **k: None
    init_stub.settings = types.SimpleNamespace()
    init_stub._data_dir = lambda: tmp_path
    init_stub._atomic_write = lambda *a, **k: None
    init_stub.get_default_synergy_weights = lambda *a, **k: None
    sys.modules["menace_sandbox.self_improvement.init"] = init_stub
    sys.modules["self_improvement.init"] = init_stub

    root_pkg = types.ModuleType("menace_sandbox")
    root_pkg.__path__ = [str(Path("."))]
    sys.modules["menace_sandbox"] = root_pkg
    si_pkg = types.ModuleType("menace_sandbox.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace_sandbox.self_improvement"] = si_pkg
    sys.modules["self_improvement"] = si_pkg

    class Settings:
        sandbox_data_dir = str(tmp_path)
        snapshot_metrics = {"roi"}
        roi_drop_threshold = -1.0
        entropy_regression_threshold = 1e9
        prompt_penalty_path = "penalties.json"

    sys.modules["sandbox_settings"] = types.SimpleNamespace(
        SandboxSettings=Settings,
        load_sandbox_settings=lambda: Settings(),
    )
    sys.modules["menace_sandbox.sandbox_settings"] = sys.modules["sandbox_settings"]

    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)

    PSM = importlib.import_module(
        "menace_sandbox.self_improvement.prompt_strategy_manager"
    ).PromptStrategyManager

    st.prompt_memory.record_regression("alpha")
    assert st.prompt_memory.load_prompt_penalties().get("alpha") == 1

    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())

    tracker = st.SnapshotTracker()
    manager = PSM()
    module = tmp_path / "mod.py"  # path-ignore
    module.write_text("a = 1\n", encoding="utf-8")

    before = st.Snapshot(1.0, 0.0, 0.5, 0.0, 0.0)
    after = st.Snapshot(2.0, 1.0, 0.4, 0.0, 0.1, prompt="alpha")
    tracker._snaps["before"] = before
    tracker._snaps["after"] = after
    tracker._context["after"] = {"prompt": "alpha", "files": [module]}

    delta = tracker.delta()
    assert delta["regression"] is False
    manager.update("alpha", delta.get("roi", 0.0), True)

    ckpt_base = tmp_path / "checkpoints"
    dirs = list(ckpt_base.iterdir())
    assert len(dirs) == 1
    ckpt_file = dirs[0] / module.name
    assert ckpt_file.exists()

    conf = json.loads((tmp_path / "prompt_strategy_state.json").read_text())
    rec = conf["metrics"]["alpha"]
    assert rec["attempts"] == 1
    assert rec["successes"] == 1
    assert rec["roi"] == delta.get("roi", 0.0)

    assert st.get_best_checkpoint(module) == ckpt_file

    st.prompt_memory.reset_penalty("alpha")
    penalties = st.prompt_memory.load_prompt_penalties()
    assert penalties.get("alpha") == 0


def test_tracker_capture_uses_repo_when_no_files(tmp_path, monkeypatch):
    sys.modules["audit_logger"] = types.SimpleNamespace(log_event=lambda *a, **k: None)
    sys.modules["menace_sandbox.audit_logger"] = sys.modules["audit_logger"]
    sys.modules["snapshot_history_db"] = types.SimpleNamespace(
        log_regression=lambda *a, **k: None,
        record_snapshot=lambda *a, **k: 1,
        record_delta=lambda *a, **k: None,
        resolve_path=lambda p: p,
    )
    sys.modules["menace_sandbox.snapshot_history_db"] = sys.modules["snapshot_history_db"]
    sys.modules["module_index_db"] = types.SimpleNamespace(ModuleIndexDB=None)
    sys.modules["menace_sandbox.module_index_db"] = sys.modules["module_index_db"]
    sys.modules["relevancy_radar"] = types.SimpleNamespace(call_graph_complexity=lambda files: 0)
    sys.modules["menace_sandbox.relevancy_radar"] = sys.modules["relevancy_radar"]
    sys.modules["dynamic_path_router"] = types.SimpleNamespace(
        resolve_path=lambda p: Path(p),
        repo_root=lambda: Path("."),
    )
    init_stub = types.ModuleType("init")
    init_stub._repo_path = lambda: tmp_path
    init_stub.init_self_improvement = lambda *a, **k: None
    init_stub.settings = types.SimpleNamespace()
    init_stub._data_dir = lambda: tmp_path
    init_stub._atomic_write = lambda *a, **k: None
    init_stub.get_default_synergy_weights = lambda *a, **k: None
    sys.modules["menace_sandbox.self_improvement.init"] = init_stub
    sys.modules["self_improvement.init"] = init_stub

    root_pkg = types.ModuleType("menace_sandbox")
    root_pkg.__path__ = [str(Path("."))]
    sys.modules["menace_sandbox"] = root_pkg
    si_pkg = types.ModuleType("menace_sandbox.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace_sandbox.self_improvement"] = si_pkg
    sys.modules["self_improvement"] = si_pkg

    captured: dict[str, list[Path]] = {}

    def fake_collect(files, settings=None):
        flist = [Path(f) for f in files]
        captured["files"] = flist
        return float(len(flist)), float(len(flist))

    metrics_stub = types.SimpleNamespace(
        collect_snapshot_metrics=fake_collect,
        compute_call_graph_complexity=lambda *a, **k: 0.0,
    )
    sys.modules["menace_sandbox.self_improvement.metrics"] = metrics_stub

    sys.modules.pop("menace_sandbox.self_improvement.snapshot_tracker", None)
    st = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")
    monkeypatch.setattr(st, "resolve_path", lambda p: p)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")  # path-ignore
    (repo / "b.py").write_text("b = 2\n", encoding="utf-8")  # path-ignore
    data_dir = tmp_path / "data"

    class Settings:
        sandbox_data_dir = str(data_dir)
        sandbox_repo_path = str(repo)
        snapshot_metrics = {"entropy", "token_diversity"}
        roi_drop_threshold = -1.0
        entropy_regression_threshold = 1e9
        prompt_penalty_path = "penalties.json"

    monkeypatch.setattr(st, "SandboxSettings", lambda: Settings())

    tracker = st.SnapshotTracker()
    snap = tracker.capture("before", {"roi": 0.0, "sandbox_score": 0.0}, repo_path=repo)

    assert snap.entropy == 2.0
    assert snap.token_diversity == 2.0
    assert set(captured["files"]) == {repo / "a.py", repo / "b.py"}  # path-ignore
