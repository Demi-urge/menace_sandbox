import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from dynamic_path_router import resolve_path

sys.path.append(str(resolve_path("")))

from sandbox_settings import SandboxSettings  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _setup_base_packages():
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg
    # minimal logging utils
    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    logging_utils = types.SimpleNamespace(
        get_logger=lambda name: logger,
        log_record=lambda **k: k,
        setup_logging=lambda: None,
    )
    sys.modules["menace.logging_utils"] = logging_utils


# ---------------------------------------------------------------------------

def test_init_creates_and_clamps_synergy_weights(tmp_path, monkeypatch):
    _setup_base_packages()
    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    meta_stub = types.ModuleType("menace.self_improvement.meta_planning")
    meta_stub.reload_settings = lambda cfg: None
    sys.modules["menace.self_improvement.meta_planning"] = meta_stub

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_repo_path = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False

    # pre-populate with out-of-range weights
    bad = {"_doc": "", "roi": 20, "efficiency": -5}
    (tmp_path / "synergy_weights.json").write_text(json.dumps(bad))

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)

    init_module.init_self_improvement()

    synergy_file = Path(settings.synergy_weight_file)
    assert synergy_file.exists()
    data = json.loads(synergy_file.read_text())
    assert data["roi"] == 10.0
    assert data["efficiency"] == 0.0
    # defaults applied for missing keys and clamped into range
    for key in init_module.get_default_synergy_weights():
        assert 0.0 <= data[key] <= 10.0


# ---------------------------------------------------------------------------

def test_start_self_improvement_cycle_thread(tmp_path, monkeypatch, in_memory_dbs):
    _setup_base_packages()
    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    InMemoryROI, InMemoryStability = in_memory_dbs
    sys.modules["menace.lock_utils"] = types.SimpleNamespace(
        SandboxLock=lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False
        ),
        Timeout=Exception,
        LOCK_TIMEOUT=1,
    )
    sys.modules["menace.unified_event_bus"] = types.SimpleNamespace(UnifiedEventBus=None)
    sys.modules["menace.meta_workflow_planner"] = types.SimpleNamespace(
        MetaWorkflowPlanner=None
    )

    import sandbox_settings as sandbox_settings_module
    sys.modules["menace.sandbox_settings"] = sandbox_settings_module

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")
    )
    meta_planning = _load_module(
        "menace.self_improvement.meta_planning",
        resolve_path("self_improvement/meta_planning.py"),
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.sandbox_repo_path = str(tmp_path)
    settings.sandbox_central_logging = False
    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    init_module.init_self_improvement()

    calls = {"count": 0}

    async def fake_cycle(workflows, *, interval, event_bus=None):
        calls["count"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr(meta_planning, "self_improvement_cycle", fake_cycle)

    thread = meta_planning.start_self_improvement_cycle(
        {"w": lambda: None},
        event_bus=types.SimpleNamespace(publish=lambda *a, **k: None),
        interval=0,
    )

    thread.start()
    thread.join(timeout=1)
    assert calls["count"] >= 1
    assert InMemoryROI.instances[0].records
    assert InMemoryStability.instances[0].data
