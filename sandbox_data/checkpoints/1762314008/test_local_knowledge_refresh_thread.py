import importlib.util
import sys
import types
import threading
import logging
from pathlib import Path

from dynamic_path_router import resolve_path

ROOT = Path(__file__).resolve().parents[1]


def _make_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load_run_autonomous(monkeypatch):
    class SandboxSettings:
        def __init__(self):
            self.local_knowledge_refresh_interval = 0.01
            self.sandbox_central_logging = False
            self.menace_mode = "test"
            self.database_url = "sqlite://"
            self.menace_local_db_path = None
            self.menace_shared_db_path = None
            self.sandbox_repo_path = str(ROOT)

    cli_mod = _make_module(
        "sandbox_runner.cli",
        full_autonomous_run=lambda args: None,
        _diminishing_modules=lambda *a, **k: set(),
        adaptive_synergy_convergence=lambda *a, **k: (True, 0.0, {}),
    )

    modules = {
        "db_router": _make_module("db_router", init_db_router=lambda *a, **k: object()),
        "sandbox_settings": _make_module("sandbox_settings", SandboxSettings=SandboxSettings),
        "sandbox_runner.bootstrap": _make_module(
            "sandbox_runner.bootstrap",
            bootstrap_environment=lambda s, v: s,
            _verify_required_dependencies=lambda: None,
        ),
        "gpt_memory": _make_module("gpt_memory", GPTMemoryManager=object),
        "memory_maintenance": _make_module(
            "memory_maintenance",
            MemoryMaintenance=object,
            _load_retention_rules=lambda: None,
        ),
        "gpt_knowledge_service": _make_module(
            "gpt_knowledge_service", GPTKnowledgeService=object
        ),
        "local_knowledge_module": _make_module(
            "local_knowledge_module",
            LocalKnowledgeModule=object,
            init_local_knowledge=lambda: None,
        ),
        "foresight_tracker": _make_module("foresight_tracker", ForesightTracker=object),
        "threshold_logger": _make_module("threshold_logger", ThresholdLogger=object),
        "forecast_logger": _make_module("forecast_logger", ForecastLogger=object),
        "preset_logger": _make_module("preset_logger", PresetLogger=object),
        "metrics_exporter": _make_module(
            "metrics_exporter",
            start_metrics_server=lambda *a, **k: None,
            roi_threshold_gauge=None,
            synergy_threshold_gauge=None,
            roi_forecast_gauge=None,
            synergy_forecast_gauge=None,
            synergy_adaptation_actions_total=None,
        ),
        "relevancy_radar_service": _make_module(
            "relevancy_radar_service", RelevancyRadarService=object
        ),
        "synergy_monitor": _make_module(
            "synergy_monitor", ExporterMonitor=object, AutoTrainerMonitor=object
        ),
        "sandbox_recovery_manager": _make_module(
            "sandbox_recovery_manager", SandboxRecoveryManager=object
        ),
        "menace": _make_module("menace", __path__=[]),
        "menace.audit_trail": _make_module("menace.audit_trail", AuditTrail=object),
        "menace.environment_generator": _make_module(
            "menace.environment_generator", generate_presets=lambda n=None: []
        ),
        "menace.roi_tracker": _make_module("menace.roi_tracker", ROITracker=object),
        "menace.synergy_exporter": _make_module(
            "menace.synergy_exporter", SynergyExporter=object
        ),
        "menace.synergy_history_db": _make_module(
            "menace.synergy_history_db",
            migrate_json_to_db=lambda *a, **k: None,
            insert_entry=lambda *a, **k: None,
            connect_locked=lambda *a, **k: None,
        ),
        "sandbox_runner.cli": cli_mod,
        "sandbox_runner": _make_module(
            "sandbox_runner", _sandbox_main=lambda p, a: None, cli=cli_mod
        ),
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(ROOT))

    spec = importlib.util.spec_from_file_location(
        "run_autonomous",
        resolve_path("run_autonomous.py"),  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    spec.loader.exec_module(mod)
    return mod


def test_refresh_thread_exception_and_cleanup(monkeypatch, caplog):
    mod = _load_run_autonomous(monkeypatch)

    class DummyConn:
        def commit(self):
            pass

    class DummyMemory:
        def __init__(self):
            self.conn = DummyConn()

    class DummyLKM:
        def __init__(self):
            self.memory = DummyMemory()
            self.count = 0
            self.ready = threading.Event()

        def refresh(self):
            self.count += 1
            if self.count == 1:
                raise RuntimeError("boom")
            self.ready.set()

    dummy = DummyLKM()
    mod.LOCAL_KNOWLEDGE_MODULE = dummy
    mod.LOCAL_KNOWLEDGE_REFRESH_INTERVAL = 0.01

    cleanup = []
    with caplog.at_level(logging.ERROR, logger=mod.logger.name):
        mod._start_local_knowledge_refresh(cleanup)
        thread = mod._LKM_REFRESH_THREAD
        assert thread is not None and thread.is_alive()
        assert dummy.ready.wait(1.0)

    cleanup[0]()
    assert mod._LKM_REFRESH_THREAD is None
    assert not mod._LKM_REFRESH_STOP.is_set()
    assert not thread.is_alive()

    records = [
        r for r in caplog.records if "failed to refresh local knowledge module" in r.message
    ]
    assert records and getattr(records[0], "run") == 1
