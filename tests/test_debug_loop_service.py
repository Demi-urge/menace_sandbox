import importlib.util
import sys
import types
import threading
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())

pkg = types.ModuleType("menace")
pkg.__path__ = [str(TMP), str(ROOT)]
sys.modules["menace"] = pkg
sys.modules.pop("vector_service", None)
vs_pkg = types.ModuleType("vector_service")
vs_pkg.ContextBuilder = type(
    "ContextBuilder", (), {"__init__": lambda self, *a, **k: None}
)
sys.modules["vector_service"] = vs_pkg
sys.modules["vector_service.context_builder"] = types.SimpleNamespace(
    ContextBuilder=vs_pkg.ContextBuilder
)

# write stub modules to temporary package path
(TMP / "telemetry_feedback.py").write_text(  # path-ignore
    """class TelemetryFeedback:\n    def __init__(self, logger=None, manager=None):\n        self.interval=0\n        self.started=False\n        self.stopped=False\n    def start(self):\n        self.started=True\n    def stop(self):\n        self.stopped=True\n"""
)
(TMP / "error_logger.py").write_text(  # path-ignore
    "class ErrorLogger:\n    def __init__(self, **kwargs):\n        pass\n"
)
(TMP / "self_coding_engine.py").write_text(  # path-ignore
    "class SelfCodingEngine:\n    def __init__(self, *a, **k):\n        pass\n"
)
(TMP / "code_database.py").write_text("class CodeDB:\n    pass\n")  # path-ignore
(TMP / "menace_memory_manager.py").write_text(  # path-ignore
    "class MenaceMemoryManager:\n    pass\n"
)
(TMP / "knowledge_graph.py").write_text(  # path-ignore
    """class KnowledgeGraph:\n    def __init__(self):\n        self.traces=[]\n    def add_crash_trace(self, name, trace):\n        self.traces.append((name, trace))\n"""
)

(TMP / "self_coding_manager.py").write_text(
    "class SelfCodingManager:\n"
    "    def __init__(self, engine, pipeline, **k):\n"
    "        self.engine = engine\n"
    "        self.pipeline = pipeline\n"
    "\n"
    "def internalize_coding_bot(name, engine, pipeline, **kwargs):\n"
    "    return SelfCodingManager(engine, pipeline, **kwargs)\n"
)
(TMP / "model_automation_pipeline.py").write_text(
    "class ModelAutomationPipeline:\n    def __init__(self, **k):\n        pass\n"
)
(TMP / "unified_event_bus.py").write_text(
    "class UnifiedEventBus:\n    def __init__(self, *a, **k):\n        pass\n"
)
(TMP / "bot_registry.py").write_text(
    "class BotRegistry:\n    def __init__(self, *a, **k):\n        pass\n"
)
(TMP / "data_bot.py").write_text(
    "class DataBot:\n    pass\n\n"
    "def persist_sc_thresholds(name, **kwargs):\n    return kwargs\n"
)

spec = importlib.util.spec_from_file_location(
    "menace.debug_loop_service",
    ROOT / "debug_loop_service.py",  # path-ignore
    submodule_search_locations=[str(TMP), str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
sys.modules["menace.debug_loop_service"] = mod
spec.loader.exec_module(mod)
mod.prepare_pipeline_for_bootstrap = lambda **kwargs: (
    types.SimpleNamespace(manager=types.SimpleNamespace()),
    lambda __: None,
)


def test_collect_crash_traces(tmp_path):
    log = tmp_path / "err.log"
    log.write_text("Traceback\nboom", encoding="utf-8")
    svc = mod.DebugLoopService(context_builder=vs_pkg.ContextBuilder())
    svc.collect_crash_traces(str(tmp_path))
    assert svc.graph.traces and svc.graph.traces[0][0] == "err"


def test_collect_crash_traces_ignores(tmp_path):
    log = tmp_path / "info.log"
    log.write_text("no error", encoding="utf-8")
    svc = mod.DebugLoopService(context_builder=vs_pkg.ContextBuilder())
    svc.collect_crash_traces(str(tmp_path))
    assert not svc.graph.traces


def test_run_continuous_logs_errors(monkeypatch, caplog):
    stop = threading.Event()
    svc = mod.DebugLoopService(context_builder=vs_pkg.ContextBuilder())

    def fail_collect(path):
        stop.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "collect_crash_traces", fail_collect)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    recorded = {}

    class DummyThread:
        def __init__(self, target, args=(), daemon=True):
            recorded["target"] = target
            recorded["args"] = args
        def start(self):
            recorded["started"] = True
            recorded["target"](*recorded["args"])

    monkeypatch.setattr(mod.threading, "Thread", DummyThread)
    caplog.set_level("ERROR")
    svc.run_continuous(interval=0.0, stop_event=stop)
    assert svc.failure_count == 1
    assert "collect_crash_traces failed" in caplog.text


def test_uses_broker_advertised_pipeline(monkeypatch):
    class DummyModelAutomationPipeline:
        def __init__(self) -> None:
            self.context_builder = None
            self.manager = types.SimpleNamespace()
            self._bot_attribute_order = []

    pipeline = DummyModelAutomationPipeline()
    mod.DebugLoopService._BOOTSTRAP_PIPELINE = None
    prepare_called = False

    def fake_prepare_pipeline_for_bootstrap(**_: object):
        nonlocal prepare_called
        prepare_called = True
        return object(), lambda __: None

    dummy_broker = types.SimpleNamespace(
        resolve=lambda: (pipeline, pipeline.manager),
        advertise=lambda **kwargs: None,
        clear=lambda: None,
    )

    monkeypatch.setattr(mod, "_bootstrap_dependency_broker", lambda: dummy_broker)
    monkeypatch.setattr(mod, "get_active_bootstrap_pipeline", lambda: (pipeline, pipeline.manager))
    monkeypatch.setattr(mod, "prepare_pipeline_for_bootstrap", fake_prepare_pipeline_for_bootstrap)

    captured: dict[str, object] = {}

    def fake_internalize_coding_bot(name, engine, pipeline_arg, **kwargs):  # type: ignore[override]
        captured.update({"name": name, "engine": engine, "pipeline": pipeline_arg, "kwargs": kwargs})

        manager = mod.SelfCodingManager(engine, pipeline_arg, **kwargs)
        captured["manager"] = manager
        return manager

    monkeypatch.setattr(mod, "internalize_coding_bot", fake_internalize_coding_bot)

    class DummyFeedback:
        def __init__(self, logger=None, manager=None) -> None:
            self.logger = logger
            self.manager = manager
            self.interval = 0

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(mod, "TelemetryFeedback", DummyFeedback)

    svc = mod.DebugLoopService(context_builder=vs_pkg.ContextBuilder())

    assert captured.get("pipeline") is pipeline
    assert svc.feedback.manager is captured.get("manager")
    assert prepare_called is False
