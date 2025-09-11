import os
import types
import sys

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault(
    "menace.self_coding_engine", types.SimpleNamespace(SelfCodingEngine=object)
)
sys.modules.setdefault(
    "menace.data_bot", types.SimpleNamespace(MetricsDB=object, DataBot=object)
)
sys.modules.setdefault(
    "vector_service",
    types.SimpleNamespace(
        EmbeddableDBMixin=type(
            "EmbeddableDBMixin", (), {"__init__": lambda self, *a, **k: None}
        ),
        CognitionLayer=object,
        Retriever=object,
        FallbackResult=object,
        ContextBuilder=object,
        PatchLogger=object,
        EmbeddingBackfill=object,
        SharedVectorService=object,
        VectorServiceError=Exception,
        RateLimitError=Exception,
        MalformedPromptError=Exception,
        ErrorResult=Exception,
    ),
)
sys.modules.setdefault(
    "vector_service.text_preprocessor",
    types.SimpleNamespace(
        get_config=lambda: None,
        PreprocessingConfig=object,
        generalise=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "vector_service.context_builder",
    types.SimpleNamespace(ContextBuilder=object),
)

# Stub error_bot to avoid heavy imports
eb_module = types.ModuleType("menace.error_bot")
class ErrorDB:
    def __init__(self, path):
        self.records = []
        self.router = object()
        self.conn = types.SimpleNamespace(execute=lambda *a, **k: types.SimpleNamespace(fetchall=lambda: []))

    def add_telemetry(self, event, source_menace_id=None):
        self.records.append(event)

    def top_error_module(self, **_):
        if not self.records:
            return None
        ev = self.records[0]
        return (ev.error_type, ev.module, {}, len(self.records), "bot")

    def _menace_id(self, source=None):
        return "bot"

eb_module.ErrorDB = ErrorDB
sys.modules["menace.error_bot"] = eb_module

import menace.error_bot as eb  # noqa: E402
import menace.error_logger as elog  # noqa: E402
import menace.telemetry_feedback as tf  # noqa: E402
import menace.dynamic_path_router as dpr  # noqa: E402
from pathlib import Path  # noqa: E402
import pytest  # noqa: E402


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str, **_: object):
        self.calls.append((path, desc))
        return 1, False, 0.0


class DummyManager:
    def __init__(self, engine, registry=None, data_bot=None):
        self.engine = engine
        self.bot_registry = registry
        self.data_bot = data_bot
        self.calls = []

    def run_patch(self, path, desc, **_):
        self.calls.append((path, desc))
        return self.engine.apply_patch(path, desc)


class DummyGraph:
    def __init__(self):
        self.events = []
        self.updated = None

    def add_telemetry_event(self, *a, **k):
        self.events.append((a, k))

    def update_error_stats(self, db):
        self.updated = db


class DummyRegistry:
    def __init__(self):
        self.names = []

    def register_bot(self, name):
        self.names.append(name)


class _DummyMetricsDB:
    def __init__(self):
        self.records = []

    def log_eval(self, name, metric, value):
        self.records.append((name, metric, value))


class DummyDataBot:
    def __init__(self):
        self.db = _DummyMetricsDB()

    def roi(self, _name):
        return 0.0


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    dpr._PROJECT_ROOT = None
    dpr._PATH_CACHE.clear()
    db = eb.ErrorDB(tmp_path / "e.db")
    db.conn.execute(
        "CREATE TABLE IF NOT EXISTS error_stats("  # noqa: E501
        "category TEXT, module TEXT, count INTEGER, PRIMARY KEY(category, module)"
        ")"
    )
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder())
    engine = DummyEngine()
    mod = tmp_path / "bot.py"  # path-ignore
    mod.write_text("def x():\n    pass\n")
    return db, logger, engine, mod


@pytest.mark.parametrize("scope, src", [("local", None)])
def test_feedback_triggers_patch(tmp_path, monkeypatch, scope, src):
    db, logger, engine, mod = _setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(3):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
                module="bot",
                module_counts={"bot": 1},
            ),
            source_menace_id=src,
        )
    monkeypatch.setattr(tf, "resolve_path", lambda _p: mod.resolve())
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    manager = DummyManager(engine, registry, data_bot)
    fb = tf.TelemetryFeedback(
        logger,
        manager,
        threshold=3,
    )
    monkeypatch.setattr(fb, "_mark_attempt", lambda *a, **k: None)
    monkeypatch.setattr(
        fb.logger.db.conn,
        "execute",
        lambda *a, **k: types.SimpleNamespace(fetchall=lambda: []),
    )
    fb._run_cycle(scope=scope)
    assert manager.calls and manager.calls[0][0] == mod.resolve()
    assert "TelemetryFeedback" in registry.names
    assert data_bot.db.records


def test_feedback_threshold(tmp_path, monkeypatch):
    db, logger, engine, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    trace = "Traceback...\nKeyError: boom"
    for _ in range(2):
        db.add_telemetry(
            elog.TelemetryEvent(
                error_type=elog.ErrorType.RUNTIME_FAULT,
                stack_trace=trace,
                root_module="bot",
                module="bot",
                module_counts={"bot": 1},
            )
        )
    manager = DummyManager(engine)
    fb = tf.TelemetryFeedback(logger, manager, threshold=3)
    monkeypatch.setattr(fb, "_mark_attempt", lambda *a, **k: None)
    monkeypatch.setattr(
        fb.logger.db.conn,
        "execute",
        lambda *a, **k: types.SimpleNamespace(fetchall=lambda: []),
    )
    fb._run_cycle()
    assert not manager.calls
