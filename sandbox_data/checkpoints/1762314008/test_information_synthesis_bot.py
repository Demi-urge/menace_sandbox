import importlib.util
import os
import pathlib
import sys
import types

import pytest

pytest.importorskip("pandas")
import pandas as pd

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

STUB_DIR = pathlib.Path(__file__).resolve().parent / "stubs"
if STUB_DIR.exists() and str(STUB_DIR) not in sys.path:
    sys.path.insert(0, str(STUB_DIR))


def _install_stub_module(
    name: str, attributes: dict[str, object], *, aliases: tuple[str, ...] = ()
) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr, value in attributes.items():
        setattr(module, attr, value)
    sys.modules[name] = module
    for alias in aliases:
        sys.modules[alias] = module
    return module


ROOT = pathlib.Path(__file__).resolve().parents[1]

menace_pkg = sys.modules.get("menace")
if menace_pkg is None:
    menace_pkg = types.ModuleType("menace")
    sys.modules["menace"] = menace_pkg
menace_pkg.__path__ = [str(ROOT)]

_install_stub_module(
    "menace_sandbox.bot_registry",
    {"BotRegistry": type("BotRegistry", (), {"__init__": lambda self, *a, **k: None})},
    aliases=("menace.bot_registry",),
)
_install_stub_module(
    "menace_sandbox.data_bot",
    {"DataBot": type("DataBot", (), {"__init__": lambda self, *a, **k: None})},
    aliases=("menace.data_bot",),
)
_install_stub_module(
    "menace_sandbox.coding_bot_interface",
    {"self_coding_managed": lambda *_a, **_k: (lambda cls: cls)},
    aliases=("menace.coding_bot_interface",),
)
_install_stub_module(
    "menace_sandbox.research_aggregator_bot",
    {
        "ResearchAggregatorBot": type("ResearchAggregatorBot", (), {"__init__": lambda self, *a, **k: None}),
        "ResearchItem": type("ResearchItem", (), {}),
    },
    aliases=("menace.research_aggregator_bot",),
)
_install_stub_module(
    "menace_sandbox.task_handoff_bot",
    {
        "WorkflowDB": type(
            "WorkflowDB",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "fetch": lambda self: [],
            },
        )
    },
    aliases=("menace.task_handoff_bot",),
)
_install_stub_module(
    "menace_sandbox.unified_event_bus",
    {"UnifiedEventBus": type("UnifiedEventBus", (), {})},
    aliases=("menace.unified_event_bus",),
)
_install_stub_module(
    "menace_sandbox.relevancy_radar",
    {
        "tracked_import": lambda name, globals=None, locals=None, fromlist=(), level=0: __import__(
            name, globals, locals, fromlist, level
        )
    },
    aliases=("menace.relevancy_radar",),
)

vector_context_module = types.ModuleType("vector_service.context_builder")
vector_context_module.ContextBuilder = type(
    "ContextBuilder",
    (),
    {"refresh_db_weights": lambda self: None},
)
sys.modules.setdefault("vector_service.context_builder", vector_context_module)

if "sqlalchemy" not in sys.modules:
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")

    class _DummyMeta:
        def reflect(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub
            pass

    sqlalchemy_mod.create_engine = lambda *a, **k: object()
    sqlalchemy_mod.MetaData = lambda *a, **k: _DummyMeta()
    sqlalchemy_mod.Table = lambda *a, **k: object()
    sqlalchemy_mod.select = lambda *a, **k: object()
    sqlalchemy_mod.engine = engine_mod
    sys.modules["sqlalchemy"] = sqlalchemy_mod
    sys.modules["sqlalchemy.engine"] = engine_mod

import menace_sandbox.information_synthesis_bot as isb

try:
    from marshmallow import Schema, fields  # type: ignore
    HAS_MM = True
except Exception:  # pragma: no cover - marshmallow not available
    Schema = isb.Schema
    fields = isb.fields
    HAS_MM = False

import menace_sandbox.task_handoff_bot as thb


class DummyAggregator:
    def __init__(self) -> None:
        self.info_db = None

    def process(self, topic: str, energy: int = 1):  # pragma: no cover - stub
        return []


class RecordSchema(Schema):
    id = fields.Int(required=True)
    name = fields.Str(required=True)
    value = fields.Str(required=True)


builder = vector_context_module.ContextBuilder()


def test_analyse_detects_issues(tmp_path):
    df = pd.DataFrame(
        {"id": [1, 2, 3], "name": ["Alpha", "Alpha", "Beta"], "value": ["1", "2", None]}
    )
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    reqs = bot.analyse(df, RecordSchema(), "records")
    reasons = [r.reason for r in reqs]
    assert "duplicate" in reasons or "invalid" in reasons


def test_dispatch_requests(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"id": [1, "x"], "name": ["Alpha", "Beta"], "value": ["1", None]},
        dtype=object,
    )
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    sent: list[tuple[str, dict[str, object] | None]] = []
    monkeypatch.setattr(
        bot.app,
        "send_task",
        lambda name, kwargs=None: sent.append((name, kwargs)),
    )
    reqs = bot.analyse(df, RecordSchema(), "records")
    bot.dispatch_requests(reqs)
    assert sent


def test_synthesise_creates_tasks(monkeypatch, tmp_path):
    df = pd.DataFrame({"id": [1], "name": ["Alpha"], "value": ["1"]})
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    monkeypatch.setattr(bot.app, "send_task", lambda *a, **k: None)
    tasks_sent = []
    monkeypatch.setattr(isb, "send_to_task_manager", lambda t: tasks_sent.append(t))
    monkeypatch.setattr(bot, "load_data", lambda table: df)
    tasks = bot.synthesise("records", RecordSchema())
    assert tasks
    assert tasks_sent


def test_simple_schema_validation(tmp_path):
    class SimpleRecord(isb.SimpleSchema):
        id = isb.simple_fields.Int(required=True)
        name = isb.simple_fields.Str(required=True)

    df = pd.DataFrame({"id": [1, "a"], "name": [None, None]})
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    reqs = bot.analyse(df, SimpleRecord(), "tbl")
    fields_seen = {r.field for r in reqs}
    assert "name" in fields_seen
    assert "id" in fields_seen


def test_celery_present(monkeypatch, tmp_path):
    class FakeCelery:
        def __init__(self, *a, **k):
            self.sent: list[tuple[str, dict[str, object] | None]] = []

        def send_task(self, name: str, kwargs=None):
            self.sent.append((name, kwargs))

    monkeypatch.setattr(isb, "Celery", FakeCelery)
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    bot.app.send_task("stage2.fetch", kwargs={"a": 1})
    assert ("stage2.fetch", {"a": 1}) in bot.app.sent


def test_queue_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(isb, "Celery", None)
    agg = DummyAggregator()
    wf = thb.WorkflowDB(tmp_path / "wf.db")
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:", aggregator=agg, workflow_db=wf, context_builder=builder
    )
    bot.app.send_task("stage2.fetch", kwargs={"b": 2})
    bot.app.queue.join()
    sent_payloads = [(task.name, task.kwargs) for task in bot.app.sent]
    executed_payloads = [(task.name, task.kwargs) for task in bot.app.executed]
    assert sent_payloads == [("stage2.fetch", {"b": 2})]
    assert executed_payloads == [("stage2.fetch", {"b": 2})]


def test_sqlalchemy_missing_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(isb, "create_engine", None, raising=False)

    class AggregatorStub:
        def process(self, topic: str, energy: int = 1):  # pragma: no cover - stub
            return []

    workflow_stub = types.SimpleNamespace(fetch=lambda: [])
    bot = isb.InformationSynthesisBot(
        db_url="sqlite:///:memory:",
        aggregator=AggregatorStub(),
        workflow_db=workflow_stub,
        context_builder=builder,
    )

    assert bot.engine is None
    assert not bot._sqlalchemy_available
    assert (
        bot._sqlalchemy_error_message
        == "SQLAlchemy helpers are unavailable: missing helper(s): create_engine"
    )

    with pytest.raises(RuntimeError) as excinfo:
        bot.load_data("records")
    assert "missing helper(s): create_engine" in str(excinfo.value)
