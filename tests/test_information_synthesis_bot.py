import pytest

pytest.importorskip("pandas")
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import types, sys
if "cryptography" not in sys.modules:
    crypto = types.ModuleType("cryptography")
    haz = types.ModuleType("cryptography.hazmat")
    primitives = types.ModuleType("cryptography.hazmat.primitives")
    asym = types.ModuleType("cryptography.hazmat.primitives.asymmetric")
    ed = types.ModuleType("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = types.ModuleType("cryptography.hazmat.primitives.serialization")
    sys.modules.update(
        {
            "cryptography": crypto,
            "cryptography.hazmat": haz,
            "cryptography.hazmat.primitives": primitives,
            "cryptography.hazmat.primitives.asymmetric": asym,
            "cryptography.hazmat.primitives.asymmetric.ed25519": ed,
            "cryptography.hazmat.primitives.serialization": serialization,
        }
    )

import pandas as pd
if "menace.audit_trail" not in sys.modules:
    audit_stub = types.ModuleType("menace.audit_trail")
    class _Audit:
        def __init__(self, *a, **k):
            pass
    audit_stub.AuditTrail = _Audit
    sys.modules["menace.audit_trail"] = audit_stub
if "jinja2" not in sys.modules:
    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    sys.modules["jinja2"] = jinja_mod
for mod in ("env_config", "httpx"):
    sys.modules.setdefault(mod, types.ModuleType(mod))
if "yaml" not in sys.modules:
    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = lambda *a, **k: {}
    sys.modules["yaml"] = yaml_mod
if "sqlalchemy" not in sys.modules:
    sqlalchemy_mod = types.ModuleType("sqlalchemy")
    engine_mod = types.ModuleType("sqlalchemy.engine")
    class DummyMeta:
        def reflect(self, *a, **k):
            pass

    sqlalchemy_mod.create_engine = lambda *a, **k: object()
    sqlalchemy_mod.MetaData = lambda *a, **k: DummyMeta()
    sqlalchemy_mod.Table = lambda *a, **k: object()
    sqlalchemy_mod.select = lambda *a, **k: object()
    sqlalchemy_mod.engine = engine_mod
    sys.modules["sqlalchemy"] = sqlalchemy_mod
    sys.modules["sqlalchemy.engine"] = engine_mod
import menace.information_synthesis_bot as isb

try:
    from marshmallow import Schema, fields  # type: ignore
    HAS_MM = True
except Exception:  # pragma: no cover - marshmallow not available
    Schema = isb.Schema
    fields = isb.fields
    HAS_MM = False
import menace.research_aggregator_bot as rab
import menace.task_handoff_bot as thb


class DummyAggregator:
    def __init__(self):
        self.info_db = rab.InfoDB(":memory:")

    def process(self, topic: str, energy: int = 1):
        return []


class RecordSchema(Schema):
    id = fields.Int(required=True)
    name = fields.Str(required=True)
    value = fields.Str(required=True)


builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)


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
    sent = []
    monkeypatch.setattr(bot.app, "send_task", lambda name, kwargs=None: sent.append((name, kwargs)))
    reqs = bot.analyse(df, RecordSchema(), "records")
    bot.dispatch_requests(reqs)
    assert sent


def test_synthesise_creates_tasks(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {"id": [1], "name": ["Alpha"], "value": ["1"]}
    )
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
        id = isb.fields.Int(required=True)
        name = isb.fields.Str(required=True)

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
            self.sent = []

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
    assert bot.app.sent == [("stage2.fetch", {"b": 2})]
    assert bot.app.executed == [("stage2.fetch", {"b": 2})]
