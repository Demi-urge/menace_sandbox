import os
import sys
import types
import pytest


os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
vs = types.ModuleType("vector_service")
vc = types.ModuleType("vector_service.context_builder")


class _StubBuilder:
    def refresh_db_weights(self):
        return {}


vc.ContextBuilder = _StubBuilder
vs.context_builder = vc
vs.EmbeddableDBMixin = object
sys.modules.setdefault("vector_service", vs)
sys.modules.setdefault("vector_service.context_builder", vc)

ue_stub = types.ModuleType("unified_event_bus")
ue_stub.UnifiedEventBus = object
ue_stub.EventBus = object
sys.modules.setdefault("unified_event_bus", ue_stub)
sys.modules.setdefault("menace.unified_event_bus", ue_stub)

ar_stub = types.ModuleType("automated_reviewer")
ar_stub.AutomatedReviewer = object
sys.modules.setdefault("automated_reviewer", ar_stub)
sys.modules.setdefault("menace.automated_reviewer", ar_stub)

import menace.diagnostic_manager as dm  # noqa: E402
import menace.data_bot as db  # noqa: E402
import menace.error_bot as eb  # noqa: E402


class DummyBuilder(dm.ContextBuilder):
    def refresh_db_weights(self):
        return {}


pytest.importorskip("pandas")


def make_metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1)
    mdb.add(rec)
    return mdb


def test_diagnose(tmp_path):
    mdb = make_metrics(tmp_path)
    e = eb.ErrorDB(tmp_path / "e.db")
    e.log_discrepancy("d")
    builder = DummyBuilder()
    manager = dm.DiagnosticManager(
        mdb, eb.ErrorBot(e, mdb, context_builder=builder), context_builder=builder
    )
    issues = manager.diagnose()
    assert "high_response_time" in issues
    assert "error_rate" in issues
    assert "discrepancies_detected" in issues


def test_resolve_and_log(tmp_path, monkeypatch):
    mdb = make_metrics(tmp_path)
    e = eb.ErrorDB(tmp_path / "e.db")
    builder = DummyBuilder()
    manager = dm.DiagnosticManager(
        mdb,
        eb.ErrorBot(e, mdb, context_builder=builder),
        context_builder=builder,
        log=dm.ResolutionDB(tmp_path / "r.db"),
    )
    monkeypatch.setattr(
        dm.DynamicResourceAllocator, "allocate", lambda self, bots: [(b, True) for b in bots]
    )
    manager.resolve_issue("high_response_time")
    rows = manager.log.fetch()
    assert rows and rows[0][0] == "high_response_time"
