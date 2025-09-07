import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.diagnostic_manager as dm
import menace.data_bot as db
import menace.error_bot as eb
import types


def make_metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1)
    mdb.add(rec)
    return mdb


def test_diagnose(tmp_path):
    mdb = make_metrics(tmp_path)
    e = eb.ErrorDB(tmp_path / "e.db")
    e.log_discrepancy("d")
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    manager = dm.DiagnosticManager(mdb, eb.ErrorBot(e, mdb, context_builder=builder))
    issues = manager.diagnose()
    assert "high_response_time" in issues
    assert "error_rate" in issues
    assert "discrepancies_detected" in issues


def test_resolve_and_log(tmp_path, monkeypatch):
    mdb = make_metrics(tmp_path)
    e = eb.ErrorDB(tmp_path / "e.db")
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    manager = dm.DiagnosticManager(
        mdb,
        eb.ErrorBot(e, mdb, context_builder=builder),
        log=dm.ResolutionDB(tmp_path / "r.db"),
    )
    monkeypatch.setattr(dm.DynamicResourceAllocator, "allocate", lambda self, bots: [(b, True) for b in bots])
    manager.resolve_issue("high_response_time")
    rows = manager.log.fetch()
    assert rows and rows[0][0] == "high_response_time"

