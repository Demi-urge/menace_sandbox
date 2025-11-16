import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import types  # noqa: E402
import menace.operational_monitor_bot as omb  # noqa: E402
import menace.data_bot as db  # noqa: E402
import codex_db_helpers as helpers  # noqa: E402
from discrepancy_db import DiscrepancyDB  # noqa: E402
from db_router import DBRouter  # noqa: E402


class StubES:
    def __init__(self) -> None:
        self.docs = []

    def add(self, doc_id: str, body: dict) -> None:
        self.docs.append({"id": doc_id, **body})

    def search_all(self):
        return list(self.docs)


def test_collect_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    bot = omb.OperationalMonitoringBot(mdb, es, db_router=router)
    bot.collect_and_export("bot1")
    df = mdb.fetch(1)
    assert not df.empty
    assert es.search_all()
    assert "bot1" in router.terms


def test_detect_anomalies(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    a_db = omb.AnomalyDB(tmp_path / "a.db")
    router2 = types.SimpleNamespace(terms=[])
    router2.query_all = lambda term: router2.terms.append(term) or {}
    bot = omb.OperationalMonitoringBot(mdb, es, a_db, db_router=router2)
    normal = db.MetricRecord("bot1", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(normal)
    anomaly = db.MetricRecord("bot1", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anomaly)
    found = bot.detect_anomalies("bot1")
    rows = a_db.fetch()
    assert found
    assert rows
    assert "bot1" in router2.terms


def test_detect_anomalies_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "Gauge", None)
    router = DBRouter("t", str(tmp_path / "loc.db"), str(tmp_path / "shared.db"))
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    a_db = omb.AnomalyDB(tmp_path / "a.db", router=router)
    d_db = DiscrepancyDB(
        path=tmp_path / "d.db", router=router, vector_index_path=tmp_path / "d.index"
    )
    monkeypatch.setattr(omb, "DiscrepancyDB", lambda: d_db)
    monkeypatch.setattr(helpers, "DiscrepancyDB", lambda: d_db)
    bot = omb.OperationalMonitoringBot(mdb, es, a_db, db_router=router)
    base = db.MetricRecord("bot1", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(base)
    anom = db.MetricRecord("bot1", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anom)
    bot.detect_anomalies("bot1", write_codex=True)
    samples = helpers.fetch_discrepancies(limit=10)
    assert any("bot1" in s.content for s in samples)
