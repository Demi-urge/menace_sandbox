import pytest

pytest.importorskip("pandas")

import pandas as pd
import menace.data_bot as db
import menace.code_database as cd
import menace.capital_management_bot as cmb
from menace.unified_event_bus import UnifiedEventBus


def test_db_roundtrip(tmp_path):
    pdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord(bot="a", cpu=1.0, memory=2.0, response_time=0.1, disk_io=5.0, net_io=3.0, errors=0)
    pdb.add(rec)
    df = pdb.fetch()
    assert not df.empty and df.iloc[0]["bot"] == "a"


def test_detect_anomalies():
    data = {
        "cpu": [1, 2, 50],
        "memory": [1, 2, 3],
    }
    df = pd.DataFrame(data)
    rows = db.DataBot.detect_anomalies(df, "cpu", threshold=1.0)
    assert rows == [2]


def test_collect_emits_event(tmp_path):
    bus = UnifiedEventBus()
    events: list[dict] = []
    bus.subscribe("metrics:new", lambda t, e: events.append(e))
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(mdb, event_bus=bus)
    bot.collect("bot")
    assert events and events[0]["bot"] == "bot"


def test_worst_bot(tmp_path):
    pdb = db.MetricsDB(tmp_path / "w.db")
    data = db.DataBot(pdb)
    pdb.add(db.MetricRecord(bot="a", cpu=1.0, memory=1.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=5))
    pdb.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=2))
    assert data.worst_bot("errors") == "a"


def test_collect_logs_extra_metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    patch_db.add(cd.PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
    cap = cmb.CapitalManagementBot(data_bot=db.DataBot(mdb))
    bot = db.DataBot(mdb, capital_bot=cap, patch_db=patch_db)
    bot.collect("bot")
    rows = mdb.fetch_eval("system")
    metrics = {r[1] for r in rows}
    assert "patch_success_rate" in metrics
    assert "avg_energy_score" in metrics


def test_collect_additional_fields(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    bot = db.DataBot(mdb)
    bot.collect(
        "b",
        security_score=0.9,
        safety_rating=0.8,
        adaptability=0.7,
        antifragility=0.6,
        shannon_entropy=0.5,
        efficiency=0.4,
        flexibility=0.3,
        projected_lucrativity=1.2,
    )
    df = mdb.fetch()
    assert {
        "security_score",
        "safety_rating",
        "adaptability",
        "antifragility",
        "shannon_entropy",
        "efficiency",
        "flexibility",
        "projected_lucrativity",
    }.issubset(df.columns)


def test_fetch_eval_scope(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.log_eval("c", "m", 1.0)
    mdb.log_eval("c", "m", 2.0, source_menace_id="other")
    local = mdb.fetch_eval("c")
    assert len(local) == 1 and local[0][2] == 1.0
    remote = mdb.fetch_eval("c", scope="global")
    assert len(remote) == 1 and remote[0][2] == 2.0
    all_rows = mdb.fetch_eval("c", scope="all")
    assert len(all_rows) == 2


def test_fetch_scope(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(
        db.MetricRecord(
            bot="a",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=0,
        )
    )
    mdb.add(
        db.MetricRecord(
            bot="b",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=0,
        ),
        source_menace_id="other",
    )
    local = mdb.fetch(limit=None)
    assert len(local) == 1 and local.iloc[0]["bot"] == "a"
    remote = mdb.fetch(limit=None, scope="global")
    assert len(remote) == 1 and remote.iloc[0]["bot"] == "b"
    all_rows = mdb.fetch(limit=None, scope="all")
    assert len(all_rows) == 2
