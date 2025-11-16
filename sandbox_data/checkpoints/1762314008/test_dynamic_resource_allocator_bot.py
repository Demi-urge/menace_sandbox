import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.dynamic_resource_allocator_bot as drab
import menace.data_bot as db
import menace.resource_prediction_bot as rpb
import menace.neuroplasticity as neu


class _DummyBuilder:
    def build(self, *_: object, **__: object) -> str:
        return "ctx"

    def refresh_db_weights(self):
        pass


def test_allocate_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    rec = db.MetricRecord("bot1", 10.0, 50.0, 0.1, 1.0, 1.0, 0)
    mdb.add(rec)
    builder = _DummyBuilder()
    allocator = drab.DynamicResourceAllocator(
        mdb,
        rpb.ResourcePredictionBot(rpb.TemplateDB(tmp_path / "t.csv")),
        drab.DecisionLedger(tmp_path / "d.db"),
        drab.ResourceAllocationBot(
            drab.AllocationDB(tmp_path / "a.db"), context_builder=builder
        ),
        context_builder=builder,
    )
    actions = allocator.allocate(["bot1"])
    rows = allocator.ledger.fetch()
    assert actions and actions[0][0] == "bot1"
    assert rows and rows[0][0] == "bot1"


def test_myelinated_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "psutil", None)
    monkeypatch.setattr(db, "Gauge", None)
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord("bot1", 10.0, 50.0, 0.1, 1.0, 1.0, 0))
    mdb.add(db.MetricRecord("bot2", 10.0, 50.0, 0.1, 1.0, 1.0, 0))
    pdb = neu.PathwayDB(tmp_path / "p.db")
    pid = pdb.log(
        neu.PathwayRecord(
            actions="run_cycle:bot1",
            inputs="",
            outputs="",
            exec_time=0,
            resources="",
            outcome=neu.Outcome.SUCCESS,
            roi=2.0,
        )
    )
    pdb._update_meta(
        pid,
        neu.PathwayRecord(
            actions="run_cycle:bot1",
            inputs="",
            outputs="",
            exec_time=0,
            resources="",
            outcome=neu.Outcome.SUCCESS,
            roi=2.0,
        ),
    )
    builder = _DummyBuilder()
    allocator = drab.DynamicResourceAllocator(
        mdb,
        rpb.ResourcePredictionBot(rpb.TemplateDB(tmp_path / "t.csv")),
        drab.DecisionLedger(tmp_path / "d.db"),
        drab.ResourceAllocationBot(
            drab.AllocationDB(tmp_path / "a.db"), context_builder=builder
        ),
        pdb,
        context_builder=builder,
    )
    actions = allocator.allocate(["bot1", "bot2"])
    result = dict(actions)
    assert result["bot1"] is True
    assert result["bot2"] is False
