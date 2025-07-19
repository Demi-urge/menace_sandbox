import pytest
import asyncio

pytest.importorskip("pandas")

import menace.self_improvement_engine as sie
import menace.diagnostic_manager as dm
import menace.error_bot as eb
import menace.data_bot as db
import menace.research_aggregator_bot as rab
import menace.model_automation_pipeline as mp
import menace.pre_execution_roi_bot as prb
import menace.code_database as cd
import menace.evolution_history_db as eh


def test_run_cycle(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))
    agg = rab.ResearchAggregatorBot(["menace"], info_db=info)
    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None,
                roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    pipe = StubPipeline()
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    engine = sie.SelfImprovementEngine(interval=0, pipeline=pipe, diagnostics=diag, info_db=info)
    mdb.add(db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1))
    edb.log_discrepancy("fail")
    res = engine.run_cycle()
    assert isinstance(res, mp.AutomationResult)


def test_schedule_energy_threshold(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return f"{model}:{energy}"

    class DummyCapitalBot:
        def __init__(self, energy: float) -> None:
            self.energy = energy

        def energy_score(self, **_: object) -> float:
            return self.energy

    pipe = StubPipeline()
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(0.2),
        energy_threshold=0.5,
    )

    monkeypatch.setattr(engine, "_should_trigger", lambda: True)
    calls: list[int] = []
    monkeypatch.setattr(engine, "run_cycle", lambda energy=1: calls.append(energy))
    async def fake_sleep(_: float) -> None:
        raise SystemExit
    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run():
        task = engine.schedule()
        await task

    with pytest.raises(SystemExit):
        asyncio.run(run())
    assert calls == []

    engine.capital_bot = DummyCapitalBot(0.8)
    calls_high: list[int] = []
    monkeypatch.setattr(engine, "run_cycle", lambda energy=1: calls_high.append(energy))
    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run_high():
        task = engine.schedule()
        await task

    with pytest.raises(SystemExit):
        asyncio.run(run_high())
    assert calls_high == [int(round(0.8 * 5))]


def test_schedule_high_energy_autoruns(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return f"{model}:{energy}"

    class DummyCapitalBot:
        def __init__(self, energy: float) -> None:
            self.energy = energy

        def energy_score(self, **_: object) -> float:
            return self.energy

    pipe = StubPipeline()
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(0.9),
        energy_threshold=0.5,
    )

    monkeypatch.setattr(engine, "_should_trigger", lambda: False)
    calls: list[int] = []
    monkeypatch.setattr(engine, "run_cycle", lambda energy=1: calls.append(energy))
    async def fake_sleep(_: float) -> None:
        raise SystemExit
    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run_task():
        task = engine.schedule()
        await task

    with pytest.raises(SystemExit):
        asyncio.run(run_task())
    assert calls == [int(round(0.9 * 5))]


def test_policy_state_with_patch_metrics(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))
    hist = eh.EvolutionHistoryDB(tmp_path / "h.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=None)

    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        data_bot=db.DataBot(mdb, patch_db=patch_db),
        patch_db=patch_db,
        evolution_history=hist,
    )

    patch_db.add(
        cd.PatchRecord(
            filename="a.py",
            description="d1",
            roi_before=1.0,
            roi_after=2.0,
            roi_delta=1.0,
            complexity_before=0.0,
            complexity_after=0.2,
            complexity_delta=0.2,
            reverted=False,
        )
    )
    patch_db.add(
        cd.PatchRecord(
            filename="b.py",
            description="d2",
            roi_before=2.0,
            roi_after=1.5,
            roi_delta=-0.5,
            complexity_before=0.2,
            complexity_after=0.1,
            complexity_delta=-0.1,
            reverted=True,
        )
    )

    hist.add(eh.EvolutionEvent(action="self_improvement", before_metric=1.0, after_metric=2.0, roi=1.0, efficiency=80.0))
    hist.add(eh.EvolutionEvent(action="self_improvement", before_metric=2.0, after_metric=3.0, roi=1.5, efficiency=90.0))

    state = engine._policy_state()
    assert len(state) == 15

