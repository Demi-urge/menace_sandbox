import os
import importlib.util
import sys
import pytest
import asyncio

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
spec = importlib.util.spec_from_file_location("menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py"))
menace = importlib.util.module_from_spec(spec)
sys.modules.setdefault("menace", menace)
spec.loader.exec_module(menace)

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
    assert len(state) == sie.POLICY_STATE_LEN


def test_pre_roi_energy_scaling(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))

    class StubPipeline:
        def __init__(self) -> None:
            self.energy = None

        def run(self, model: str, energy: int = 1):
            self.energy = energy
            return mp.AutomationResult(package=None, roi=None)

    class DummyCapitalBot:
        def energy_score(self, **_: object) -> float:
            return 0.4

    class HighROIBot:
        def predict_model_roi(self, *_: object) -> prb.ROIResult:
            return prb.ROIResult(0.0, 0.0, 0.0, 1.0, 0.0)

    class LowROIBot:
        def predict_model_roi(self, *_: object) -> prb.ROIResult:
            return prb.ROIResult(0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    pipe_high = StubPipeline()
    eng_high = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe_high,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        pre_roi_bot=HighROIBot(),
    )
    eng_high.run_cycle()

    pipe_low = StubPipeline()
    eng_low = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe_low,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        pre_roi_bot=LowROIBot(),
    )
    eng_low.run_cycle()

    assert pipe_high.energy > pipe_low.energy


def test_policy_state_includes_synergy(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=None)

    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
    )

    class DummyTracker:
        def __init__(self) -> None:
            self.metrics_history = {
                "synergy_roi": [0.25],
                "synergy_efficiency": [0.35],
                "synergy_resilience": [-0.15],
                "synergy_antifragility": [0.12],
            }

    engine.tracker = DummyTracker()
    state = engine._policy_state()
    assert len(state) == sie.POLICY_STATE_LEN
    assert state[-4:] == (2, 4, -2, 1)

