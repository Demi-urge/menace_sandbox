import pytest

pytest.importorskip("pandas")

import types
import sys

od_stub = types.ModuleType("sandbox_runner.orphan_discovery")
for _fn in (
    "append_orphan_cache",
    "append_orphan_classifications",
    "prune_orphan_cache",
    "load_orphan_cache",
):
    setattr(od_stub, _fn, lambda *a, **k: None)
sys.modules.setdefault("sandbox_runner.orphan_discovery", od_stub)
sys.modules.setdefault("orphan_discovery", od_stub)
neuro_stub = types.ModuleType("neurosales")
neuro_stub.add_message = lambda *a, **k: None
sys.modules.setdefault("neurosales", neuro_stub)

import menace.self_improvement as sie
import asyncio
import menace.diagnostic_manager as dm
import menace.error_bot as eb
import menace.data_bot as db
import menace.research_aggregator_bot as rab
import menace.pre_execution_roi_bot as prb
import menace.model_automation_pipeline as mp

class StubPipeline:
    def __init__(self):
        self.calls = []

    def run(self, model: str, energy: int = 1):
        self.calls.append((model, energy))
        return mp.AutomationResult(
            package=None,
            roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
        )

def _make_engine(tmp_path, name: str, monkeypatch):
    mdb = db.MetricsDB(tmp_path / f"{name}.m.db")
    edb = eb.ErrorDB(tmp_path / f"{name}.e.db")
    info = rab.InfoDB(tmp_path / f"{name}.i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))
    pipe = StubPipeline()
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        bot_name=name,
    )
    return engine, pipe

def test_registry_runs_multiple(tmp_path, monkeypatch):
    eng1, pipe1 = _make_engine(tmp_path, "menace", monkeypatch)
    eng2, pipe2 = _make_engine(tmp_path, "alpha", monkeypatch)
    reg = sie.ImprovementEngineRegistry()
    reg.register_engine("menace", eng1)
    reg.register_engine("alpha", eng2)
    monkeypatch.setattr(eng1, "_should_trigger", lambda: True)
    monkeypatch.setattr(eng2, "_should_trigger", lambda: True)
    res = reg.run_all_cycles()
    assert set(res) == {"menace", "alpha"}
    assert isinstance(res["menace"], mp.AutomationResult)
    assert pipe1.calls and pipe2.calls

def test_engine_custom_name(tmp_path, monkeypatch):
    eng, pipe = _make_engine(tmp_path, "beta", monkeypatch)
    monkeypatch.setattr(eng, "_should_trigger", lambda: True)
    res = eng.run_cycle()
    assert isinstance(res, mp.AutomationResult)
    assert eng.aggregator.requirements == ["beta"]


def test_registry_autoscale(tmp_path, monkeypatch):
    eng, _ = _make_engine(tmp_path, "base", monkeypatch)
    reg = sie.ImprovementEngineRegistry()
    reg.register_engine("base", eng)

    class Cap:
        def __init__(self, val: float) -> None:
            self.val = val

        def energy_score(self, **_: object) -> float:
            return self.val

    class Data:
        def __init__(self, t: float) -> None:
            self.t = t

        def long_term_roi_trend(self, limit: int = 200) -> float:
            return self.t

    def factory(name: str):
        return _make_engine(tmp_path, name, monkeypatch)[0]

    reg.autoscale(
        capital_bot=Cap(0.9),
        data_bot=Data(0.5),
        factory=factory,
        max_engines=2,
        create_energy=0.8,
        roi_threshold=0.1,
    )
    assert len(reg.engines) == 2

    reg.autoscale(
        capital_bot=Cap(0.2),
        data_bot=Data(-0.2),
        factory=factory,
        min_engines=1,
        remove_energy=0.3,
        roi_threshold=0.0,
    )
    assert len(reg.engines) == 1


def test_concurrent_schedules(tmp_path, monkeypatch):
    eng1, _ = _make_engine(tmp_path, "e1", monkeypatch)
    eng2, _ = _make_engine(tmp_path, "e2", monkeypatch)

    monkeypatch.setattr(eng1, "_should_trigger", lambda: True)
    monkeypatch.setattr(eng2, "_should_trigger", lambda: True)

    calls: list[str] = []
    monkeypatch.setattr(eng1, "run_cycle", lambda energy=1: calls.append("e1"))
    monkeypatch.setattr(eng2, "run_cycle", lambda energy=1: calls.append("e2"))

    async def fake_sleep(_: float) -> None:
        raise SystemExit

    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run_all():
        t1 = eng1.schedule()
        t2 = eng2.schedule()
        await asyncio.gather(t1, t2)

    with pytest.raises(SystemExit):
        asyncio.run(run_all())

    assert sorted(calls) == ["e1", "e2"]
