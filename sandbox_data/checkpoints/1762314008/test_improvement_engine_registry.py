import os
import types
import sys
import pytest


os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
vs = types.ModuleType("vector_service")
vc = types.ModuleType("vector_service.context_builder")


class _StubBuilder:
    def refresh_db_weights(self):
        return {}


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: _StubBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)


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


class DummyBuilder(dm.ContextBuilder):
    def refresh_db_weights(self):
        return {}


vs.context_builder.ContextBuilder = DummyBuilder

pytest.importorskip("pandas")


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

boot_stub = types.ModuleType("sandbox_runner.bootstrap")
boot_stub.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot_stub)

sts_stub = types.ModuleType("menace.self_test_service")
sts_stub.SelfTestService = object
sys.modules.setdefault("menace.self_test_service", sts_stub)

import menace.self_improvement as sie  # noqa: E402
import menace.error_bot as eb  # noqa: E402
import menace.data_bot as db  # noqa: E402
import menace.research_aggregator_bot as rab  # noqa: E402
import menace.pre_execution_roi_bot as prb  # noqa: E402
import menace.model_automation_pipeline as mp  # noqa: E402


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
    builder = DummyBuilder()
    diag = dm.DiagnosticManager(
        mdb, eb.ErrorBot(edb, mdb, context_builder=builder), context_builder=builder
    )
    pipe = StubPipeline()
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
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

    # establish baseline
    reg.autoscale(capital_bot=Cap(0.5), data_bot=Data(0.1), factory=factory)

    sie.registry.settings.autoscale_create_dev_multiplier = 0.0
    sie.registry.settings.autoscale_remove_dev_multiplier = 0.0
    sie.registry.settings.autoscale_roi_dev_multiplier = 0.0
