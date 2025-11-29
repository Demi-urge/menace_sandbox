import pytest

pytest.importorskip("pandas")

import importlib
import threading
import contextlib
import types
import sys

import pandas as pd

import menace.data_bot as real_db
from db_router import init_db_router


def _load_bot(monkeypatch):
    def _normalise_manager_arg(manager, owner, *, fallback=None):
        return manager or fallback

    @contextlib.contextmanager
    def _structural_guard(owner=None):
        yield

    def _prepare_pipeline_for_bootstrap(*args, **kwargs):
        return types.SimpleNamespace(name="pipeline"), lambda *_: None

    def _get_structural_bootstrap_owner():
        return None

    stub_cbi = types.SimpleNamespace(
        self_coding_managed=lambda **_: (lambda cls: cls),
        normalise_manager_arg=_normalise_manager_arg,
        prepare_pipeline_for_bootstrap=_prepare_pipeline_for_bootstrap,
        structural_bootstrap_owner_guard=_structural_guard,
        get_structural_bootstrap_owner=_get_structural_bootstrap_owner,
    )
    monkeypatch.setitem(sys.modules, "coding_bot_interface", stub_cbi)
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", stub_cbi)
    class DummyManager:
        def __init__(self, *a, **k):
            pass
    def _internalize(*args, **kwargs):
        return DummyManager()
    stub_mgr_mod = types.SimpleNamespace(SelfCodingManager=DummyManager)
    stub_mgr_mod.internalize_coding_bot = _internalize
    monkeypatch.setitem(sys.modules, "self_coding_manager", stub_mgr_mod)
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", stub_mgr_mod)
    class StubDataBot:
        def __init__(self, *a, **k):
            pass
        @staticmethod
        def complexity_score(df):
            return real_db.DataBot.complexity_score(df)
        def average_errors(self, *a, **k):
            return 0.0
    stub_db_mod = types.SimpleNamespace(
        MetricsDB=real_db.MetricsDB, MetricRecord=real_db.MetricRecord, DataBot=StubDataBot
    )
    monkeypatch.setitem(sys.modules, "data_bot", stub_db_mod)
    monkeypatch.setitem(sys.modules, "menace.data_bot", stub_db_mod)
    import menace.structural_evolution_bot as seb
    return seb


def test_predict_and_apply(tmp_path, monkeypatch):
    seb = _load_bot(monkeypatch)
    mdb = real_db.MetricsDB(tmp_path / "m.db")
    rec = real_db.MetricRecord("bot", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    class T:
        roi_drop = 0.0
        error_threshold = 500.0
        test_failure_threshold = 0.0
    monkeypatch.setattr(
        seb.manager,
        "threshold_service",
        types.SimpleNamespace(get=lambda name=None: T()),
        raising=False,
    )
    router = init_db_router("evo", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(metrics_db=mdb, db=seb.EvolutionDB(router=router))
    snap = bot.take_snapshot()
    preds = bot.predict_changes(snap)
    assert preds
    applied = bot.apply_minor_changes()
    if preds[0].severity == "minor":
        assert applied == [preds[0].change]
    else:
        assert applied == []
    assert router._access_counts["local"]["evolutions"] >= 1
    router.close()


def test_major_change(tmp_path, monkeypatch):
    seb = _load_bot(monkeypatch)
    mdb = real_db.MetricsDB(tmp_path / "m.db")
    rec = real_db.MetricRecord("bot", 200.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(rec)
    class T:
        roi_drop = 0.0
        error_threshold = 10.0
        test_failure_threshold = 0.0
    monkeypatch.setattr(
        seb.manager,
        "threshold_service",
        types.SimpleNamespace(get=lambda name=None: T()),
        raising=False,
    )
    router = init_db_router("evo2", str(tmp_path / "e.db"), str(tmp_path / "e.db"))
    bot = seb.StructuralEvolutionBot(metrics_db=mdb, db=seb.EvolutionDB(router=router))
    snap = bot.take_snapshot()
    recs = bot.predict_changes(snap)
    if recs[0].severity == "major":
        approved = bot.apply_major_change(recs[0], approve_cb=lambda r: True)
        assert approved
    assert router._access_counts["local"]["evolutions"] >= 1
    router.close()


def test_import_while_bootstrap_in_progress(monkeypatch):
    sys.modules.pop("menace.structural_evolution_bot", None)
    sys.modules.pop("structural_evolution_bot", None)

    calls: list[str] = []
    start_gate = threading.Event()
    release_gate = threading.Event()

    class DummyManager:
        pass

    def fake_prepare(*args, **kwargs):
        calls.append("prepare")
        start_gate.set()
        release_gate.wait(timeout=5)
        return types.SimpleNamespace(name="pipeline"), lambda *_: None

    def fake_internalize(*args, **kwargs):
        return DummyManager()

    monkeypatch.setattr(
        "menace.coding_bot_interface.prepare_pipeline_for_bootstrap", fake_prepare
    )
    monkeypatch.setattr(
        "menace.self_coding_manager.internalize_coding_bot", fake_internalize
    )
    monkeypatch.setattr("menace.code_database.CodeDB", lambda: object())
    monkeypatch.setattr("menace.gpt_memory.GPTMemoryManager", lambda *a, **k: object())
    monkeypatch.setattr(
        "menace.self_coding_engine.SelfCodingEngine", lambda *a, **k: types.SimpleNamespace()
    )
    monkeypatch.setattr("menace.context_builder_util.create_context_builder", lambda: object())
    monkeypatch.setattr("menace.shared_evolution_orchestrator.get_orchestrator", lambda *a, **k: object())
    monkeypatch.setattr(
        "menace.self_coding_thresholds.get_thresholds",
        lambda *a, **k: types.SimpleNamespace(
            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
        ),
    )
    monkeypatch.setattr("menace.data_bot.persist_sc_thresholds", lambda *a, **k: None)
    monkeypatch.setattr(
        "menace.threshold_service.ThresholdService", lambda *a, **k: types.SimpleNamespace()
    )
    monkeypatch.setattr("menace.data_bot.DataBot", lambda *a, **k: types.SimpleNamespace())
    monkeypatch.setattr(
        "menace.model_automation_pipeline.ModelAutomationPipeline", type("Pipeline", (), {})
    )

    owner_token = object()

    def bootstrap_thread():
        seb = importlib.import_module("menace.structural_evolution_bot")
        seb.get_structural_evolution_manager(owner_token)

    bootstrap = threading.Thread(target=bootstrap_thread)
    bootstrap.start()

    assert start_gate.wait(timeout=5)
    seb = importlib.import_module("menace.structural_evolution_bot")

    waiter = threading.Thread(
        target=lambda: seb.get_structural_evolution_manager(owner_token)
    )
    waiter.start()

    assert calls == ["prepare"]
    release_gate.set()
    bootstrap.join(timeout=5)
    waiter.join(timeout=5)
    assert calls == ["prepare"]
    assert not bootstrap.is_alive()
    assert not waiter.is_alive()
