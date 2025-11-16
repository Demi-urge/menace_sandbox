import types
import sys

import pytest


def _setup_stubs(monkeypatch):
    stub = types.ModuleType("stub")
    monkeypatch.setitem(sys.modules, "pandas", stub)
    monkeypatch.setattr(stub, "read_sql", lambda *a, **k: [], raising=False)
    monkeypatch.setitem(sys.modules, "networkx", types.SimpleNamespace(DiGraph=object))
    monkeypatch.setitem(sys.modules, "jinja2", types.SimpleNamespace(Template=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "yaml", stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy", types.ModuleType("sqlalchemy"))
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", types.SimpleNamespace(Engine=object))
    monkeypatch.setitem(sys.modules, "psutil", stub)
    monkeypatch.setitem(sys.modules, "numpy", stub)
    monkeypatch.setitem(sys.modules, "git", types.SimpleNamespace(Repo=object))
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", types.ModuleType("pyplot"))  # path-ignore
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "prometheus_client", types.SimpleNamespace(CollectorRegistry=object, Counter=object, Gauge=object))
    monkeypatch.setitem(sys.modules, "sklearn", stub)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", stub)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", types.SimpleNamespace(TfidfVectorizer=object))
    monkeypatch.setitem(sys.modules, "sklearn.cluster", types.SimpleNamespace(KMeans=object))
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", types.SimpleNamespace(LinearRegression=object))
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", types.SimpleNamespace(train_test_split=lambda *a, **k: ([], [])))
    monkeypatch.setitem(sys.modules, "sklearn.metrics", types.SimpleNamespace(accuracy_score=lambda *a, **k: 0.0))
    monkeypatch.setattr(sys.modules["sklearn.linear_model"], "LogisticRegression", object, raising=False)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", types.SimpleNamespace(RandomForestClassifier=object))
    return stub


def test_roi_event_roundtrip(tmp_path):
    import menace.capital_management_bot as cmb
    db = cmb.ROIEventDB(tmp_path / "r.db")
    rec = cmb.ROIEvent("a", 1.0, 2.0)
    db.add(rec)
    rows = db.fetch()
    assert rows and rows[0][0] == "a"


def test_capital_bot_logs_event(tmp_path):
    import menace.capital_management_bot as cmb
    roi_db = cmb.ROIEventDB(tmp_path / "c.db")
    bot = cmb.CapitalManagementBot(roi_db=roi_db)
    bot.log_evolution_event("t", 1.0, 2.0)
    rows = roi_db.fetch()
    assert rows and rows[0][1] == 1.0


def test_orchestrator_calls_capital_log(monkeypatch, tmp_path):
    _setup_stubs(monkeypatch)
    import menace.evolution_orchestrator as eo
    calls = []
    data_bot = types.SimpleNamespace(db=types.SimpleNamespace(fetch=lambda limit=50: []), log_evolution_cycle=lambda *a, **k: None)
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0, log_evolution_event=lambda *a, **k: calls.append(a))
    orch = eo.EvolutionOrchestrator(
        data_bot,
        cap_bot,
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(roi=None)),
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(ga_results={}, predictions=[])),
    )
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)
    orch.run_cycle()
    assert calls and calls[0][0] == "self_improvement"


class DummyPipeline:
    def run(self, model: str, energy: int = 1):
        return types.SimpleNamespace(package=None, roi=None)


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)


def test_self_improvement_logs_capital(monkeypatch, tmp_path):
    stub = _setup_stubs(monkeypatch)
    import menace.self_improvement as sie
    import menace.capital_management_bot as cmb
    import menace.research_aggregator_bot as rab
    import menace.data_bot as db

    mdb = db.MetricsDB(tmp_path / "m.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = types.SimpleNamespace(metrics=mdb, error_bot=types.SimpleNamespace(db=types.SimpleNamespace(discrepancies=lambda: [])), diagnose=lambda: [])
    cap_calls = []
    cap_bot = cmb.CapitalManagementBot(roi_db=cmb.ROIEventDB(tmp_path / "d.db"))
    cap_bot.log_evolution_event = lambda *a, **k: cap_calls.append(a)
    data_stub = types.SimpleNamespace(db=mdb, log_evolution_cycle=lambda *a, **k: None)
    engine = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        pipeline=DummyPipeline(),
        diagnostics=diag,
        info_db=info,
        capital_bot=cap_bot,
        data_bot=data_stub,
    )
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    monkeypatch.setattr(engine, "_record_state", lambda: None)
    mdb.add(db.MetricRecord("bot", 0, 0, 0, 0, 0, 0))
    engine.run_cycle()
    assert cap_calls and cap_calls[0][0] == "self_improvement"
