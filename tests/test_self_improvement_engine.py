import os
import importlib.util
import sys
import pytest
import asyncio

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

try:
    import jinja2 as _j2  # noqa: F401
except Exception:  # pragma: no cover - optional
    import types

    jinja_stub = types.ModuleType("jinja2")
    jinja_stub.Template = lambda *a, **k: None
    sys.modules["jinja2"] = jinja_stub

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_stub)

if "networkx" not in sys.modules:
    nx_stub = types.ModuleType("networkx")
    nx_stub.DiGraph = object
    sys.modules["networkx"] = nx_stub

if "psutil" not in sys.modules:
    sys.modules["psutil"] = types.ModuleType("psutil")

if "loguru" not in sys.modules:
    loguru_mod = types.ModuleType("loguru")

    class DummyLogger:
        def __getattr__(self, name):
            def stub(*a, **k):
                return None

            return stub

        def add(self, *a, **k):  # pragma: no cover - optional
            pass

    loguru_mod.logger = DummyLogger()
    sys.modules["loguru"] = loguru_mod

if "git" not in sys.modules:
    git_mod = types.ModuleType("git")
    git_mod.Repo = object
    exc_mod = types.ModuleType("git.exc")
    class _Err(Exception):
        pass

    exc_mod.GitCommandError = _Err
    exc_mod.InvalidGitRepositoryError = _Err
    exc_mod.NoSuchPathError = _Err
    git_mod.exc = exc_mod
    sys.modules["git.exc"] = exc_mod
    sys.modules["git"] = git_mod

if "filelock" not in sys.modules:
    filelock_mod = types.ModuleType("filelock")
    filelock_mod.FileLock = object
    filelock_mod.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = filelock_mod

if "matplotlib" not in sys.modules:
    mpl_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    mpl_mod.pyplot = pyplot_mod
    sys.modules["matplotlib"] = mpl_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod

if "dotenv" not in sys.modules:
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv_mod

if "prometheus_client" not in sys.modules:
    prom_mod = types.ModuleType("prometheus_client")
    prom_mod.CollectorRegistry = object
    prom_mod.Counter = prom_mod.Gauge = lambda *a, **k: object()
    sys.modules["prometheus_client"] = prom_mod

if "joblib" not in sys.modules:
    joblib_mod = types.ModuleType("joblib")
    joblib_mod.dump = joblib_mod.load = lambda *a, **k: None
    sys.modules["joblib"] = joblib_mod

if "sklearn" not in sys.modules:
    sk_mod = types.ModuleType("sklearn")
    fe_mod = types.ModuleType("sklearn.feature_extraction")
    text_mod = types.ModuleType("sklearn.feature_extraction.text")
    text_mod.TfidfVectorizer = object
    fe_mod.text = text_mod
    cluster_mod = types.ModuleType("sklearn.cluster")
    cluster_mod.KMeans = object
    lm_mod = types.ModuleType("sklearn.linear_model")
    lm_mod.LinearRegression = object
    sk_mod.feature_extraction = fe_mod
    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.feature_extraction"] = fe_mod
    sys.modules["sklearn.feature_extraction.text"] = text_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.linear_model"] = lm_mod

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
import menace.self_improvement_policy as sip


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


def test_engine_policy_persistence(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None,
                roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    path = tmp_path / "policy.pkl"
    policy = sip.SelfImprovementPolicy(path=path)
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        policy=policy,
    )
    mdb.add(db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1))
    edb.log_discrepancy("fail")
    state = engine._policy_state()
    engine.run_cycle()
    val = engine.policy.score(state)

    policy2 = sip.SelfImprovementPolicy(path=path)
    engine2 = sie.SelfImprovementEngine(
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        policy=policy2,
    )
    assert engine2.policy.score(state) == val

