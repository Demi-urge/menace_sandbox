# flake8: noqa
import os
import importlib.util
import sys
import pytest
import asyncio

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import types

from pathlib import Path

vs = types.ModuleType("vector_service")


class _StubBuilder:
    def __init__(self, *a, **k):
        pass

    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: _StubBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)


vs.ContextBuilder = _StubBuilder
vs.CognitionLayer = object
sys.modules["vector_service"] = vs

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules["menace"] = menace_pkg
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path
import types
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(os.path.dirname(__file__), "..", "__init__.py")  # path-ignore
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace

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

    class DummyLock:
        def __init__(self, *a, **k):
            self.is_locked = False
            self.lock_file = ""

        def acquire(self, *a, **k):
            self.is_locked = True

        def release(self):  # pragma: no cover - simplicity
            self.is_locked = False

        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, *a):  # pragma: no cover
            pass

    filelock_mod.FileLock = DummyLock
    filelock_mod.Timeout = type("Timeout", (Exception,), {})
    sys.modules["filelock"] = filelock_mod

if "matplotlib" not in sys.modules:
    mpl_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")  # path-ignore
    mpl_mod.pyplot = pyplot_mod  # path-ignore
    sys.modules["matplotlib"] = mpl_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod  # path-ignore

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

if "pydantic" not in sys.modules:
    pyd_mod = types.ModuleType("pydantic")
    pyd_mod.__path__ = []  # type: ignore
    pyd_dc = types.ModuleType("dataclasses")
    pyd_dc.dataclass = lambda *a, **k: (lambda f: f)
    pyd_mod.dataclasses = pyd_dc
    pyd_mod.Field = lambda default=None, **k: default
    pyd_mod.ConfigDict = dict
    pyd_mod.field_validator = lambda *a, **k: (lambda f: f)
    pyd_mod.model_validator = lambda *a, **k: (lambda f: f)
    class _VE(Exception):
        pass
    pyd_mod.ValidationError = _VE
    pyd_mod.BaseModel = object
    sys.modules["pydantic"] = pyd_mod
    sys.modules["pydantic.dataclasses"] = pyd_dc
    pyd_settings_mod = types.ModuleType("pydantic_settings")
    pyd_settings_mod.BaseSettings = object
    pyd_settings_mod.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = pyd_settings_mod

sk_mod = types.ModuleType("sklearn")
sk_mod.__path__ = []  # type: ignore
fe_mod = types.ModuleType("sklearn.feature_extraction")
fe_mod.__path__ = []  # type: ignore
text_mod = types.ModuleType("sklearn.feature_extraction.text")
text_mod.__path__ = []  # type: ignore
text_mod.TfidfVectorizer = object
fe_mod.text = text_mod
sk_mod.feature_extraction = fe_mod
sk_mod.feature_extraction.text = text_mod
cluster_mod = types.ModuleType("sklearn.cluster")
cluster_mod.__path__ = []  # type: ignore
cluster_mod.KMeans = object
lm_mod = types.ModuleType("sklearn.linear_model")
lm_mod.__path__ = []  # type: ignore
lm_mod.LinearRegression = object
sk_mod.cluster = cluster_mod
sk_mod.linear_model = lm_mod
pre_mod = types.ModuleType("sklearn.preprocessing")
pre_mod.__path__ = []  # type: ignore
pre_mod.PolynomialFeatures = object
sk_mod.preprocessing = pre_mod
sys.modules["sklearn"] = sk_mod
sys.modules["sklearn.feature_extraction"] = fe_mod
sys.modules["sklearn.feature_extraction.text"] = text_mod
sys.modules["sklearn.cluster"] = cluster_mod
sys.modules["sklearn.linear_model"] = lm_mod
sys.modules["sklearn.preprocessing"] = pre_mod

spec.loader.exec_module(menace)

run_auto_mod = types.ModuleType("run_autonomous")
run_auto_mod._verify_required_dependencies = lambda: None
sys.modules["run_autonomous"] = run_auto_mod
sys.modules["menace.run_autonomous"] = run_auto_mod

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
sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
boot_mod = types.ModuleType("sandbox_runner.bootstrap")
boot_mod.initialize_autonomous_sandbox = lambda *a, **k: None
sys.modules.setdefault("sandbox_runner.bootstrap", boot_mod)
sys.modules.setdefault(
    "sandbox_runner.cycle", types.ModuleType("sandbox_runner.cycle")
)
sys.modules.setdefault(
    "sandbox_runner.environment", types.ModuleType("sandbox_runner.environment")
)
ss_mod = types.ModuleType("sandbox_settings")
class SandboxSettingsStub:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return 1

ss_mod.SandboxSettings = SandboxSettingsStub
ss_mod.load_sandbox_settings = lambda *a, **k: SandboxSettingsStub()
class ROISettings:
    def __init__(self, *a, **k):
        pass

ss_mod.ROISettings = ROISettings
sys.modules["sandbox_settings"] = ss_mod
sys.modules["menace.sandbox_settings"] = ss_mod

gpt_mem_mod = types.ModuleType("gpt_memory")

class GPTMemoryManager:
    def __init__(self, *a, **k):
        pass

gpt_mem_mod.GPTMemoryManager = GPTMemoryManager
gpt_mem_mod.INSIGHT = object()
gpt_mem_mod.STANDARD_TAGS = {}
sys.modules["gpt_memory"] = gpt_mem_mod
sys.modules["menace.gpt_memory"] = gpt_mem_mod

gpt_mem_int_mod = types.ModuleType("gpt_memory_interface")
gpt_mem_int_mod.GPTMemoryInterface = object
sys.modules["gpt_memory_interface"] = gpt_mem_int_mod
sys.modules["menace.gpt_memory_interface"] = gpt_mem_int_mod

gpt_knowledge_mod = types.ModuleType("gpt_knowledge_service")
gpt_knowledge_mod.GPTKnowledgeService = object
sys.modules["gpt_knowledge_service"] = gpt_knowledge_mod
sys.modules["menace.gpt_knowledge_service"] = gpt_knowledge_mod

rab_stub = types.ModuleType("research_aggregator_bot")

class InfoDB:
    def __init__(self, *a, **k):
        pass


class ResearchAggregatorBot:
    def __init__(self, *a, **k):
        pass


rab_stub.InfoDB = InfoDB
rab_stub.ResearchAggregatorBot = ResearchAggregatorBot
class ResearchItem:  # minimal placeholder
    pass

rab_stub.ResearchItem = ResearchItem
sys.modules["research_aggregator_bot"] = rab_stub
sys.modules["menace.research_aggregator_bot"] = rab_stub

arp_stub = types.ModuleType("adaptive_roi_predictor")

class AdaptiveROIPredictor:
    def __init__(self, *a, **k):
        pass

def load_training_data(*a, **k):
    return []

arp_stub.AdaptiveROIPredictor = AdaptiveROIPredictor
arp_stub.load_training_data = load_training_data
sys.modules["adaptive_roi_predictor"] = arp_stub
sys.modules["menace.adaptive_roi_predictor"] = arp_stub

roi_tracker_stub = types.ModuleType("roi_tracker")

class ROITracker:
    def __init__(self, *a, **k):
        pass

    def diminishing(self) -> float:
        return 0.05

    def calculate_raroi(self, roi, **k):
        return roi, roi, []

    def score_workflow(self, wf_id, raroi, tau=None):
        pass

roi_tracker_stub.ROITracker = ROITracker
sys.modules["roi_tracker"] = roi_tracker_stub
sys.modules["menace.roi_tracker"] = roi_tracker_stub

# Stub heavy modules to minimal placeholders
for name in [
    "chatgpt_idea_bot",
    "chatgpt_enhancement_bot",
    "shared_gpt_memory",
    "shared_knowledge_module",
    "local_knowledge_module",
    "implementation_optimiser_bot",
    "self_coding_engine",
    "error_bot",
    "data_bot",
]:
    mod = types.ModuleType(name)
    if name == "local_knowledge_module":
        class LocalKnowledgeModule:
            pass

        mod.LocalKnowledgeModule = LocalKnowledgeModule
        mod.init_local_knowledge = lambda *a, **k: None
    if name == "self_coding_engine":
        class SelfCodingEngine:
            pass

        mod.SelfCodingEngine = SelfCodingEngine
        mod.MANAGER_CONTEXT = types.SimpleNamespace(
            set=lambda *_a, **_k: None,
            reset=lambda _token: None,
        )
    if name == "error_bot":
        class ErrorDB:
            def __init__(self, *a, **k):
                pass

        class ErrorBot:
            def __init__(self, *a, **k):
                pass

        mod.ErrorDB = ErrorDB
        mod.ErrorBot = ErrorBot
    if name == "data_bot":
        class DataBot:
            def __init__(self, *a, **k):
                pass

        mod.DataBot = DataBot
    sys.modules[name] = mod
    sys.modules[f"menace.{name}"] = mod

mp_stub = types.ModuleType("model_automation_pipeline")

class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi
        self.workflow_evolution = None
        self.warnings = None

mp_stub.AutomationResult = AutomationResult

class ModelAutomationPipeline:
    def __init__(self, *a, **k):
        pass

    def run(self, model: str, energy: int = 1):
        return AutomationResult(package=None, roi=prb_stub.ROIResult(1.0))

mp_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["model_automation_pipeline"] = mp_stub
sys.modules["menace.model_automation_pipeline"] = mp_stub

prb_stub = types.ModuleType("pre_execution_roi_bot")

class ROIResult:
    def __init__(self, roi, *a):
        self.roi = roi

prb_stub.ROIResult = ROIResult
class PreExecutionROIBot:
    pass

prb_stub.PreExecutionROIBot = PreExecutionROIBot
class BuildTask:
    pass

prb_stub.BuildTask = BuildTask
sys.modules["pre_execution_roi_bot"] = prb_stub
sys.modules["menace.pre_execution_roi_bot"] = prb_stub

neuro_mod = types.ModuleType("neurosales")
neuro_mod.add_message = lambda *a, **k: None
neuro_mod.get_recent_messages = lambda *a, **k: []
sys.modules["neurosales"] = neuro_mod

vs_mod = types.ModuleType("vector_service")
vs_mod.CognitionLayer = object
vs_mod.EmbeddableDBMixin = object
vs_mod.Retriever = object
vs_mod.FallbackResult = object
vs_mod.SharedVectorService = object
sys.modules["vector_service"] = vs_mod
sub = types.ModuleType("vector_service.cognition_layer")
sub.CognitionLayer = object
sys.modules["vector_service.cognition_layer"] = sub
analytics_mod = types.ModuleType("analytics")
analytics_mod.adaptive_roi_model = types.ModuleType("analytics.adaptive_roi_model")
sys.modules["analytics"] = analytics_mod
sys.modules["analytics.adaptive_roi_model"] = analytics_mod.adaptive_roi_model
pytest.importorskip("pandas")

err_logger_mod = types.ModuleType("error_logger")
err_logger_mod.TelemetryEvent = object
class _ErrLogger:
    pass
err_logger_mod.ErrorLogger = _ErrLogger
sys.modules["error_logger"] = err_logger_mod
sys.modules["menace.error_logger"] = err_logger_mod

import menace.self_improvement as sie

_BT_SPEC = importlib.util.spec_from_file_location(
    "baseline_tracker", Path(__file__).resolve().parents[1] / "self_improvement" / "baseline_tracker.py"  # path-ignore
)
baseline_tracker = importlib.util.module_from_spec(_BT_SPEC)
assert _BT_SPEC and _BT_SPEC.loader
_BT_SPEC.loader.exec_module(baseline_tracker)  # type: ignore[attr-defined]
BaselineTracker = baseline_tracker.BaselineTracker
import menace.diagnostic_manager as dm
import menace.error_bot as eb
import menace.data_bot as db
import menace.research_aggregator_bot as rab
import menace.model_automation_pipeline as mp
import menace.pre_execution_roi_bot as prb
import menace.code_database as cd
import menace.evolution_history_db as eh
import menace.self_improvement_policy as sip


class DummyBuilder(dm.ContextBuilder):
    def refresh_db_weights(self):
        return {}


vs.ContextBuilder = DummyBuilder


def _diag(edb, mdb):
    builder = DummyBuilder()
    return dm.DiagnosticManager(
        mdb, eb.ErrorBot(edb, mdb, context_builder=builder), context_builder=builder
    )


def test_run_cycle(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)
    builder = DummyBuilder()
    agg = rab.ResearchAggregatorBot(["menace"], info_db=info, context_builder=builder)
    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None,
                roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    pipe = StubPipeline()
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
    )
    mdb.add(db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1))
    edb.log_discrepancy("fail")
    res = engine.run_cycle()
    assert isinstance(res, mp.AutomationResult)


def test_run_cycle_triggers_workflow_evolution(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None,
                roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    pipe = StubPipeline()
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
    )
    monkeypatch.setattr(engine, "_should_trigger", lambda: True)
    monkeypatch.setattr(engine.pathway_db, "top_sequences", lambda limit=3: [])

    called: dict[str, object] = {}

    def fake_evolve(limit: int = 10):
        called["ran"] = True
        return {1: {"status": "promoted"}}

    monkeypatch.setattr(engine, "_evolve_workflows", fake_evolve)

    mdb.add(db.MetricRecord("bot", 5.0, 10.0, 3.0, 1.0, 1.0, 1))
    edb.log_discrepancy("fail")
    res = engine.run_cycle()
    assert called.get("ran") is True
    assert res.workflow_evolution[0]["workflow_id"] == 1


def test_run_all_cycles_respects_workflow_stability():
    class DummyEvolver:
        def __init__(self):
            self.calls = 0
            self.variant_calls = 0
            self.stable = False

        def build_callable(self, seq: str):
            return lambda: True

        def evolve(self, baseline, wf_id, variants=1):
            self.calls += 1
            if not self.stable:
                self.variant_calls += 1
                self.stable = True
            return baseline

        def is_stable(self, wf_id):
            return self.stable

    dummy = DummyEvolver()

    class DummyEngine:
        def __init__(self):
            self.workflow_evolver = dummy

        def _should_trigger(self):
            return True

        def _evaluate_workflow_variants(self, seq: str, wf_id: int) -> str:
            base = self.workflow_evolver.build_callable(seq)
            self.workflow_evolver.evolve(base, wf_id)
            return (
                "stable" if self.workflow_evolver.is_stable(wf_id) else "baseline"
            )

        def _evolve_workflows(self, limit: int = 10):
            status = self._evaluate_workflow_variants("a-b", 1)
            return {1: {"status": status}}

        def run_cycle(self, energy: int = 1):
            res = types.SimpleNamespace(workflow_evolution=None)
            evo = [
                {"workflow_id": wf, **summary}
                for wf, summary in self._evolve_workflows().items()
            ]
            res.workflow_evolution = evo
            return res

    engine = DummyEngine()
    reg = sie.ImprovementEngineRegistry()
    reg.register_engine("bot", engine)

    res1 = reg.run_all_cycles()
    assert dummy.calls == 1
    assert dummy.variant_calls == 1
    assert res1["bot"].workflow_evolution[0]["status"] == "stable"

    res2 = reg.run_all_cycles()
    assert dummy.calls == 2
    assert dummy.variant_calls == 1
    assert res2["bot"].workflow_evolution[0]["status"] == "stable"


def test_schedule_baseline_deviation(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return f"{model}:{energy}"

    class DummyCapitalBot:
        def __init__(self, energy: float) -> None:
            self.energy = energy

        def energy_score(self, **_: object) -> float:
            return self.energy

    pipe = StubPipeline()
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(0.2),
        baseline_margin=0.1,
    )

    engine.baseline_tracker = BaselineTracker(window=3, energy=[0.8])
    monkeypatch.setattr(engine, "_should_trigger", lambda: True)
    calls_low: list[int] = []
    monkeypatch.setattr(engine, "run_cycle", lambda energy=1: calls_low.append(energy))

    async def fake_sleep(_: float) -> None:
        raise SystemExit

    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run():
        task = engine.schedule()
        await task

    with pytest.raises(SystemExit):
        asyncio.run(run())
    assert calls_low == [int(round(0.2 * 5))]

    engine.capital_bot = DummyCapitalBot(0.8)
    engine.baseline_tracker = BaselineTracker(window=3, energy=[0.2])
    calls_high: list[int] = []
    monkeypatch.setattr(engine, "run_cycle", lambda energy=1: calls_high.append(energy))
    monkeypatch.setattr(sie.asyncio, "sleep", fake_sleep)

    async def run_high():
        task = engine.schedule()
        await task

    with pytest.raises(SystemExit):
        asyncio.run(run_high())
    assert calls_high == []


def test_schedule_deviation_autoruns(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return f"{model}:{energy}"

    class DummyCapitalBot:
        def __init__(self, energy: float) -> None:
            self.energy = energy

        def energy_score(self, **_: object) -> float:
            return self.energy

    pipe = StubPipeline()
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(0.9),
        baseline_margin=0.05,
    )

    engine.baseline_tracker = BaselineTracker(window=3, energy=[1.0])
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
    diag = _diag(edb, mdb)
    hist = eh.EvolutionHistoryDB(tmp_path / "h.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=None)

    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
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
            filename="a.py",  # path-ignore
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
            filename="b.py",  # path-ignore
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
    diag = _diag(edb, mdb)

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
    builder_high = DummyBuilder()
    eng_high = sie.SelfImprovementEngine(
        context_builder=builder_high,
        interval=0,
        pipeline=pipe_high,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        pre_roi_bot=HighROIBot(),
    )
    eng_high.run_cycle()

    pipe_low = StubPipeline()
    builder_low = DummyBuilder()
    eng_low = sie.SelfImprovementEngine(
        context_builder=builder_low,
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
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=None)

    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
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
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None,
                roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
            )

    path = tmp_path / "policy.pkl"
    policy = sip.SelfImprovementPolicy(path=path)
    monkeypatch.setattr(sie, "bootstrap", lambda: 0)
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
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
    builder2 = DummyBuilder()
    engine2 = sie.SelfImprovementEngine(
        context_builder=builder2,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        policy=policy2,
    )
    assert engine2.policy.score(state) == val


class _DummyTracker:
    def __init__(self, roi=None, eff=None, res=None, af=None):
        self.metrics_history = {}
        if roi is not None:
            self.metrics_history["synergy_roi"] = roi
        if eff is not None:
            self.metrics_history["synergy_efficiency"] = eff
        if res is not None:
            self.metrics_history["synergy_resilience"] = res
        if af is not None:
            self.metrics_history["synergy_antifragility"] = af


def test_synergy_energy_scaling(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def __init__(self) -> None:
            self.energy = None

        def run(self, model: str, energy: int = 1):
            self.energy = energy
            return mp.AutomationResult(package=None, roi=None)

    class DummyCapitalBot:
        def energy_score(self, **_: object) -> float:
            return 0.5

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    base_pipe = StubPipeline()
    base_builder = DummyBuilder()
    base_engine = sie.SelfImprovementEngine(
        context_builder=base_builder,
        interval=0,
        pipeline=base_pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
    )
    base_engine.tracker = _DummyTracker([0.0])
    base_engine.run_cycle()
    base_energy = base_pipe.energy

    pos_pipe = StubPipeline()
    pos_builder = DummyBuilder()
    pos_engine = sie.SelfImprovementEngine(
        context_builder=pos_builder,
        interval=0,
        pipeline=pos_pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
    )
    pos_engine.tracker = _DummyTracker([0.0, 0.2], eff=[0.0, 0.3])
    pos_engine.run_cycle()
    assert pos_pipe.energy > base_energy


def test_synergy_energy_cap(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def __init__(self) -> None:
            self.energy = None

        def run(self, model: str, energy: int = 1):
            self.energy = energy
            return mp.AutomationResult(package=None, roi=None)

    class DummyCapitalBot:
        def energy_score(self, **_: object) -> float:
            return 0.5

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    pipe = StubPipeline()
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
    )
    engine.tracker = _DummyTracker([0.0, 50.0], eff=[0.0, 50.0])
    engine.run_cycle()
    assert pipe.energy == 100


def test_policy_update_receives_synergy_deltas(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=None)

    class DummyPolicy(sip.SelfImprovementPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.deltas: tuple[float | None, float | None] | None = None

        def update(
            self,
            state: tuple[int, ...],
            reward: float,
            next_state: tuple[int, ...] | None = None,
            action: int = 1,
            *,
            synergy_roi_delta: float | None = None,
            synergy_efficiency_delta: float | None = None,
        ) -> float:
            self.deltas = (synergy_roi_delta, synergy_efficiency_delta)
            return super().update(
                state,
                reward,
                next_state,
                action,
                synergy_roi_delta=synergy_roi_delta,
                synergy_efficiency_delta=synergy_efficiency_delta,
            )

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    policy = DummyPolicy()
    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        policy=policy,
    )

    engine.tracker = _DummyTracker([0.0, 0.2], eff=[0.0, 0.3])
    engine.run_cycle()

    assert policy.deltas == (0.2, 0.3)


def test_roi_history_group_ids(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0))

    class DummyCapitalBot:
        def __init__(self) -> None:
            self.val = 0.0

        def energy_score(self, **_: object) -> float:
            return 1.0

        def profit(self) -> float:
            v = self.val
            self.val += 1.0
            return v

    class DummyPatchDB:
        def _connect(self):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE patch_history(id INTEGER PRIMARY KEY, filename TEXT, roi_delta REAL, complexity_delta REAL, entropy_delta REAL, reverted INTEGER)"
            )
            conn.execute(
                "INSERT INTO patch_history(filename, roi_delta, complexity_delta, entropy_delta, reverted) VALUES ('a.py', 1.0, 0.0, 0.0, 0)"  # path-ignore
            )
            conn.commit()
            import contextlib

            @contextlib.contextmanager
            def ctx():
                try:
                    yield conn
                finally:
                    conn.close()

            return ctx()

    patch_db = DummyPatchDB()

    from module_index_db import ModuleIndexDB

    module_index = ModuleIndexDB(tmp_path / "map.json")

    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        patch_db=patch_db,
        module_index=module_index,
        module_groups={"a.py": "core"},  # path-ignore
    )

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    engine.run_cycle()

    gid = module_index.group_id("core")
    assert engine.roi_group_history.get(gid) == [1.0]



def test_module_map_refresh_updates_roi_groups(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "old.py").write_text("print('old')")  # path-ignore
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {"old.py": 0}, "groups": {"0": 0}}))  # path-ignore
    os.utime(map_path, (0, 0))

    class DummyPatchDB:
        def _connect(self):
            import sqlite3
            from datetime import datetime
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE patch_history(id INTEGER PRIMARY KEY, filename TEXT, roi_delta REAL, complexity_delta REAL, entropy_delta REAL, reverted INTEGER, ts TEXT)"
            )
            conn.execute(
                "INSERT INTO patch_history(filename, roi_delta, complexity_delta, entropy_delta, reverted, ts) VALUES ('new.py', 0.0, 0.0, 0.0, 0, ?)",  # path-ignore
                (datetime.utcnow().isoformat(),),
            )
            conn.commit()
            import contextlib

            @contextlib.contextmanager
            def ctx():
                try:
                    yield conn
                finally:
                    conn.close()

            return ctx()

    patch_db = DummyPatchDB()

    from module_index_db import ModuleIndexDB

    module_index = ModuleIndexDB(map_path)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))

    def fake_build(repo_path, **kw):
        assert Path(repo_path) == repo
        return {"old.py": 0, "new.py": 1}  # path-ignore

    monkeypatch.setattr(sie, "build_module_map", fake_build)

    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0))

    class DummyCapitalBot:
        def __init__(self) -> None:
            self.val = 0.0
        def energy_score(self, **_: object) -> float:
            return 1.0
        def profit(self) -> float:
            v = self.val
            self.val += 1.0
            return v

    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        patch_db=patch_db,
        module_index=module_index,
        auto_refresh_map=True,
    )

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    engine.run_cycle()
    gid = module_index.get("new.py")  # path-ignore
    assert gid == 1
    assert engine.roi_group_history.get(gid) == [1.0]

    engine.run_cycle()
    assert engine.roi_group_history.get(gid) == [1.0, 1.0]


def test_init_refresh_called_for_unknown_patches(tmp_path, monkeypatch):
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {"old.py": 0}, "groups": {"0": 0}}))  # path-ignore
    class DummyPatchDB:
        def _connect(self):
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE patch_history(id INTEGER PRIMARY KEY, filename TEXT)")
            conn.execute("INSERT INTO patch_history(filename) VALUES('new.py')")  # path-ignore
            conn.commit()
            import contextlib
            @contextlib.contextmanager
            def ctx():
                try:
                    yield conn
                finally:
                    conn.close()
            return ctx()

    patch_db = DummyPatchDB()
    from module_index_db import ModuleIndexDB
    module_index = ModuleIndexDB(map_path)
    called = {}
    def fake_refresh(mods=None, *, force=False):
        called["mods"] = list(mods or [])
        called["force"] = force
    monkeypatch.setattr(module_index, "refresh", fake_refresh)
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)
    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=prb.ROIResult(0.0, 0.0, 0.0, 0.0, 0.0))
    builder = DummyBuilder()
    sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        patch_db=patch_db,
        module_index=module_index,
    )
    assert called.get("mods") == ["new.py"]  # path-ignore


def test_init_discovers_module_groups(tmp_path, monkeypatch):
    class DummyPatchDB:
        def _connect(self):
            import sqlite3

            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE patch_history(id INTEGER PRIMARY KEY, filename TEXT, roi_delta REAL, complexity_delta REAL, entropy_delta REAL, reverted INTEGER)"
            )
            conn.execute(
                "INSERT INTO patch_history(filename, roi_delta, complexity_delta, entropy_delta, reverted) VALUES ('a.py', 1.0, 0.0, 0.0, 0)"  # path-ignore
            )
            conn.commit()

            import contextlib

            @contextlib.contextmanager
            def ctx():
                try:
                    yield conn
                finally:
                    conn.close()

            return ctx()

    patch_db = DummyPatchDB()

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    def fake_discover(path):
        assert Path(path) == tmp_path
        return {"core": ["a"]}

    monkeypatch.setattr(sie, "discover_module_groups", fake_discover)

    from module_index_db import ModuleIndexDB

    module_index = ModuleIndexDB(tmp_path / "map.json")

    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(package=None, roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0))

    class DummyCapitalBot:
        def __init__(self) -> None:
            self.val = 0.0

        def energy_score(self, **_: object) -> float:
            return 1.0

        def profit(self) -> float:
            v = self.val
            self.val += 1.0
            return v

    builder = DummyBuilder()
    engine = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info,
        capital_bot=DummyCapitalBot(),
        patch_db=patch_db,
        module_index=module_index,
    )

    monkeypatch.setattr(sie, "bootstrap", lambda: 0)

    engine.run_cycle()

    gid = engine.module_clusters.get("a.py")  # path-ignore
    assert gid is not None
    assert engine.roi_group_history.get(gid) == [1.0]
    data = json.loads((tmp_path / "map.json").read_text())
    assert data["modules"].get("a.py") == gid  # path-ignore


def test_update_orphan_modules_nested_dependencies(tmp_path, monkeypatch):
    import types

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import b\n")  # path-ignore
    (repo / "b.py").write_text("import c\n")  # path-ignore
    (repo / "c.py").write_text("x = 1\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    eng = types.SimpleNamespace(
        orphan_traces={"c.py": {"parents": ["b.py"], "redundant": True}},  # path-ignore
        module_index=None,
        module_clusters={},
        logger=types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None),
    )
    eng._collect_recursive_modules = types.MethodType(
        sie.SelfImprovementEngine._collect_recursive_modules, eng
    )

    integrated: list[str] = []

    def fake_integrate(paths: list[str]) -> set[str]:
        integrated.extend(sorted(Path(p).name for p in paths))
        data = json.loads(map_path.read_text())
        for p in paths:
            data["modules"][Path(p).name] = 1
        map_path.write_text(json.dumps(data))
        return {Path(p).name for p in paths}

    eng._integrate_orphans = fake_integrate
    eng._refresh_module_map = lambda modules=None: None

    def fake_test(mods: list[str]) -> list[str]:
        return [m for m in mods if not eng.orphan_traces.get(m, {}).get("redundant")]

    eng._test_orphan_modules = fake_test

    calls: list[list[str]] = []
    env = types.SimpleNamespace(
        auto_include_modules=lambda mods, recursive=False, validate=False: calls.append(
            sorted(mods)
        )
    )
    monkeypatch.setattr(sie, "environment", env)
    import sandbox_runner
    monkeypatch.setattr(
        sandbox_runner, "discover_recursive_orphans", lambda repo, module_map=None: {}
    )

    eng._update_orphan_modules = types.MethodType(
        sie.SelfImprovementEngine._update_orphan_modules, eng
    )
    eng._update_orphan_modules(["a.py"])  # path-ignore

    assert calls[0] == ["a.py", "b.py"]  # path-ignore
    assert integrated[:2] == ["a.py", "b.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"a.py", "b.py"}  # path-ignore
    assert "c.py" not in data["modules"]  # path-ignore
    orphan_file = data_dir / "orphan_modules.json"
    assert orphan_file.exists()
    assert eng.orphan_traces.get("c.py", {}).get("redundant") is True  # path-ignore
    assert eng.orphan_traces["b.py"]["parents"] == ["a.py"]  # path-ignore
    assert eng.orphan_traces["c.py"]["parents"] == ["b.py"]  # path-ignore

class DummyPredictor:
    def __init__(self, mapping):
        self.mapping = mapping

    def predict(self, feats, horizon=None):
        key = feats[0][0]
        seq, category = self.mapping[key]
        conf = [1.0] * len(seq)
        return seq, category, conf, None


def test_score_modifications_prefers_higher_roi() -> None:
    mapping = {1.0: ([1.5], "linear"), 0.0: ([0.5], "linear")}
    predictor = DummyPredictor(mapping)
    eng = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    eng.roi_predictor = predictor
    eng.use_adaptive_roi = True
    eng.growth_weighting = True
    eng.growth_multipliers = {"linear": 1.0}
    eng._candidate_features = (
        lambda name: [[1.0] + [0.0] * 14] if name == "high" else [[0.0] * 15]
    )
    ranked = eng._score_modifications(["low", "high"])
    assert [r[0] for r in ranked] == ["high", "low"]
    assert ranked[0][1] > ranked[1][1]
    assert ranked[0][3] > ranked[1][3]


def test_deployment_governor_promote_marks_workflow_ready(
    tmp_path, monkeypatch
) -> None:
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None, roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0)
            )

    pipe = StubPipeline()
    engine = sie.SelfImprovementEngine(
        interval=0, pipeline=pipe, diagnostics=diag, info_db=info
    )
    monkeypatch.setattr(engine.roi_tracker, "generate_scorecards", lambda: [])
    engine.roi_tracker.last_raroi = 0.2
    engine.roi_tracker.confidence_history = [0.9]
    engine.deployment_governor = types.SimpleNamespace(
        evaluate=lambda *a, **k: {
            "verdict": "promote",
            "reasons": ["ok"],
            "overrides": {},
        }
    )

    engine.run_cycle()
    assert engine.workflow_ready is True


def test_deployment_governor_pilot_enqueues_borderline(
    tmp_path, monkeypatch
) -> None:
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mp.AutomationResult(
                package=None, roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0)
            )

    pipe = StubPipeline()
    engine = sie.SelfImprovementEngine(
        interval=0, pipeline=pipe, diagnostics=diag, info_db=info
    )
    monkeypatch.setattr(engine.roi_tracker, "generate_scorecards", lambda: [])
    engine.roi_tracker.last_raroi = 0.1
    engine.roi_tracker.confidence_history = [0.8]

    calls: dict[str, object] = {}

    class DummyBucket:
        def add_candidate(self, wf, raroi, conf, reason):
            calls["candidate"] = (wf, raroi, conf, reason)

        def process(self, *a, **k):
            calls["processed"] = True

    bucket = DummyBucket()
    engine.borderline_bucket = bucket
    engine.roi_tracker.borderline_bucket = bucket

    engine.deployment_governor = types.SimpleNamespace(
        evaluate=lambda *a, **k: {
            "verdict": "pilot",
            "reasons": ["borderline"],
            "overrides": {},
        }
    )

    engine.run_cycle()
    assert "candidate" in calls and calls["candidate"][0] == engine.bot_name


class _StubPipeline:
    def run(self, model: str, energy: int = 1):
        return mp.AutomationResult(
            package=None,
            roi=prb.ROIResult(1.0, 0.0, 0.0, 1.0, 0.0),
        )


class _ROITracker:
    def __init__(self):
        self.confidence_history = [0.9]
        self.last_raroi = 1.0
        self.confidence_threshold = 0.0

    def update(self, *a, **k):
        pass

    def record_roi_prediction(self, *a, **k):
        pass

    def generate_scorecards(self):
        class Card:
            raroi_delta = 1.0
        return [Card()]


class _ForesightTracker:
    def predict_roi_collapse(self, workflow_id):
        return {"risk": "Stable", "brittle": False}


class _Bucket:
    def __init__(self):
        self.candidates = []

    def add_candidate(self, *a):
        self.candidates.append(a)

    def process(self, *a, **k):
        pass


def test_deployment_gate_promotes(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=_StubPipeline(),
        diagnostics=diag,
        info_db=info,
        roi_tracker=_ROITracker(),
        foresight_tracker=_ForesightTracker(),
        borderline_bucket=_Bucket(),
    )
    events: list[dict] = []
    monkeypatch.setattr(
        sie,
        "deployment_evaluate",
        lambda *a, **k: {
            "verdict": "promote",
            "reasons": [],
            "foresight": {"reason_codes": [], "forecast_id": "fid"},
        },
    )
    monkeypatch.setattr(sie, "audit_log_event", lambda name, payload: events.append(payload))
    monkeypatch.setattr(sie, "SandboxSettings", lambda: types.SimpleNamespace(micropilot_mode=""))
    engine.run_cycle()
    assert engine.workflow_ready is True
    assert engine.borderline_bucket.candidates == []
    assert events and events[-1]["verdict"] == "promote"


def test_deployment_gate_borderline(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)
    bucket = _Bucket()
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=_StubPipeline(),
        diagnostics=diag,
        info_db=info,
        roi_tracker=_ROITracker(),
        foresight_tracker=_ForesightTracker(),
        borderline_bucket=bucket,
    )
    events: list[dict] = []
    monkeypatch.setattr(sie, "deployment_evaluate", lambda *a, **k: {"verdict": "borderline", "reasons": ["low_confidence"], "foresight": {"reason_codes": ["low_confidence"]}})
    monkeypatch.setattr(sie, "audit_log_event", lambda name, payload: events.append(payload))
    monkeypatch.setattr(sie, "SandboxSettings", lambda: types.SimpleNamespace(micropilot_mode=""))
    engine.run_cycle()
    assert engine.workflow_ready is False
    assert bucket.candidates and bucket.candidates[0][-1] == "low_confidence"
    assert events and events[-1]["verdict"] == "borderline"
    assert events[-1]["downgrade_type"] == "borderline"


def test_deployment_gate_pilot(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = _diag(edb, mdb)
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=_StubPipeline(),
        diagnostics=diag,
        info_db=info,
        roi_tracker=_ROITracker(),
        foresight_tracker=_ForesightTracker(),
        borderline_bucket=None,
    )
    events: list[dict] = []
    monkeypatch.setattr(sie, "deployment_evaluate", lambda *a, **k: {"verdict": "pilot", "reasons": ["low_confidence"], "foresight": {"reason_codes": ["low_confidence"]}})
    monkeypatch.setattr(sie, "audit_log_event", lambda name, payload: events.append(payload))
    monkeypatch.setattr(sie, "SandboxSettings", lambda: types.SimpleNamespace(micropilot_mode=""))
    engine.run_cycle()
    assert engine.workflow_ready is False
    assert events and events[-1]["verdict"] == "pilot"
    assert events[-1]["downgrade_type"] == "pilot"


def test_compute_delta_score_weights():
    if not hasattr(sie, "SelfImprovementEngine"):
        pytest.skip("SelfImprovementEngine unavailable")
    eng = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    eng.entropy_dev_multiplier = 0.0
    eng.pass_rate_dev_multiplier = 0.0
    eng.baseline_tracker = types.SimpleNamespace(
        std=lambda metric: 0.2 if metric == "entropy" else 0.0,
        get=lambda name: 0.0,
        delta_history=lambda m: [1.0]
        if m == "roi"
        else [0.3]
        if m == "pass_rate"
        else [0.5]
        if m == "entropy"
        else [],
        momentum=0.2,
        update=lambda record_momentum=True, **kw: None,
    )
    eng.roi_weight = 2.0
    eng.momentum_weight = 3.0
    eng.entropy_weight = 4.0
    eng.pass_rate_weight = 5.0
    eng.entropy_weight_scale = 1.0
    eng.momentum_weight_scale = 1.0
    score, components = sie.SelfImprovementEngine._compute_delta_score(eng)
    assert score == pytest.approx(2.0 * 1.0 + 5.0 * 0.3 + 3.6 * 0.2 - 4.8 * 0.5)
    assert components == {
        "roi_delta": 1.0,
        "pass_rate_delta": 0.3,
        "entropy_delta": 0.5,
        "momentum_delta": 0.2,
        "roi_component": 2.0,
        "pass_rate_component": 1.5,
        "momentum_component": pytest.approx(0.72),
        "entropy_component": pytest.approx(-2.4),
    }
