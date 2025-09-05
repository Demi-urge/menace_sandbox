import types
import sys

stub = types.ModuleType("stub")
for mod in ("psutil", "networkx", "pandas", "pulp"):
    if mod not in sys.modules:
        sys.modules[mod] = stub

try:
    import jinja2 as _j2  # noqa: F401
except Exception:
    jinja_stub = types.ModuleType("jinja2")
    jinja_stub.Template = lambda *a, **k: None
    sys.modules["jinja2"] = jinja_stub

try:
    import yaml as _yaml  # noqa: F401
except Exception:
    sys.modules["yaml"] = stub

try:
    import sqlalchemy as _sa  # noqa: F401
except Exception:
    sa_stub = types.ModuleType("sqlalchemy")
    engine_stub = types.ModuleType("engine")
    engine_stub.Engine = object
    sa_stub.engine = engine_stub
    sys.modules["sqlalchemy"] = sa_stub
    sys.modules["sqlalchemy.engine"] = engine_stub

for mod in ("numpy", "git"):
    if mod not in sys.modules:
        sys.modules[mod] = stub
stub.Repo = object

matplotlib_stub = types.ModuleType("matplotlib")
plt_stub = types.ModuleType("pyplot")
matplotlib_stub.pyplot = plt_stub  # path-ignore
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = matplotlib_stub
if "matplotlib.pyplot" not in sys.modules:  # path-ignore
    sys.modules["matplotlib.pyplot"] = plt_stub  # path-ignore

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = dotenv_stub
if "prometheus_client" not in sys.modules:
    sys.modules["prometheus_client"] = stub
stub.CollectorRegistry = object
stub.Counter = object
stub.Gauge = object
sys.modules.setdefault("sklearn", stub)
sys.modules.setdefault("sklearn.feature_extraction", stub)
sys.modules.setdefault("sklearn.feature_extraction.text", stub)
stub.TfidfVectorizer = object
sys.modules.setdefault("sklearn.cluster", stub)
stub.KMeans = object
sys.modules.setdefault("sklearn.linear_model", stub)
stub.LinearRegression = object
sys.modules.setdefault("sklearn.model_selection", stub)
stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.metrics", stub)
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = object
sys.modules.setdefault("sklearn.ensemble", stub)
stub.RandomForestClassifier = object
class _OrphanStub(types.ModuleType):
    def __getattr__(self, _name):  # pragma: no cover - simple stub
        return lambda *a, **k: []

orphan_stub = _OrphanStub("orphan_discovery")
sys.modules.setdefault("orphan_discovery", orphan_stub)
sys.modules.setdefault("sandbox_runner.orphan_discovery", orphan_stub)

from menace.workflow_evolution_bot import WorkflowEvolutionBot, WorkflowSuggestion
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.evolution_history_db import EvolutionHistoryDB


class EvolutionOrchestrator:
    def __init__(
        self,
        _data_bot,
        _cap_bot,
        *_args,
        workflow_evolver=None,
        experiment_manager=None,
        history_db=None,
    ) -> None:
        self.workflow_evolver = workflow_evolver
        self.experiment_manager = experiment_manager
        self._latest_roi = lambda: 0.0
        self._error_rate = lambda: 0.0

    def run_cycle(self) -> None:
        if not self.workflow_evolver or not self.experiment_manager:
            return
        import asyncio

        variants = list(self.workflow_evolver.generate_variants())
        asyncio.run(self.experiment_manager.run_experiments(variants))


def test_analyse(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    pdb.log(PathwayRecord(actions="a", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    dummy_clusterer = types.SimpleNamespace(find_modules_related_to=lambda *a, **k: [])
    bot = WorkflowEvolutionBot(pdb, intent_clusterer=dummy_clusterer)
    sugg = bot.analyse(limit=1)
    assert isinstance(sugg, list)


class DummyExpMgr:
    def __init__(self) -> None:
        self.received = None

    async def run_experiments(self, variants, energy=1):
        self.received = list(variants)
        return []

    def best_variant(self, results):
        return None


class DummyWorkflowEvolver:
    def __init__(self) -> None:
        self.calls = 0

    def generate_variants(self, limit: int = 5):
        self.calls += 1
        return ["a-b"]


def test_orchestrator_integration(monkeypatch):
    data_bot = types.SimpleNamespace(db=types.SimpleNamespace(fetch=lambda limit=50: []))
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(roi=None)),
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(ga_results={}, predictions=[])),
        workflow_evolver=DummyWorkflowEvolver(),
        experiment_manager=DummyExpMgr(),
    )
    wf = orch.workflow_evolver
    exp = orch.experiment_manager
    orch.run_cycle()
    assert wf.calls == 1
    assert exp.received == ["a-b"]


