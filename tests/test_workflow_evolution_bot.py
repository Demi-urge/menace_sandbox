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
matplotlib_stub.pyplot = plt_stub
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = matplotlib_stub
if "matplotlib.pyplot" not in sys.modules:
    sys.modules["matplotlib.pyplot"] = plt_stub

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

from menace.workflow_evolution_bot import WorkflowEvolutionBot, WorkflowSuggestion
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.evolution_orchestrator import EvolutionOrchestrator
from menace.evolution_history_db import EvolutionHistoryDB


def test_analyse(tmp_path):
    pdb = PathwayDB(tmp_path / "p.db")
    pdb.log(PathwayRecord(actions="a", inputs="", outputs="", exec_time=1.0, resources="", outcome=Outcome.SUCCESS, roi=1.0))
    bot = WorkflowEvolutionBot(pdb)
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

    def propose_rearrangements(self, limit: int = 5):
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
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.0)
    wf = orch.workflow_evolver
    exp = orch.experiment_manager
    orch.run_cycle()
    assert wf.calls == 1
    assert exp.received == ["a-b"]


def test_orchestrator_logs_cycle(monkeypatch, tmp_path):
    calls = []
    cap_calls = []
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        log_evolution_cycle=lambda *a, **k: calls.append(a),
    )
    cap_bot = types.SimpleNamespace(
        energy_score=lambda **k: 1.0,
        log_evolution_event=lambda *a, **k: cap_calls.append(a),
    )
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(roi=None)),
        types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(ga_results={}, predictions=[])),
        history_db=hist,
    )
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)
    orch.run_cycle()
    rows = hist.fetch()
    assert calls and calls[0][0] == "self_improvement"
    assert cap_calls and cap_calls[0][0] == "self_improvement"
    assert rows and rows[0][3] == calls[0][3]


