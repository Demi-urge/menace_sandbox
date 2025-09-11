import types
import sys
import pytest

stub = types.ModuleType("stub")
for mod in ("psutil", "networkx", "pandas", "pulp", "scipy"):
    if mod not in sys.modules:
        sys.modules[mod] = stub
stub.stats = object
stub.DiGraph = object
stub.log2 = lambda *a, **k: 0.0
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
git_exc = types.ModuleType("git.exc")
git_exc.GitCommandError = Exception
git_exc.InvalidGitRepositoryError = Exception
git_exc.NoSuchPathError = Exception
sys.modules["git.exc"] = git_exc

matplotlib_stub = types.ModuleType("matplotlib")
plt_stub = types.ModuleType("pyplot")
matplotlib_stub.pyplot = plt_stub  # path-ignore
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = matplotlib_stub
if "matplotlib.pyplot" not in sys.modules:  # path-ignore
    sys.modules["matplotlib.pyplot"] = plt_stub  # path-ignore

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
dotenv_stub.dotenv_values = lambda *a, **k: {}
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
stub.LinearRegression = type("LinearRegression", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("sklearn.model_selection", stub)
stub.train_test_split = lambda *a, **k: ([], [])
sys.modules.setdefault("sklearn.metrics", stub)
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = type("LogisticRegression", (), {"__init__": lambda self, *a, **k: None})
sys.modules.setdefault("sklearn.ensemble", stub)
stub.RandomForestClassifier = type("RandomForestClassifier", (), {"__init__": lambda self, *a, **k: None})

error_bot_stub = types.ModuleType("menace.error_bot")
error_bot_stub.ErrorDB = object
sys.modules.setdefault("menace.error_bot", error_bot_stub)
aem_stub = types.ModuleType("menace.advanced_error_management")
aem_stub.FormalVerifier = object
aem_stub.AutomatedRollbackManager = object
sys.modules.setdefault("menace.advanced_error_management", aem_stub)
rm_stub = types.ModuleType("menace.rollback_manager")
rm_stub.RollbackManager = object
sys.modules.setdefault("menace.rollback_manager", rm_stub)
mutation_logger_stub = types.ModuleType("menace.mutation_logger")
mutation_logger_stub.log_mutation = lambda *a, **k: None
sys.modules.setdefault("menace.mutation_logger", mutation_logger_stub)

sr_pkg = types.ModuleType("menace.sandbox_runner")
th_stub = types.ModuleType("menace.sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th_stub.TestHarnessResult = types.SimpleNamespace
sr_pkg.test_harness = th_stub
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)

from menace.evolution_orchestrator import EvolutionOrchestrator, EvolutionTrigger
from menace.evolution_history_db import EvolutionHistoryDB


def make_orchestrator(tmp_path, energy, analysis_predict, seq_predict=None):
    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        log_evolution_cycle=lambda *a, **k: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: energy)
    improver = types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(roi=None))
    evolver = types.SimpleNamespace(run_cycle=lambda: types.SimpleNamespace(ga_results={}, predictions=[]))
    analysis = types.SimpleNamespace(predict=analysis_predict, train=lambda: None)
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=hist,
        analysis_bot=analysis,
        multi_predictor=seq_predict,
    )
    return orch, hist


def test_predicted_roi_selects_best(monkeypatch, tmp_path):
    def predict(action, _metric):
        return 2.0 if action == "system_evolution" else 1.0

    orch, hist = make_orchestrator(tmp_path, 0.1, predict)
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    orch.prev_roi = 1.0
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)
    orch.run_cycle()
    rows = hist.fetch()
    assert rows and rows[0][0] == "system_evolution"


def test_near_threshold_uses_prediction(monkeypatch, tmp_path):
    def predict(action, _metric):
        return 0.5 if action == "self_improvement" else 0.4

    trig = EvolutionTrigger(error_rate=0.1, roi_drop=-0.1, energy_threshold=0.3)
    orch, hist = make_orchestrator(tmp_path, 0.31, predict)
    orch.triggers = trig
    monkeypatch.setattr(orch, "_latest_roi", lambda: 1.0)
    orch.prev_roi = 1.05
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.095)
    orch.run_cycle()
    rows = hist.fetch()
    assert rows and rows[0][0] == "self_improvement"


def test_ensemble_predictor(monkeypatch, tmp_path):
    def predict(_action, _metric):
        return 1.0

    class EnsPred:
        def __init__(self) -> None:
            self.trained = 0

        def predict(self, action, metric):
            if action == "self_improvement":
                return 0.6, 0.1
            return 0.6, 0.2

        def train(self):
            self.trained += 1

    ens = EnsPred()
    orch, hist = make_orchestrator(tmp_path, 0.1, predict, seq_predict=ens)
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    orch.prev_roi = 1.0
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)
    orch.run_cycle()
    rows = hist.fetch()
    assert rows and rows[0][0] == "self_improvement"
    assert ens.trained == 1


def test_reason_and_parent_lineage(monkeypatch, tmp_path):
    def predict(_action, _metric):
        return 0.0

    orch, hist = make_orchestrator(tmp_path, 0.5, predict)
    monkeypatch.setattr(orch, "_latest_roi", lambda: 1.0)
    orch.prev_roi = 1.0
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.5)

    orch.run_cycle()
    first_id = hist.conn.execute("SELECT max(rowid) FROM evolution_history").fetchone()[0]
    row1 = hist.conn.execute(
        'SELECT reason, "trigger", parent_event_id FROM evolution_history WHERE rowid=?',
        (first_id,),
    ).fetchone()
    assert "error_rate" in row1[0]
    assert "error_rate" in row1[1]
    assert row1[2] is None

    orch.run_cycle()
    second_id = hist.conn.execute("SELECT max(rowid) FROM evolution_history").fetchone()[0]
    row2 = hist.conn.execute(
        'SELECT parent_event_id FROM evolution_history WHERE rowid=?', (second_id,)
    ).fetchone()
    assert row2[0] == first_id


def test_on_bot_degraded_invokes_manager(monkeypatch, tmp_path):
    mod_path = tmp_path / "dmod.py"
    mod_path.write_text("pass\n")
    mod_name = "dmod"
    module = types.ModuleType(mod_name)
    module.__file__ = str(mod_path)
    sys.modules[mod_name] = module

    calls: list[tuple] = []

    class Manager:
        bot_name = "bot"

        def __init__(self) -> None:
            self.event_bus = types.SimpleNamespace(publish=lambda *a, **k: None)

        def generate_and_patch(self, path, desc, **kwargs):
            calls.append((path, desc, kwargs))

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    improver = types.SimpleNamespace()
    evolver = types.SimpleNamespace()
    manager = Manager()
    hist_stub = types.SimpleNamespace(add=lambda *a, **k: None)
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=hist_stub,
        selfcoding_manager=manager,
    )

    orch._on_bot_degraded({"bot": mod_name})
    assert calls and calls[0][0] == mod_path


def test_on_bot_degraded_generation_failure(tmp_path):
    mod_path = tmp_path / "dmod_fail.py"
    mod_path.write_text("pass\n")
    mod_name = "dmod_fail"
    module = types.ModuleType(mod_name)
    module.__file__ = str(mod_path)
    sys.modules[mod_name] = module

    published: list[tuple] = []

    class Bus:
        def publish(self, topic, event):
            published.append((topic, event))

    from menace.self_coding_manager import HelperGenerationError

    class Manager:
        bot_name = "bot"

        def __init__(self) -> None:
            self.event_bus = Bus()

        def generate_and_patch(self, *a, **k):
            raise HelperGenerationError("boom")

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
    )
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 1.0)
    improver = types.SimpleNamespace()
    evolver = types.SimpleNamespace()
    manager = Manager()
    hist_stub = types.SimpleNamespace(add=lambda *a, **k: None)
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        evolver,
        history_db=hist_stub,
        selfcoding_manager=manager,
    )

    orch._on_bot_degraded({"bot": mod_name})
    assert any(t == "bot:patch_failed" and e["stage"] == "context" for t, e in published)

