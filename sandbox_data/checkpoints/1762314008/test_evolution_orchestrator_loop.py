import types
import sys

import pytest


class DummyDB:
    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, limit: int = 50):
        self.calls += 1
        if self.calls <= 3:
            return [{"bot": "m", "revenue": 1.0, "expense": 0.0, "errors": 5, "cpu": 1.0}]
        return [{"bot": "m", "revenue": 3.0, "expense": 0.0, "errors": 0, "cpu": 1.0}]


class DummyDataBot:
    def __init__(self, hist, db: DummyDB) -> None:
        self.db = db
        self.hist = hist
        self.logged: list[tuple[str, dict]] = []

    def log_evolution_cycle(
        self,
        action: str,
        before: float,
        after: float,
        roi: float,
        predicted_roi: float = 0.0,
        **kwargs,
    ) -> None:
        from menace.evolution_history_db import EvolutionEvent

        self.logged.append((action, kwargs))
        self.hist.add(
            EvolutionEvent(
                action=action,
                before_metric=before,
                after_metric=after,
                roi=roi,
                predicted_roi=predicted_roi,
                patch_id=kwargs.get("patch_id"),
                workflow_id=kwargs.get("workflow_id"),
                trending_topic=kwargs.get("trending_topic"),
            )
        )


def test_full_loop(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "jinja2",
        sys.modules.get("jinja2")
        or types.SimpleNamespace(Template=lambda *a, **k: None),
    )
    monkeypatch.setitem(sys.modules, "yaml", sys.modules.get("yaml") or types.ModuleType("yaml"))
    sa_stub = types.ModuleType("sqlalchemy")
    engine_stub = types.ModuleType("sqlalchemy.engine")
    engine_stub.Engine = object
    sa_stub.engine = engine_stub
    monkeypatch.setitem(sys.modules, "sqlalchemy", sa_stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", engine_stub)
    stub = types.ModuleType("stub")
    stub.stats = object
    stub.DiGraph = object
    stub.log2 = lambda *a, **k: 0.0
    for mod in ("psutil", "networkx", "pandas", "pulp", "scipy"):
        monkeypatch.setitem(sys.modules, mod, stub)
    for mod in ("numpy", "git"):
        monkeypatch.setitem(sys.modules, mod, stub)
    stub.Repo = object
    git_exc = types.ModuleType("git.exc")
    git_exc.GitCommandError = Exception
    git_exc.InvalidGitRepositoryError = Exception
    git_exc.NoSuchPathError = Exception
    monkeypatch.setitem(sys.modules, "git.exc", git_exc)
    matplotlib_stub = types.ModuleType("matplotlib")
    plt_stub = types.ModuleType("pyplot")
    matplotlib_stub.pyplot = plt_stub  # path-ignore
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)  # path-ignore
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(
            load_dotenv=lambda *a, **k: None, dotenv_values=lambda *a, **k: {}
        ),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", stub)
    stub.CollectorRegistry = object
    stub.Counter = object
    stub.Gauge = object
    monkeypatch.setitem(sys.modules, "sklearn", stub)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction", stub)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", stub)
    stub.TfidfVectorizer = object
    monkeypatch.setitem(sys.modules, "sklearn.cluster", stub)
    stub.KMeans = object
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", stub)
    stub.LinearRegression = object
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", stub)
    stub.train_test_split = lambda *a, **k: ([], [])
    monkeypatch.setitem(sys.modules, "sklearn.metrics", stub)
    stub.accuracy_score = lambda *a, **k: 0.0
    stub.LogisticRegression = object
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", stub)
    stub.RandomForestClassifier = object

    from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
    from menace.evolution_orchestrator import EvolutionOrchestrator, EvolutionTrigger

    hist = EvolutionHistoryDB(tmp_path / "e.db")
    db = DummyDB()
    data_bot = DummyDataBot(hist, db)
    cap_bot = types.SimpleNamespace(energy_score=lambda **k: 0.05)

    class Improver:
        def run_cycle(self):
            data_bot.log_evolution_cycle("self_improvement", 1.0, 1.5, 0.5, patch_id=42)
            return types.SimpleNamespace(roi=types.SimpleNamespace(roi=0.5), trending_topic="trend")

    class Evolver:
        def __init__(self) -> None:
            self.bots = ["m"]

        def run_cycle(self):
            data_bot.log_evolution_cycle("system_evolution", 1.5, 2.0, 0.6)
            return types.SimpleNamespace(ga_results={"m": 0.6}, predictions=[])

    class Predictor:
        def predict(self, action, metric):
            return 1.0

    class EnsPred:
        def __init__(self) -> None:
            self.trained = 0

        def predict(self, action, metric):
            if action == "self_improvement":
                return 0.8, 0.05
            return 0.7, 0.1

        def train(self):
            self.trained += 1

    class Optimizer:
        def __init__(self) -> None:
            self.updated = 0

        def available_workflows(self):
            return ["wf1"]

        def update_priorities(self, bots, workflows=None, metrics_db=None, prune_threshold: float = 0.0):
            self.updated += 1
            return {b: 1.0 for b in bots}

    opt = Optimizer()

    ens = EnsPred()
    orch = EvolutionOrchestrator(
        data_bot,
        cap_bot,
        Improver(),
        Evolver(),
        history_db=hist,
        resource_optimizer=opt,
        analysis_bot=Predictor(),
        multi_predictor=ens,
    )
    orch.triggers = EvolutionTrigger(error_rate=0.1, roi_drop=-0.1, energy_threshold=0.1)
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    orch.prev_roi = 1.0
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)

    orch.run_cycle()

    rows = hist.fetch()
    assert rows and rows[0][0] == "self_improvement"
    assert opt.updated == 1
    assert ens.trained == 1
    assert any(action == "self_improvement" and kw.get("patch_id") == 42 for action, kw in data_bot.logged)
