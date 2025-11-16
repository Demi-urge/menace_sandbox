import types
import sys

# Stub optional dependencies before importing menace package
stub = types.ModuleType("stub")
for mod in (
    "psutil",
    "networkx",
    "pulp",
    "sqlalchemy",
    "sqlalchemy.engine",
    "httpx",
    "fastapi",
    "prometheus_client",
    "jinja2",
    "pandas",
    "yaml",
    "numpy",
    "git",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.cluster",
    "sklearn.linear_model",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.ensemble",
    "dotenv",
):
    sys.modules.setdefault(mod, stub)

matplotlib_stub = types.ModuleType("matplotlib")
matplotlib_stub.__path__ = []
sys.modules.setdefault("matplotlib", matplotlib_stub)
plt_stub = types.ModuleType("pyplot")
plt_stub.plot = lambda *a, **k: None
sys.modules["matplotlib.pyplot"] = plt_stub  # path-ignore

stub.TfidfVectorizer = object
stub.KMeans = object
stub.LinearRegression = object
stub.train_test_split = lambda *a, **k: ([], [])
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = object
stub.RandomForestClassifier = object

stub.Engine = object
stub.CollectorRegistry = object
stub.Gauge = None
stub.Counter = object
stub.Repo = object
if "dotenv" in sys.modules:
    sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
if "jinja2" in sys.modules:
    sys.modules["jinja2"].Template = lambda *a, **k: None

import menace.dynamic_resource_allocator_bot as drab
import menace.advanced_error_management as aem
import menace.resource_allocation_optimizer as rao


def test_scaling_hint_logged(monkeypatch, tmp_path, caplog):
    class Col(list):
        def __eq__(self, other):
            return [v == other for v in self]

        def mean(self):
            return sum(self) / len(self) if self else 0.0

    class DummyDF:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            if isinstance(key, list):
                return DummyDF([r for r, k in zip(self.rows, key) if k])
            return Col([r[key] for r in self.rows])

        @property
        def empty(self):
            return not self.rows

        @property
        def iloc(self):
            class ILoc:
                def __init__(self, rows):
                    self.rows = rows

                def __getitem__(self, idx):
                    return self.rows[idx]

            return ILoc(self.rows)

    class DummyDB(drab.MetricsDB):
        def __init__(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

        def fetch(self, limit=100, *, start=None, end=None):
            return DummyDF([
                {
                    "bot": "bot",
                    "cpu": 90.0,
                    "memory": 50.0,
                    "response_time": 0.1,
                    "disk_io": 1.0,
                }
            ])

    mdb = DummyDB()
    caplog.set_level("INFO")

    dummy_pred = types.SimpleNamespace(predict=lambda b: drab.ResourceMetrics(1, 1, 1, 1))
    alloc_bot = types.SimpleNamespace(allocate=lambda metrics_map, weights=None: [(b, True) for b in metrics_map])
    orch = types.SimpleNamespace(hints=[])

    def recv(h):
        orch.hints.append(h)

    orch.receive_scaling_hint = recv
    allocator = drab.DynamicResourceAllocator(
        metrics_db=mdb,
        prediction_bot=dummy_pred,
        ledger=drab.DecisionLedger(tmp_path / "d.db"),
        alloc_bot=alloc_bot,
        pathway_db=None,
        predictive_allocator=aem.PredictiveResourceAllocator(mdb),
        orchestrator=orch,
        context_builder=types.SimpleNamespace(),
    )
    allocator.allocate(["bot"])

    assert any("scale up resources" in r.message for r in caplog.records)
    assert orch.hints and orch.hints[0] == "scale_up"


def test_scaling_respects_optimizer(monkeypatch, tmp_path):
    class Col(list):
        def mean(self):
            return sum(self) / len(self) if self else 0.0

    class DummyDF:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            return Col([r[key] for r in self.rows])

        @property
        def empty(self):
            return not self.rows

    class DummyDB(drab.MetricsDB):
        def __init__(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

        def fetch(self, limit=100, *, start=None, end=None):
            return DummyDF([
                {
                    "bot": "bot",
                    "cpu": 90.0,
                    "memory": 50.0,
                    "response_time": 0.1,
                    "disk_io": 1.0,
                }
            ])

    mdb = DummyDB()
    dummy_pred = types.SimpleNamespace(predict=lambda b: drab.ResourceMetrics(1, 1, 1, 1))
    alloc_bot = types.SimpleNamespace(allocate=lambda metrics_map, weights=None: [(b, True) for b in metrics_map])
    orch = types.SimpleNamespace(hints=[])
    orch.receive_scaling_hint = lambda h: orch.hints.append(h)
    opt = rao.ResourceAllocationOptimizer(rao.ROIDB(tmp_path / "r.db"))
    opt.bandit.weights = {"bot": 0.1}
    allocator = drab.DynamicResourceAllocator(
        metrics_db=mdb,
        prediction_bot=dummy_pred,
        ledger=drab.DecisionLedger(tmp_path / "d2.db"),
        alloc_bot=alloc_bot,
        pathway_db=None,
        predictive_allocator=aem.PredictiveResourceAllocator(mdb),
        orchestrator=orch,
        optimizer=opt,
        context_builder=types.SimpleNamespace(),
    )
    allocator.allocate(["bot"])

    assert not orch.hints
