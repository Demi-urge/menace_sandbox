# flake8: noqa
import sys
import types
import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)

stub = types.ModuleType("stub")
stub.Repo = object
stub.Engine = object
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
    "cryptography",
    "cryptography.hazmat",
    "cryptography.hazmat.primitives",
    "cryptography.hazmat.primitives.asymmetric",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "dotenv",
):
    sys.modules.setdefault(mod, stub)
matplotlib_stub = types.ModuleType("matplotlib")
matplotlib_stub.__path__ = []
sys.modules.setdefault("matplotlib", matplotlib_stub)
plt_stub = types.ModuleType("pyplot")
plt_stub.plot = lambda *a, **k: None
sys.modules["matplotlib.pyplot"] = plt_stub  # path-ignore
if "git" in sys.modules:
    sys.modules["git"].Repo = object
if "dotenv" in sys.modules:
    sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
if "jinja2" in sys.modules:
    sys.modules["jinja2"].Template = lambda *a, **k: None
stub.TfidfVectorizer = object
stub.KMeans = object
stub.LinearRegression = object
stub.train_test_split = lambda *a, **k: ([], [])
stub.accuracy_score = lambda *a, **k: 0.0
stub.LogisticRegression = object
stub.RandomForestClassifier = object
stub.CollectorRegistry = object
stub.Gauge = None
stub.Counter = object

ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
if "jinja2" in sys.modules:
    sys.modules["jinja2"].Template = lambda *a, **k: None

import menace.menace_orchestrator as mo

class DummyAuto:
    def __init__(self):
        self.calls = []
    def scale(self, metrics):
        self.calls.append(metrics)

class DummyAlloc:
    def __init__(self):
        self.called = False
    def allocate(self, bots, weight=1.0):
        self.called = True


def test_hint_triggers_autoscaler():
    orch = mo.MenaceOrchestrator(context_builder=mo.ContextBuilder())
    orch.planner.autoscaler = DummyAuto()
    orch.receive_scaling_hint("scale_up")
    assert orch.planner.autoscaler.calls


def test_hint_triggers_allocator():
    orch = mo.MenaceOrchestrator(context_builder=mo.ContextBuilder())
    orch.pipeline.allocator = DummyAlloc()
    orch.engines = {"a": None}
    orch.receive_scaling_hint("rebalance")
    assert orch.pipeline.allocator.called
