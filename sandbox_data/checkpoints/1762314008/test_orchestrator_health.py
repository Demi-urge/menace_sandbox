# flake8: noqa
import sys
import types
from pathlib import Path

sys.modules.setdefault("psutil", types.ModuleType("psutil"))
git_mod = types.ModuleType("git")
class DummyRepo:
    def __init__(self, *a, **k):
        pass
git_mod.Repo = DummyRepo
sys.modules.setdefault("git", git_mod)
for mod in [
    "marshmallow",
    "matplotlib",
    "matplotlib.pyplot",  # path-ignore
    "requests",
    "pymongo",
    "redis",
    "docker",
    "dotenv",
    "prometheus_client",
    "sklearn",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.cluster",
    "sklearn.linear_model",
    "pandas",
    "numpy",
    "yaml",
    "networkx",
]:
    sys.modules.setdefault(mod, types.ModuleType(mod))
if "sklearn.feature_extraction.text" in sys.modules:
    sys.modules["sklearn.feature_extraction.text"].TfidfVectorizer = object
if "sklearn.cluster" in sys.modules:
    sys.modules["sklearn.cluster"].KMeans = object
if "sklearn.linear_model" in sys.modules:
    sys.modules["sklearn.linear_model"].LinearRegression = object
if "dotenv" in sys.modules:
    sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
if "prometheus_client" in sys.modules:
    prom = sys.modules["prometheus_client"]
    prom.CollectorRegistry = object
    prom.Counter = lambda *a, **k: None
    prom.Gauge = lambda *a, **k: None
if "networkx" in sys.modules:
    sys.modules["networkx"].DiGraph = object
if "pandas" in sys.modules:
    pd_mod = sys.modules["pandas"]
    class DummyDF:
        def __init__(self, *a, **k):
            pass
    pd_mod.DataFrame = DummyDF
    pd_mod.read_csv = lambda *a, **k: DummyDF()
if "requests" in sys.modules:
    sys.modules["requests"].Session = lambda *a, **k: None

dyn = types.ModuleType("dynamic_path_router")
dyn.resolve_path = lambda p, **k: Path(p)
dyn.get_project_root = lambda: Path(".")
sys.modules.setdefault("dynamic_path_router", dyn)

map_mod = types.ModuleType("menace.model_automation_pipeline")
class DummyPipeline:
    def __init__(self, *a, **k):
        pass
    def run(self, *a, **k):
        return types.SimpleNamespace(package=None, roi=None)
map_mod.ModelAutomationPipeline = DummyPipeline
map_mod.AutomationResult = types.SimpleNamespace
sys.modules.setdefault("menace.model_automation_pipeline", map_mod)

sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass
engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sqlalchemy_mod.create_engine = lambda *a, **k: object
class DummyMeta:
    def reflect(self, *a, **k):
        pass
sqlalchemy_mod.MetaData = lambda *a, **k: DummyMeta()
sqlalchemy_mod.Table = lambda *a, **k: object
sqlalchemy_mod.select = lambda *a, **k: object
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

import menace.menace_orchestrator as mo


def test_reroute_on_failure():
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    orch.create_oversight("A", "L1")
    orch.create_oversight("B", "L1")
    orch.create_oversight("C", "L1")

    class DummyKG:
        def root_causes(self, bot: str):
            return [f"error:{bot}"]

    orch.knowledge_graph = DummyKG()
    causes = orch.record_failure("B")
    assert causes == ["error:B"]

    assigned = orch.reassign_task("A", ["B"], ["C"])
    assert assigned == "C"
