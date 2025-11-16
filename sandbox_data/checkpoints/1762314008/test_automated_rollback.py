# flake8: noqa
import sys
import types
from pathlib import Path

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
sys.modules.setdefault("yaml", yaml_mod)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

sys.modules.setdefault("psutil", types.ModuleType("psutil"))
git_mod = types.ModuleType("git")
git_mod.Repo = object
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
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

import menace.self_coding_engine as sce
import menace.code_database as cd
import menace.menace_memory_manager as mm
from menace.menace_orchestrator import MenaceOrchestrator
from menace.advanced_error_management import AutomatedRollbackManager


def test_multi_node_auto_rollback(tmp_path, monkeypatch):
    rb = AutomatedRollbackManager(str(tmp_path / "rb.db"))
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    tracker_a = sce.BaselineTracker()
    tracker_b = sce.BaselineTracker()
    eng_a = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mm.MenaceMemoryManager(tmp_path / "m1.db"),
        patch_db=patch_db,
        bot_name="A",
        rollback_mgr=rb,
        delta_tracker=tracker_a,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    eng_b = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mm.MenaceMemoryManager(tmp_path / "m2.db"),
        patch_db=patch_db,
        bot_name="B",
        rollback_mgr=rb,
        delta_tracker=tracker_b,
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )

    for eng in (eng_a, eng_b):
        monkeypatch.setattr(eng, "_run_ci", lambda: True)
        monkeypatch.setattr(eng, "generate_helper", lambda d: "#patch\n")

    monkeypatch.setattr(eng_a, "_current_errors", lambda: 0)
    call = {"n": 0}

    def err_b():
        call["n"] += 1
        return 0 if call["n"] == 1 else 2

    monkeypatch.setattr(eng_b, "_current_errors", err_b)

    auto_calls = []

    def fake_auto(pid: str, nodes):
        auto_calls.append((pid, tuple(nodes)))
        rb.rollback(pid)

    monkeypatch.setattr(rb, "auto_rollback", fake_auto)

    orch = MenaceOrchestrator(
        rollback_mgr=rb,
        context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None),
    )
    orch.register_engine("A", eng_a)
    orch.register_engine("B", eng_b)

    pa = tmp_path / "a.py"  # path-ignore
    pb = tmp_path / "b.py"  # path-ignore
    pa.write_text("def a():\n    pass\n")
    pb.write_text("def b():\n    pass\n")

    results = orch.apply_patch_all({"A": pa, "B": pb}, "helper")

    assert results["B"][1]
    assert "#patch" not in pa.read_text()
    assert "#patch" not in pb.read_text()
    assert auto_calls
    assert auto_calls[0][1] == ("A", "B")
    assert not rb.applied_patches()
