import types
import sys

stub = types.ModuleType("stub")
for mod in ("psutil", "networkx", "pandas", "pulp", "scipy"):
    if mod not in sys.modules:
        sys.modules[mod] = stub
stub.stats = object
stub.read_sql = lambda *a, **k: []
stub.DiGraph = object
try:  # optional dependency
    import jinja2 as _j2  # noqa: F401
except Exception:  # pragma: no cover - stub
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
stub.Gauge = None
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

import menace.evolution_orchestrator as eo
import menace.self_improvement as sie
import menace.self_coding_engine as sce
import menace.model_automation_pipeline as mapl
import menace.pre_execution_roi_bot as prb
import menace.code_database as cd
import menace.menace_memory_manager as mm
import menace.data_bot as db
import menace.research_aggregator_bot as rab
from menace.evolution_history_db import EvolutionHistoryDB


def test_full_self_optimisation(tmp_path, monkeypatch):
    mdb = db.MetricsDB(tmp_path / "m.db")
    patch_db = cd.PatchHistoryDB(tmp_path / "p.db")
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    info_db = rab.InfoDB(tmp_path / "i.db")
    data_bot = db.DataBot(mdb, patch_db=patch_db, evolution_db=hist)

    builder = types.SimpleNamespace(
        build_context=lambda *a, **k: {},
        refresh_db_weights=lambda *a, **k: None,
    )
    engine = sce.SelfCodingEngine(
        cd.CodeDB(tmp_path / "c.db"),
        mm.MenaceMemoryManager(tmp_path / "mem.db"),
        data_bot=data_bot,
        patch_db=patch_db,
        context_builder=builder,
    )
    monkeypatch.setattr(engine, "_run_ci", lambda: True)
    monkeypatch.setattr(engine, "generate_helper", lambda d: "def auto_x():\n    pass\n")

    class StubPipeline:
        def run(self, model: str, energy: int = 1):
            return mapl.AutomationResult(
                package=object(),
                roi=prb.ROIResult(1.0, 0.5, 1.0, 0.5, 0.1),
            )

    diag = types.SimpleNamespace(
        metrics=mdb,
        error_bot=types.SimpleNamespace(db=types.SimpleNamespace(discrepancies=lambda: [])),
        diagnose=lambda: [],
    )

    improver = sie.SelfImprovementEngine(
        context_builder=builder,
        interval=0,
        pipeline=StubPipeline(),
        diagnostics=diag,
        info_db=info_db,
        self_coding_engine=engine,
        data_bot=data_bot,
        optimize_self=True,
    )
    monkeypatch.setattr(improver, "_record_state", lambda: None)
    monkeypatch.setattr(improver, "_evaluate_learning", lambda: None)

    cap_bot = types.SimpleNamespace(
        energy_score=lambda **k: 1.0,
        profit=lambda: 0.0,
        log_evolution_event=lambda *a, **k: None,
    )

    class DummyEvolver:
        def __init__(self) -> None:
            self.bots = ["m"]

        def run_cycle(self):
            return types.SimpleNamespace(ga_results={}, predictions=[])

    orch = eo.EvolutionOrchestrator(
        data_bot,
        cap_bot,
        improver,
        DummyEvolver(),
        history_db=hist,
    )

    orch.triggers = eo.EvolutionTrigger(error_rate=0.1, roi_drop=-0.1, energy_threshold=0.1)
    monkeypatch.setattr(orch, "_latest_roi", lambda: 0.0)
    monkeypatch.setattr(orch, "_error_rate", lambda: 0.2)

    patch_file = tmp_path / "auto_helpers.py"  # path-ignore
    patch_file.write_text("def x():\n    pass\n")
    monkeypatch.chdir(tmp_path)

    orch.run_cycle()

    patches = patch_db.top_patches(limit=10)
    assert patches and any(p.filename.endswith("auto_helpers.py") for p in patches)  # path-ignore

    self_patch = [p for p in patches if p.filename.endswith("self_improvement.py")]  # path-ignore
    assert self_patch

    events = hist.fetch()
    assert len(events) >= 2
    assert any(ev[7] == 1 for ev in events)
