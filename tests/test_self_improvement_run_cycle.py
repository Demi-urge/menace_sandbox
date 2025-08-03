import types
import importlib.util
import importlib.machinery
import sys
import json
import os
from module_index_db import ModuleIndexDB
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_modules():
    if "menace" not in sys.modules:
        pkg = types.ModuleType("menace")
        pkg.__path__ = [str(ROOT)]
        pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
        pkg.RAISE_ERRORS = False
        sys.modules["menace"] = pkg
    if "menace.self_model_bootstrap" not in sys.modules:
        stub = types.ModuleType("menace.self_model_bootstrap")
        stub.bootstrap = lambda: 0
        sys.modules["menace.self_model_bootstrap"] = stub
    sys.modules.setdefault("jinja2", types.ModuleType("jinja2"))
    sys.modules["jinja2"].Template = lambda *a, **k: None
    # stub cryptography for AuditTrail dependency
    sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
    sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
    sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
    sys.modules.setdefault(
        "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
    )
    sys.modules.setdefault(
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        types.ModuleType("ed25519"),
    )
    ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
    ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
    ed.Ed25519PublicKey = object
    serialization = types.ModuleType("serialization")
    primitives = sys.modules["cryptography.hazmat.primitives"]
    primitives.serialization = serialization
    sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

    # stub sandbox_runner.environment to avoid heavy side effects
    pkg = sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
    pkg.__path__ = []
    env = types.ModuleType("sandbox_runner.environment")
    env.generate_workflows_for_modules = lambda *a, **k: None
    env.run_workflow_simulations = lambda *a, **k: None
    env.try_integrate_into_workflows = lambda *a, **k: None
    env.auto_include_modules = lambda *a, **k: None
    pkg.environment = env
    sys.modules["sandbox_runner.environment"] = env

    mods = {}
    for name in [
        "logging_utils",
        "diagnostic_manager",
        "error_bot",
        "data_bot",
        "research_aggregator_bot",
        "pre_execution_roi_bot",
        "model_automation_pipeline",
        "roi_tracker",
        "self_improvement_engine",
        "code_database",
    ]:
        if f"menace.{name}" in sys.modules:
            mods[name] = sys.modules[f"menace.{name}"]
            continue
        spec = importlib.util.spec_from_file_location(
            f"menace.{name}", ROOT / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"menace.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


mods = _load_modules()

sie = mods["self_improvement_engine"]
dm = mods["diagnostic_manager"]
eb = mods["error_bot"]
db = mods["data_bot"]
rab = mods["research_aggregator_bot"]
prb = mods["pre_execution_roi_bot"]
mp = mods["model_automation_pipeline"]
rt = mods["roi_tracker"]
cd = mods["code_database"]


class StubPipeline:
    def run(self, model: str, energy: int = 1):
        return mp.AutomationResult(package=None, roi=prb.ROIResult(0.0, 0.0, 0.0, 0.0, 0.0))


class CapBot:
    def __init__(self, vals):
        self.vals = list(vals)
        self.idx = 0

    def profit(self):
        val = self.vals[self.idx]
        if self.idx < len(self.vals) - 1:
            self.idx += 1
        return val

    def energy_score(self, **_: object) -> float:
        return 1.0

    def log_evolution_event(self, *a, **k):
        pass


class PatchDB:
    def __init__(self, path=None):
        self.path = path
        self.records: list[tuple[str]] = []

    def filter(self, *a, **k):
        return []

    def success_rate(self, limit: int = 50) -> float:
        return 0.0

    def keyword_features(self):
        return 0, 0

    def add(self, rec):
        self.records.append((rec.filename,))

    def _connect(self):
        class Conn:
            def __init__(self, rows):
                self.rows = rows

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

            def execute(self, *a, **k):
                return type("R", (), {"fetchall": lambda self: self.rows})(self.rows)

        return Conn(self.records)


class CapturePolicy:
    def __init__(self):
        self.records = []

    def score(self, state):
        return 0.0

    def update(self, state, reward, next_state=None, **kw):
        self.records.append((tuple(state), reward, tuple(next_state) if next_state else None))
        return 0.0


def _make_engine(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))
    pipe = StubPipeline()
    capital = CapBot([0.0, 1.0, 1.5, 2.0, 2.0, 4.0])
    policy = CapturePolicy()
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=capital,
        patch_db=PatchDB(),
        policy=policy,
        roi_ema_alpha=0.5,
        bot_name="testbot",
    )
    engine.tracker = rt.ROITracker()
    sie.bootstrap = lambda: 0
    engine._test_orphan_modules = lambda paths: set(paths)
    engine._update_orphan_modules = lambda modules=None: None
    return engine, policy


def test_run_cycle_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    engine, policy = _make_engine(tmp_path)
    for _ in range(3):
        res = engine.run_cycle()
        assert isinstance(res, mp.AutomationResult)
    assert len(engine.roi_history) == 3
    assert engine.roi_delta_ema is not None
    assert len(policy.records) == 3


class DummyMetaLogger:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.audit = types.SimpleNamespace(record=lambda obj: self.events.append(obj))


def _make_refresh_engine(tmp_path, patch_db, module_index, meta_log):
    mdb = db.MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db")
    info = rab.InfoDB(tmp_path / "i.db")
    diag = dm.DiagnosticManager(mdb, eb.ErrorBot(edb, mdb))
    pipe = StubPipeline()
    capital = CapBot([0.0])
    policy = CapturePolicy()
    engine = sie.SelfImprovementEngine(
        interval=0,
        pipeline=pipe,
        diagnostics=diag,
        info_db=info,
        capital_bot=capital,
        patch_db=patch_db,
        policy=policy,
        roi_ema_alpha=0.5,
        bot_name="testbot",
        module_index=module_index,
        meta_logger=meta_log,
        auto_refresh_map=True,
    )
    engine.tracker = rt.ROITracker()
    sie.bootstrap = lambda: 0
    engine._update_orphan_modules = lambda modules=None: None
    return engine, policy


def test_auto_refresh_module_map(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    map_path = tmp_path / "module_map.json"
    map_path.write_text(json.dumps({"modules": {"old.py": 0}, "groups": {"0": 0}}))
    module_index = ModuleIndexDB(map_path)
    patch_db = PatchDB(tmp_path / "p.db")
    patch_db.add(types.SimpleNamespace(filename="new.py"))
    meta = DummyMetaLogger()

    def fake_build(repo_path, **kw):
        return {"new.py": 1, "old.py": 0}

    monkeypatch.setattr(sie, "build_module_map", fake_build)
    engine, _ = _make_refresh_engine(tmp_path, patch_db, module_index, meta)
    res = engine.run_cycle()
    assert isinstance(res, mp.AutomationResult)
    assert module_index.get("new.py") is not None
    data = json.loads(map_path.read_text())
    assert "new.py" in data.get("modules", {})


def test_run_cycle_processes_orphans(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    orphan_path = tmp_path / "orphan_modules.json"
    orphan_path.write_text(json.dumps(["foo.py"]))
    (tmp_path / "foo.py").write_text("def foo():\n    return 1\n")

    engine, _ = _make_engine(tmp_path)

    called: dict[str, list[str]] = {}

    def fake_refresh(self, modules=None):
        called["mods"] = list(modules or [])
        orphan_path.write_text("[]")

    engine._refresh_module_map = types.MethodType(fake_refresh, engine)
    engine._update_orphan_modules = lambda modules=None: None

    engine.run_cycle()

    assert called["mods"] == ["foo.py"]
    assert json.loads(orphan_path.read_text()) == []
