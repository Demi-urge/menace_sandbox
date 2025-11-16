import os
import importlib.util
import sys
import types
import json
import sqlite3
from pathlib import Path

import sandbox_runner.environment as env


def _load_engine():
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    spec = importlib.util.spec_from_file_location(
        "menace", Path(__file__).resolve().parents[1] / "__init__.py"  # path-ignore
    )
    menace = importlib.util.module_from_spec(spec)
    sys.modules["menace"] = menace
    spec.loader.exec_module(menace)

    modules = [
        "menace.self_model_bootstrap",
        "menace.research_aggregator_bot",
        "menace.model_automation_pipeline",
        "menace.diagnostic_manager",
        "menace.error_bot",
        "menace.data_bot",
        "menace.code_database",
        "menace.capital_management_bot",
        "menace.learning_engine",
        "menace.unified_event_bus",
        "menace.neuroplasticity",
        "menace.self_coding_engine",
        "menace.action_planner",
        "menace.evolution_history_db",
        "menace.self_improvement_policy",
        "menace.pre_execution_roi_bot",
        "menace.env_config",
    ]
    for name in modules:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["menace.self_model_bootstrap"].bootstrap = lambda *a, **k: 0
    ra = sys.modules["menace.research_aggregator_bot"]
    class DummyAgg:
        def __init__(self, *a, **k):
            pass
    ra.ResearchAggregatorBot = DummyAgg
    ra.ResearchItem = object
    ra.InfoDB = object
    map_mod = sys.modules["menace.model_automation_pipeline"]
    map_mod.ModelAutomationPipeline = lambda *a, **k: object()
    map_mod.AutomationResult = object
    sys.modules["menace.diagnostic_manager"].DiagnosticManager = lambda *a, **k: object()
    err_mod = sys.modules["menace.error_bot"]
    err_mod.ErrorBot = lambda *a, **k: object()
    err_mod.ErrorDB = lambda *a, **k: object()
    sys.modules["menace.data_bot"].MetricsDB = object
    sys.modules["menace.data_bot"].DataBot = object
    sys.modules["menace.code_database"].PatchHistoryDB = object
    sys.modules["menace.capital_management_bot"].CapitalManagementBot = object
    sys.modules["menace.learning_engine"].LearningEngine = object
    sys.modules["menace.unified_event_bus"].UnifiedEventBus = object
    sys.modules["menace.neuroplasticity"].PathwayRecord = object
    sys.modules["menace.neuroplasticity"].Outcome = object
    sys.modules["menace.self_coding_engine"].SelfCodingEngine = object
    sys.modules["menace.action_planner"].ActionPlanner = object
    sys.modules["menace.evolution_history_db"].EvolutionHistoryDB = object
    policy_mod = sys.modules["menace.self_improvement_policy"]
    policy_mod.SelfImprovementPolicy = object
    policy_mod.ConfigurableSelfImprovementPolicy = lambda *a, **k: object()
    class DummyStrategy:
        def update(self, *a, **k):
            return 0.0
        def predict(self, *_):
            return [0.0] * 7
    policy_mod.DQNStrategy = lambda *a, **k: DummyStrategy()
    policy_mod.DoubleDQNStrategy = lambda *a, **k: DummyStrategy()
    policy_mod.ActorCriticStrategy = lambda *a, **k: DummyStrategy()
    policy_mod.torch = None
    pre_mod = sys.modules["menace.pre_execution_roi_bot"]
    pre_mod.PreExecutionROIBot = object
    pre_mod.BuildTask = object
    pre_mod.ROIResult = object
    env_mod = sys.modules["menace.env_config"]
    env_mod.PRE_ROI_SCALE = 1.0
    env_mod.PRE_ROI_BIAS = 0.0
    env_mod.PRE_ROI_CAP = 1.0


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)

    jinja_mod = types.ModuleType("jinja2")
    jinja_mod.Template = lambda *a, **k: None
    sys.modules.setdefault("jinja2", jinja_mod)

    for name in [
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        "cryptography.hazmat.primitives.serialization",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    pyd_mod = types.ModuleType("pydantic")
    pyd_dc = types.ModuleType("dataclasses")
    pyd_dc.dataclass = lambda *a, **k: (lambda f: f)
    pyd_mod.dataclasses = pyd_dc
    pyd_mod.Field = lambda default=None, **k: default
    pyd_mod.BaseModel = object
    sys.modules.setdefault("pydantic", pyd_mod)
    pyd_settings_mod = types.ModuleType("pydantic_settings")
    pyd_settings_mod.BaseSettings = object
    pyd_settings_mod.SettingsConfigDict = dict
    sys.modules.setdefault("pydantic_settings", pyd_settings_mod)

    import menace.self_improvement as sie
    return sie


def test_synergy_integration(monkeypatch, tmp_path):
    sie = _load_engine()

    clone = tmp_path / "clone"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sandbox_runner.py").write_text("print('ok')")  # path-ignore

    monkeypatch.setattr(env, "SANDBOX_REPO_PATH", repo)
    import sandbox_runner.config as conf
    monkeypatch.setattr(conf, "SANDBOX_REPO_PATH", repo)

    monkeypatch.setattr(env.tempfile, "mkdtemp", lambda prefix="": str(clone))

    def fake_copytree(src, dst, dirs_exist_ok=True):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "data").mkdir(exist_ok=True)
        (Path(dst) / "sandbox_runner.py").write_text("print('ok')")  # path-ignore
    monkeypatch.setattr(env.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(env.shutil, "rmtree", lambda path, ignore_errors=True: None)
    monkeypatch.setattr(env, "_docker_available", lambda: False)
    monkeypatch.setattr(env.shutil, "which", lambda n: "/usr/bin/true")

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}
            self.roi_history = []
        def load_history(self, path):
            self.loaded = path
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
                self.roi_history = data.get("roi_history", [])
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    def fake_run(cmd, **kwargs):
        data_dir = clone / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        roi_file.write_text(json.dumps({"roi_history": [0.0], "module_deltas": {}, "metrics_history": {"synergy_roi": [0.0]}}))
        hist_db = data_dir / "synergy_history.db"
        conn = sie.shd.connect(hist_db)
        sie.shd.insert_entry(conn, {"synergy_roi": 0.2})
        conn.close()
        weights_path = data_dir / "synergy_weights.json"
        engine = sie.SelfImprovementEngine(
            context_builder=DummyContextBuilder(),
            interval=0,
            synergy_weights_path=weights_path,
            synergy_weights_lr=1.0,
        )
        engine.tracker = types.SimpleNamespace(metrics_history={"synergy_roi": [0.0]}, roi_history=[0.0])
        monkeypatch.setattr(engine, "_metric_delta", lambda name, window=3: 0.1 if name == "synergy_roi" else 0.0)
        start = engine.synergy_weight_roi
        engine._update_synergy_weights(1.0)
        assert engine.synergy_weight_roi != start
        return types.SimpleNamespace()
    monkeypatch.setattr(env.subprocess, "run", fake_run)

    tracker = env.simulate_full_environment({})

    hist_db = clone / "data" / "synergy_history.db"
    conn = sie.shd.connect(hist_db)
    rows = sie.shd.fetch_all(conn)
    conn.close()
    assert rows and rows[-1]["synergy_roi"] == 0.2

    weights_file = clone / "data" / "synergy_weights.json"
    data = json.loads(weights_file.read_text())
    assert data.get("roi") != 1.0
