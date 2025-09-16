import types
import sys

# stubs for modules referenced during engine import
auto_env = types.ModuleType("auto_env_setup")
auto_env.get_recursive_isolated = lambda *a, **k: None
auto_env.set_recursive_isolated = lambda *a, **k: None
sys.modules.setdefault("auto_env_setup", auto_env)
sys.modules.setdefault("menace.auto_env_setup", auto_env)
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}))
sts_mod = types.ModuleType("self_test_service")
sts_mod.SelfTestService = type("SelfTestService", (), {})
sys.modules.setdefault("self_test_service", sts_mod)
sys.modules.setdefault("menace.self_test_service", sts_mod)
shd_mod = types.ModuleType("synergy_history_db")
shd_mod.fetch_all = lambda conn: []
shd_mod.HistoryParseError = Exception
sys.modules.setdefault("synergy_history_db", shd_mod)
sys.modules.setdefault("menace.synergy_history_db", shd_mod)
mid_mod = types.ModuleType("module_index_db")
mid_mod.ModuleIndexDB = type(
    "ModuleIndexDB",
    (),
    {"__init__": lambda self, *a, **k: None, "_norm": lambda self, m: m},
)
sys.modules.setdefault("module_index_db", mid_mod)
sys.modules.setdefault("menace.module_index_db", mid_mod)


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)

log_stub = types.ModuleType("menace_sandbox.logging_utils")
log_stub.get_logger = lambda *a, **k: types.SimpleNamespace(
    info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
)
log_stub.setup_logging = lambda *a, **k: None
log_stub.log_record = lambda **kw: kw
sys.modules.setdefault("menace_sandbox.logging_utils", log_stub)
log_mod = sys.modules.setdefault("menace.logging_utils", types.ModuleType("menace.logging_utils"))
log_mod.get_logger = lambda *a, **k: types.SimpleNamespace(
    info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None
)
log_mod.setup_logging = lambda *a, **k: None
log_mod.log_record = lambda **kw: kw

from tests.test_self_improvement_logging import _load_engine  # noqa: E402


def test_high_risk_prediction_triggers_patch(monkeypatch, tmp_path):
    # stub sandbox settings to simulate autonomous mode with auto_patch disabled
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_score_db = "db"
            self.synergy_weight_roi = 1.0
            self.synergy_weight_efficiency = 1.0
            self.synergy_weight_resilience = 1.0
            self.synergy_weight_antifragility = 1.0
            self.roi_ema_alpha = 0.1
            self.synergy_weights_lr = 0.1
            self.auto_patch_high_risk = False
            self.menace_mode = "autonomous"

    ss_mod = sys.modules["sandbox_settings"]
    ss_mod.SandboxSettings = DummySettings
    ss_mod.DEFAULT_SEVERITY_SCORE_MAP = {}
    sys.modules["menace.sandbox_settings"] = ss_mod

    sie = _load_engine()

    sie.ResearchAggregatorBot = lambda *a, **k: object()
    sie.ErrorDB = lambda *a, **k: object()
    sie.MetricsDB = lambda *a, **k: object()
    sie.SelfImprovementPolicy = lambda *a, **k: object()
    sie.ConfigurableSelfImprovementPolicy = lambda *a, **k: object()
    sie.AutomationResult = (
        lambda package=None, roi=None: types.SimpleNamespace(package=package, roi=roi)
    )
    sie.ROIResult = lambda val=0.0: types.SimpleNamespace(roi=val)

    class DummyPipe:
        def run(self, model, energy=1):
            return sie.AutomationResult(package=None, roi=sie.ROIResult(0.0))

    class DummyDiag:
        def __init__(self):
            self.metrics = types.SimpleNamespace(fetch=lambda *a, **k: [])
            self.error_bot = types.SimpleNamespace(
                db=types.SimpleNamespace(discrepancies=lambda: [])
            )

        def diagnose(self):
            return []

    class DummyInfo:
        def set_current_model(self, *a, **k):
            pass

    class DummyCapital:
        def energy_score(self, **k):
            return 1.0

        def profit(self):
            return 0.0

        def log_evolution_event(self, *a, **k):
            pass

    class DummyErrorDB:
        def __init__(self):
            self.telemetry = []

        def add_telemetry(self, event):
            self.telemetry.append(event)

    class DummyErrorBot:
        def __init__(self):
            self.db = DummyErrorDB()

        def predict_errors(self):
            return []

        def get_error_clusters(self):
            return {}

        def summarize_telemetry(self, limit: int = 10):
            return []

        def auto_patch_recurrent_errors(self):
            pass

    class DummyGraph:
        def __init__(self):
            self.events = []
            self.updated = False

        def add_telemetry_event(self, bot, event_type, module, patch_id=None):
            self.events.append((bot, event_type, module, patch_id))

        def update_error_stats(self, db):
            self.updated = True

    class DummyPredictor:
        def __init__(self):
            self.graph = DummyGraph()

        def predict_high_risk_modules(self):
            return ["mod1.py"]  # path-ignore

    sie.ErrorBot = lambda *a, **k: DummyErrorBot()

    eng = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        pipeline=DummyPipe(),
        diagnostics=DummyDiag(),
        info_db=DummyInfo(),
        capital_bot=DummyCapital(),
        synergy_weights_path=tmp_path / "weights.json",
        synergy_weights_lr=1.0,
        error_predictor=DummyPredictor(),
    )

    eng.self_coding_engine = object()

    calls = []
    monkeypatch.setattr(
        sie, "generate_patch", lambda mod, engine=None, **kw: calls.append(mod) or 7
    )

    eng._apply_high_risk_patches()

    assert eng.auto_patch_high_risk is True
    assert calls == ["mod1.py"]  # path-ignore
    assert (
        eng.error_bot.db.telemetry
        and eng.error_bot.db.telemetry[0].module == "mod1.py"
    )  # path-ignore
    assert eng.error_predictor.graph.updated
    assert any(
        evt[1] == "preemptive_patch" and evt[2] == "mod1.py"
        for evt in eng.error_predictor.graph.events
    )  # path-ignore
