import types
import sys

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
from tests.test_self_improvement_logging import _load_engine


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


context_builder_util = types.ModuleType("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda builder: None
sys.modules.setdefault("context_builder_util", context_builder_util)


def test_proactive_auto_patch(monkeypatch, tmp_path, caplog):
    sie = _load_engine()
    sys.modules["menace"].RAISE_ERRORS = False
    sie.ResearchAggregatorBot = lambda *a, **k: object()
    sie.ErrorBot = lambda *a, **k: types.SimpleNamespace(
        predict_errors=lambda: [],
        get_error_clusters=lambda: {},
        summarize_telemetry=lambda limit=10: [],
        auto_patch_recurrent_errors=lambda: None,
    )
    sie.ErrorDB = lambda *a, **k: object()
    sie.MetricsDB = lambda *a, **k: object()
    sie.SelfImprovementPolicy = lambda *a, **k: object()
    sie.ConfigurableSelfImprovementPolicy = lambda *a, **k: object()
    sie.SandboxSettings = lambda: types.SimpleNamespace(
        sandbox_data_dir=str(tmp_path),
        sandbox_score_db="db",
        synergy_weight_roi=1.0,
        synergy_weight_efficiency=1.0,
        synergy_weight_resilience=1.0,
        synergy_weight_antifragility=1.0,
        roi_ema_alpha=0.1,
        synergy_weights_lr=0.1,
    )

    class DummyPipe:
        def run(self, model: str, energy: int = 1):
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
        def energy_score(self, **k) -> float:
            return 1.0

        def profit(self) -> float:
            return 0.0

        def log_evolution_event(self, *a, **k):
            pass

    class DummyErrorBot:
        def __init__(self):
            self.auto_patched = False
            self.summary = [
                {"error_type": "foo_error", "count": 5, "success_rate": 0.2}
            ]

        def predict_errors(self):
            return ["foo_error"]

        def get_error_clusters(self):
            return {"foo_error": 1}

        def summarize_telemetry(self, limit: int = 10):
            return self.summary

        def auto_patch_recurrent_errors(self):
            self.auto_patched = True

    sie.AutomationResult = lambda package=None, roi=None: types.SimpleNamespace(
        package=package, roi=roi
    )
    sie.ROIResult = lambda val=0.0: types.SimpleNamespace(roi=val)
    monkeypatch.setattr(sie.SelfImprovementEngine, "_record_state", lambda self: None)

    eng = sie.SelfImprovementEngine(
        context_builder=DummyContextBuilder(),
        interval=0,
        pipeline=DummyPipe(),
        diagnostics=DummyDiag(),
        info_db=DummyInfo(),
        capital_bot=DummyCapital(),
        synergy_weights_path=tmp_path / "weights.json",
        synergy_weights_lr=1.0,
    )
    sie.bootstrap = lambda: 0
    eng.module_clusters = {"mod1": 1}
    err = DummyErrorBot()
    eng.error_bot = err
    caplog.set_level("INFO")

    eng.run_cycle()
    assert err.auto_patched
    assert "proactive patch prevented faults" in caplog.text
