# flake8: noqa
import logging
import types
import sys
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing target modules
# ---------------------------------------------------------------------------

# Lightweight vector_service stub for PatchSuggestionDB
vec_mod = types.ModuleType("vector_service")
class _EmbeddableDBMixin:
    def __init__(self, *a, **k):
        pass
    def encode_text(self, *a, **k):
        return [0.0]
    def add_embedding(self, *a, **k):
        pass
    def backfill_embeddings(self, *a, **k):
        pass
vec_mod.EmbeddableDBMixin = _EmbeddableDBMixin
vec_mod.SharedVectorService = object
class PatchLogger:
    def __init__(self, *a, **k):
        pass
vec_mod.PatchLogger = PatchLogger
sys.modules.setdefault("vector_service", vec_mod)

cb_mod = types.ModuleType("vector_service.context_builder")
cb_mod.record_failed_tags = lambda *a, **k: None
cb_mod.load_failed_tags = lambda *a, **k: {}


class ContextBuilder:
    def __init__(self, *a, **k):
        pass


cb_mod.ContextBuilder = ContextBuilder
sys.modules.setdefault("vector_service.context_builder", cb_mod)

tp_mod = types.ModuleType("vector_service.text_preprocessor")
class PreprocessingConfig:
    split_sentences = False
    filter_semantic_risks = False
    chunk_size = None
tp_mod.PreprocessingConfig = PreprocessingConfig
tp_mod.get_config = lambda name: PreprocessingConfig()
tp_mod.generalise = lambda text, *, config=None, db_key=None: text
sys.modules.setdefault("vector_service.text_preprocessor", tp_mod)

emb_mod = types.ModuleType("vector_service.embed_utils")
emb_mod.get_text_embeddings = lambda texts: [[0.0]]
emb_mod.EMBED_DIM = 1
sys.modules.setdefault("vector_service.embed_utils", emb_mod)

# Minimal menace package to avoid executing heavy __init__
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules.setdefault("menace", menace_pkg)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Stubs for modules referenced by SelfCodingManager
sce_mod = types.ModuleType("menace.self_coding_engine")
class SelfCodingEngine:
    def __init__(self):
        builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
sce_mod.SelfCodingEngine = SelfCodingEngine
sys.modules.setdefault("menace.self_coding_engine", sce_mod)

mapl_mod = types.ModuleType("menace.model_automation_pipeline")
class ModelAutomationPipeline:
    def __init__(self, *a, **k):
        pass
mapl_mod.ModelAutomationPipeline = ModelAutomationPipeline
mapl_mod.AutomationResult = object
sys.modules.setdefault("menace.model_automation_pipeline", mapl_mod)

adv_mod = types.ModuleType("menace.advanced_error_management")
class FormalVerifier:
    pass
class AutomatedRollbackManager:
    pass
adv_mod.FormalVerifier = FormalVerifier
adv_mod.AutomatedRollbackManager = AutomatedRollbackManager
sys.modules.setdefault("menace.advanced_error_management", adv_mod)

mlog_mod = types.ModuleType("menace.mutation_logger")
mlog_mod.log_mutation = lambda *a, **k: 0
mlog_mod.record_mutation_outcome = lambda *a, **k: None
sys.modules.setdefault("menace.mutation_logger", mlog_mod)

rb_mod = types.ModuleType("menace.rollback_manager")
class RollbackManager:
    pass
rb_mod.RollbackManager = RollbackManager
sys.modules.setdefault("menace.rollback_manager", rb_mod)

db_mod = types.ModuleType("menace.data_bot")


class DataBot:
    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return 0.0

    def get_thresholds(self, _bot: str):
        from menace.roi_thresholds import ROIThresholds

        return ROIThresholds(roi_drop=-0.5, error_threshold=2.0, test_failure_threshold=2.0)



db_mod.DataBot = DataBot
db_mod.persist_sc_thresholds = lambda *a, **k: None
sys.modules.setdefault("menace.data_bot", db_mod)

err_mod = types.ModuleType("menace.error_bot")
class ErrorDB:
    pass
err_mod.ErrorDB = ErrorDB
sys.modules.setdefault("menace.error_bot", err_mod)

# QuickFixEngine stub used for patch tests
qfe_mod = types.ModuleType("menace.quick_fix_engine")
class QuickFixEngine:
    def __init__(self, *a, **k):
        self.context_builder = None
qfe_mod.QuickFixEngine = QuickFixEngine
qfe_mod.QuickFixEngineError = type("QuickFixEngineError", (Exception,), {})
qfe_mod.generate_patch = lambda *a, **k: None
sys.modules.setdefault("menace.quick_fix_engine", qfe_mod)

psdb_mod = types.ModuleType("menace.patch_suggestion_db")
class PatchSuggestionDB:
    def __init__(self, *a, **k):
        pass


psdb_mod.PatchSuggestionDB = PatchSuggestionDB
sys.modules.setdefault("menace.patch_suggestion_db", psdb_mod)
sys.modules.setdefault("patch_suggestion_db", psdb_mod)

ueb_mod = types.ModuleType("menace.unified_event_bus")
class UnifiedEventBus:
    pass
class EventBus:
    pass
ueb_mod.UnifiedEventBus = UnifiedEventBus
ueb_mod.EventBus = EventBus
sys.modules.setdefault("menace.unified_event_bus", ueb_mod)
sys.modules.setdefault("unified_event_bus", ueb_mod)

br_mod = types.ModuleType("menace.bot_registry")
class BotRegistry:
    def register_bot(self, name):
        pass
br_mod.BotRegistry = BotRegistry
sys.modules.setdefault("menace.bot_registry", br_mod)
sys.modules.setdefault("bot_registry", br_mod)

thr_mod = types.ModuleType("menace.sandbox_runner.test_harness")
class HarnessResult:
    def __init__(self, success=True, failure=None, stdout="", stderr="", duration=0.0):
        self.success = success
        self.failure = failure
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration

def run_tests(repo, path, *, backend="venv"):
    return HarnessResult(True)
thr_mod.TestHarnessResult = HarnessResult
thr_mod.run_tests = run_tests
sys.modules.setdefault("menace.sandbox_runner", types.ModuleType("menace.sandbox_runner"))
sys.modules.setdefault("menace.sandbox_runner.test_harness", thr_mod)
code_db_mod = types.ModuleType("menace.code_database")
class PatchRecord:
    pass
code_db_mod.PatchRecord = PatchRecord
class PatchHistoryDB:
    pass
code_db_mod.PatchHistoryDB = PatchHistoryDB
sys.modules["menace.code_database"] = code_db_mod
sys.modules["code_database"] = code_db_mod

# ---------------------------------------------------------------------------
# Import target modules
# ---------------------------------------------------------------------------
import menace.self_coding_manager as scm

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_thresholds_from_helper():
    data_bot = scm.DataBot()
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(),
        scm.ModelAutomationPipeline(),
        bot_name="alpha",
        data_bot=data_bot,
        bot_registry=scm.BotRegistry(),
    )
    assert mgr.roi_drop_threshold == -0.5
    assert mgr.error_rate_threshold == 2.0

def test_threshold_overrides():
    data_bot = scm.DataBot()
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(),
        scm.ModelAutomationPipeline(),
        bot_name="beta",
        data_bot=data_bot,
        bot_registry=scm.BotRegistry(),
        roi_drop_threshold=-0.2,
        error_rate_threshold=3.3,
    )
    assert mgr.roi_drop_threshold == -0.2
    assert mgr.error_rate_threshold == 3.3


def test_bootstrap_fast_without_validation_flag_disables_fast_path(
    monkeypatch, caplog
):
    dummy_module = types.ModuleType("dummy_settings_mod")
    sys.modules["dummy_settings_mod"] = dummy_module

    class DummySettings:
        __module__ = "dummy_settings_mod"

    caplog.set_level(logging.WARNING, logger="SelfCodingManager")

    monkeypatch.setattr(
        scm.SelfCodingManager, "_get_settings", lambda self: DummySettings()
    )
    monkeypatch.setattr(
        scm,
        "BotRegistry",
        type("BotRegistry", (), {"register_bot": lambda *a, **k: None}),
    )

    class DummyPipeline:
        pass

    manager = scm.SelfCodingManager(
        scm.SelfCodingEngine(),
        DummyPipeline(),
        bot_name="gamma",
        data_bot=scm.DataBot(),
        bot_registry=scm.BotRegistry(),
        bootstrap_fast=True,
    )

    assert manager.bootstrap_fast is False
    assert any(
        "falling back to full validation" in record.message for record in caplog.records
    )


def test_should_refactor_on_failed_tests(monkeypatch):
    class DummyDataBot:
        def __init__(self) -> None:
            self.failures = 0

        def roi(self, bot):
            return 1.0

        def average_errors(self, bot):
            return 0.0

        def average_test_failures(self, bot):
            return self.failures

        def get_thresholds(self, bot):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=2.0
            )

        def reload_thresholds(self, bot):
            return self.get_thresholds(bot)

        def update_thresholds(self, *a, **k):  # pragma: no cover - simple stub
            pass

        def check_degradation(self, _bot, _roi, _err, failures):  # pragma: no cover - simple stub
            return failures > 2

    data_bot = DummyDataBot()
    class Engine:
        def __init__(self):
            builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    mgr = scm.SelfCodingManager(
        Engine(),
        scm.ModelAutomationPipeline(),
        bot_name="alpha",
        data_bot=data_bot,
        bot_registry=scm.BotRegistry(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )
    mgr._last_errors = data_bot.average_errors("alpha")
    data_bot.failures = 1
    assert not mgr.should_refactor()
    data_bot.failures = 5
    assert mgr.should_refactor()


def test_missing_context_builder_raises(tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(context_builder=object())

    mgr = scm.SelfCodingManager(
        Engine(),
        scm.ModelAutomationPipeline(),
        bot_name="x",
        data_bot=scm.DataBot(),
        bot_registry=scm.BotRegistry(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    path = tmp_path / "mod.py"
    path.write_text("print('hi')\n")
    with pytest.raises(ValueError, match="ContextBuilder"):
        mgr.generate_and_patch(path, "desc")


def test_missing_quick_fix_engine_raises(monkeypatch):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(context_builder=object())

    mgr = scm.SelfCodingManager(
        Engine(),
        scm.ModelAutomationPipeline(),
        bot_name="x",
        data_bot=scm.DataBot(),
        bot_registry=scm.BotRegistry(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="tok"
        ),
    )
    monkeypatch.setattr(scm, "QuickFixEngine", None)
    mgr.quick_fix = None
    with pytest.raises(RuntimeError, match="QuickFixEngine"):
        mgr._ensure_quick_fix_engine(scm.ContextBuilder())


def test_manager_timeout_generic_default(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_GENERICBOT",
        raising=False,
    )
    monkeypatch.delenv("SELF_CODING_MANAGER_TIMEOUT_SECONDS_GENERICBOT", raising=False)

    assert scm._resolve_manager_timeout_seconds("GenericBot") == 45.0


def test_manager_timeout_generic_bot_override(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_GENERICBOT", "77"
    )

    assert scm._resolve_manager_timeout_seconds("GenericBot") == 77.0


def test_manager_timeout_generic_malformed_env_falls_back(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_GENERICBOT", "invalid"
    )
    monkeypatch.delenv("SELF_CODING_MANAGER_TIMEOUT_SECONDS_GENERICBOT", raising=False)

    assert scm._resolve_manager_timeout_seconds("GenericBot") == 45.0


def test_manager_timeout_logs_resolution_source(monkeypatch, caplog):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )

    with caplog.at_level(logging.INFO):
        assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 105.0

    assert "resolved manager construction timeout" in caplog.text
    assert "botplanningbot_default_fallback" in caplog.text


def test_manager_timeout_per_bot_override_precedence(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "110"
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT", "135"
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 110.0


def test_manager_timeout_invalid_env_falls_back_safely(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "invalid"
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT", "118"
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 118.0


def test_manager_timeout_botplanningbot_default_fallback(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 105.0


def test_manager_timeout_warns_for_heavy_bot_below_min(monkeypatch, caplog):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(scm, "_HEAVY_MANAGER_TIMEOUT_MIN_SECONDS", 90.0)
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "60"
    )

    with caplog.at_level(logging.WARNING):
        assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 60.0

    assert "below recommended 90.00s" in caplog.text

def test_compute_adaptive_retry_timeout_uses_success_history_for_late_phase_timeout(monkeypatch):
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda _bot_name, primary_timeout: primary_timeout,
    )

    retry_timeout = scm._compute_adaptive_manager_retry_timeout_seconds(
        "BotPlanningBot",
        primary_timeout=10.0,
        timeout_phase="manager_init:deferred_scope",
        timeout_phase_elapsed_seconds=9.0,
        timeout_history=[("queued", 1.0), ("manager_init:enter", 2.0)],
        phase_metrics={
            "manager_init:deferred_scope": {
                "successful_elapsed_seconds": [12.0, 14.0],
            }
        },
    )

    assert retry_timeout == pytest.approx(15.4)


def test_compute_adaptive_retry_timeout_is_capped_and_ignored_after_return(monkeypatch):
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda _bot_name, primary_timeout: primary_timeout * 1.25,
    )

    # Hard cap at 2x primary timeout
    retry_timeout = scm._compute_adaptive_manager_retry_timeout_seconds(
        "BotPlanningBot",
        primary_timeout=10.0,
        timeout_phase="manager_init:deferred_scope",
        timeout_phase_elapsed_seconds=1.0,
        timeout_history=[("manager_init:return", 3.0)],
        phase_metrics={
            "manager_init:deferred_scope": {
                "successful_elapsed_seconds": [50.0],
            }
        },
    )

    assert retry_timeout == 12.5


