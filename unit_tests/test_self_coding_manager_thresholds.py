# flake8: noqa
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
sys.modules.setdefault("vector_service", vec_mod)

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
    pass
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
    pass
db_mod.DataBot = DataBot
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
sys.modules.setdefault("menace.quick_fix_engine", qfe_mod)

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

# ---------------------------------------------------------------------------
# Import target modules
# ---------------------------------------------------------------------------
import menace.self_coding_manager as scm
from menace.self_coding_thresholds import SelfCodingThresholds

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_thresholds_from_helper(monkeypatch):
    calls = {}
    def fake_get_thresholds(bot):
        calls['bot'] = bot
        return SelfCodingThresholds(roi_drop=-0.5, error_increase=2.0)
    monkeypatch.setattr(scm, 'get_thresholds', fake_get_thresholds)
    mgr = scm.SelfCodingManager(scm.SelfCodingEngine(), scm.ModelAutomationPipeline(), bot_name='alpha')
    assert calls['bot'] == 'alpha'
    assert mgr.roi_drop_threshold == -0.5
    assert mgr.error_rate_threshold == 2.0

def test_threshold_overrides(monkeypatch):
    calls = {}
    def fake_get_thresholds(bot):
        calls['bot'] = bot
        return SelfCodingThresholds(roi_drop=-0.5, error_increase=2.0)
    monkeypatch.setattr(scm, 'get_thresholds', fake_get_thresholds)
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(),
        scm.ModelAutomationPipeline(),
        bot_name='beta',
        roi_drop_threshold=-0.2,
        error_rate_threshold=3.3,
    )
    assert calls['bot'] == 'beta'
    assert mgr.roi_drop_threshold == -0.2
    assert mgr.error_rate_threshold == 3.3


def test_should_refactor_on_failed_tests(monkeypatch):
    def fake_get_thresholds(bot):
        return SelfCodingThresholds(
            roi_drop=-999.0, error_increase=999.0, test_failure_increase=2.0
        )

    monkeypatch.setattr(scm, "get_thresholds", fake_get_thresholds)

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

    data_bot = DummyDataBot()
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(),
        scm.ModelAutomationPipeline(),
        bot_name="alpha",
        data_bot=data_bot,
    )
    mgr._last_errors = data_bot.average_errors("alpha")
    data_bot.failures = 1
    assert not mgr.should_refactor()
    data_bot.failures = 5
    assert mgr.should_refactor()


def test_missing_context_builder_raises(tmp_path):
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(), scm.ModelAutomationPipeline(), bot_name="x"
    )
    path = tmp_path / "mod.py"
    path.write_text("print('hi')\n")
    with pytest.raises(RuntimeError, match="ContextBuilder"):
        mgr.generate_and_patch(path, "desc")


def test_missing_quick_fix_engine_raises(monkeypatch):
    mgr = scm.SelfCodingManager(
        scm.SelfCodingEngine(), scm.ModelAutomationPipeline(), bot_name="x"
    )
    monkeypatch.setattr(scm, "quick_fix_engine", None)
    with pytest.raises(RuntimeError, match="QuickFixEngine"):
        mgr._ensure_quick_fix_engine()
