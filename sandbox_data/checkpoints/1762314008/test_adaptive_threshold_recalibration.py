import sys, types, pathlib, importlib.util
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Create lightweight package placeholder
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", pkg)

# ---------------------------------------------------------------------------
# Global threshold store used by stubs
THRESHOLDS = {
    "roi_drop": -0.1,
    "error_threshold": 1.0,
    "test_failure_threshold": 0.0,
}
UPDATED: list[tuple[str, dict]] = []

# ---------------------------------------------------------------------------
# Minimal stub modules required by self_coding_manager
err = types.ModuleType("menace.error_parser")
class FailureCache: ...
class ErrorReport: ...
class ErrorParser: ...
err.FailureCache = FailureCache
err.ErrorReport = ErrorReport
err.ErrorParser = ErrorParser
sys.modules["menace.error_parser"] = err

ff = types.ModuleType("menace.failure_fingerprint_store")
class FailureFingerprint: ...
class FailureFingerprintStore: ...
ff.FailureFingerprint = FailureFingerprint
ff.FailureFingerprintStore = FailureFingerprintStore
sys.modules["menace.failure_fingerprint_store"] = ff

fru = types.ModuleType("menace.failure_retry_utils")
fru.check_similarity_and_warn = lambda *a, **k: None
fru.record_failure = lambda *a, **k: None
sys.modules["menace.failure_retry_utils"] = fru

th = types.ModuleType("menace.sandbox_runner.test_harness")
class TestHarnessResult: ...
th.TestHarnessResult = TestHarnessResult
th.run_tests = lambda *a, **k: TestHarnessResult()
sys.modules["menace.sandbox_runner.test_harness"] = th

sce = types.ModuleType("menace.self_coding_engine")
class SelfCodingEngine:
    def __init__(self):
        self.cognition_layer = SimpleNamespace(context_builder=SimpleNamespace())
sce.SelfCodingEngine = SelfCodingEngine
sys.modules["menace.self_coding_engine"] = sce

map_mod = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult: ...
class ModelAutomationPipeline: ...
map_mod.AutomationResult = AutomationResult
map_mod.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = map_mod

class DataBot:
    def __init__(self, *a, **k):
        pass
    def reload_thresholds(self, bot):
        from menace.self_coding_thresholds import ROIThresholds
        return ROIThresholds(
            THRESHOLDS["roi_drop"],
            THRESHOLDS["error_threshold"],
            THRESHOLDS["test_failure_threshold"],
        )
    def get_thresholds(self, bot):
        from menace.self_coding_thresholds import ROIThresholds
        return ROIThresholds(
            THRESHOLDS["roi_drop"],
            THRESHOLDS["error_threshold"],
            THRESHOLDS["test_failure_threshold"],
        )
    def roi(self, bot):
        return 0.0
    def average_errors(self, bot):
        return 0.0
    def average_test_failures(self, bot):
        return 0.0
    def check_degradation(self, bot, r, e, f):
        return False

db_mod = types.ModuleType("menace.data_bot")
db_mod.DataBot = DataBot
sys.modules["menace.data_bot"] = db_mod

errdb = types.ModuleType("menace.error_bot")
class ErrorDB: ...
errdb.ErrorDB = ErrorDB
sys.modules["menace.error_bot"] = errdb

adv = types.ModuleType("menace.advanced_error_management")
class FormalVerifier: ...
class AutomatedRollbackManager: ...
adv.FormalVerifier = FormalVerifier
adv.AutomatedRollbackManager = AutomatedRollbackManager
sys.modules["menace.advanced_error_management"] = adv

qfe = types.ModuleType("menace.quick_fix_engine")
class QuickFixEngine:
    def __init__(self, *a, **k):
        pass
qfe.QuickFixEngine = QuickFixEngine
qfe.generate_patch = lambda *a, **k: ""
sys.modules["menace.quick_fix_engine"] = qfe

mut = types.ModuleType("menace.mutation_logger")
mut.log_mutation = lambda *a, **k: None
mut.record_mutation_outcome = lambda *a, **k: None
sys.modules["menace.mutation_logger"] = mut

rb = types.ModuleType("menace.rollback_manager")
class RollbackManager: ...
rb.RollbackManager = RollbackManager
sys.modules["menace.rollback_manager"] = rb

# Real baseline tracker
spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.baseline_tracker", ROOT / "self_improvement" / "baseline_tracker.py"
)
bt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt_module)
sys.modules["menace.self_improvement.baseline_tracker"] = bt_module

tr = types.ModuleType("menace.self_improvement.target_region")
class TargetRegion: ...
tr.TargetRegion = TargetRegion
sys.modules["menace.self_improvement.target_region"] = tr

ss = types.ModuleType("menace.sandbox_settings")
class SandboxSettings:
    def __init__(self, baseline_window=3, adaptive_thresholds=True):
        self.baseline_window = baseline_window
        self.adaptive_thresholds = adaptive_thresholds
ss.SandboxSettings = SandboxSettings
sys.modules["menace.sandbox_settings"] = ss

pat = types.ModuleType("menace.patch_attempt_tracker")
class PatchAttemptTracker: ...
pat.PatchAttemptTracker = PatchAttemptTracker
sys.modules["menace.patch_attempt_tracker"] = pat

sct = types.ModuleType("menace.self_coding_thresholds")
class ROIThresholds:
    def __init__(self, roi_drop, error_threshold, test_failure_threshold):
        self.roi_drop = roi_drop
        self.error_threshold = error_threshold
        self.test_failure_threshold = test_failure_threshold
sct.ROIThresholds = ROIThresholds

def update_thresholds(bot, *, roi_drop=None, error_increase=None, test_failure_increase=None, **_):
    if roi_drop is not None:
        THRESHOLDS["roi_drop"] = roi_drop
    if error_increase is not None:
        THRESHOLDS["error_threshold"] = error_increase
    if test_failure_increase is not None:
        THRESHOLDS["test_failure_threshold"] = test_failure_increase
    UPDATED.append((bot, {
        "roi_drop": roi_drop,
        "error_increase": error_increase,
        "test_failure_increase": test_failure_increase,
    }))

sct.get_thresholds = lambda bot, settings=None: ROIThresholds(
    THRESHOLDS["roi_drop"], THRESHOLDS["error_threshold"], THRESHOLDS["test_failure_threshold"]
)
sct.update_thresholds = update_thresholds
sys.modules["menace.self_coding_thresholds"] = sct

psd = types.ModuleType("menace.patch_suggestion_db")
class PatchSuggestionDB: ...
psd.PatchSuggestionDB = PatchSuggestionDB
sys.modules["menace.patch_suggestion_db"] = psd

br = types.ModuleType("menace.bot_registry")
class BotRegistry:
    def register_bot(self, bot):
        pass
br.BotRegistry = BotRegistry
sys.modules["menace.bot_registry"] = br

ueb = types.ModuleType("menace.unified_event_bus")
class UnifiedEventBus:
    def __init__(self):
        self.published = []
    def publish(self, topic, payload):
        self.published.append((topic, payload))
    def subscribe(self, topic, fn):
        pass
ueb.UnifiedEventBus = UnifiedEventBus
sys.modules["menace.unified_event_bus"] = ueb

pp = types.ModuleType("menace.patch_provenance")
pp.record_patch_metadata = lambda *a, **k: None
sys.modules["menace.patch_provenance"] = pp

cd = types.ModuleType("menace.code_database")
class PatchRecord: ...
cd.PatchRecord = PatchRecord
sys.modules["menace.code_database"] = cd

cbi = types.ModuleType("menace.coding_bot_interface")
cbi.manager_generate_helper = lambda *a, **k: ""
sys.modules["menace.coding_bot_interface"] = cbi
sys.modules["coding_bot_interface"] = cbi

vec = types.ModuleType("vector_service.context_builder")
vec.ContextBuilder = type("ContextBuilder", (), {})
vec.record_failed_tags = lambda *a, **k: None
vec.load_failed_tags = lambda: None
sys.modules["vector_service"] = types.ModuleType("vector_service")
sys.modules["vector_service.context_builder"] = vec

# ---------------------------------------------------------------------------
from menace.self_coding_engine import SelfCodingEngine
from menace.model_automation_pipeline import ModelAutomationPipeline
from menace.data_bot import DataBot
from menace.bot_registry import BotRegistry
from menace.unified_event_bus import UnifiedEventBus
from menace.self_coding_manager import SelfCodingManager


def reset():
    THRESHOLDS.update({"roi_drop": -0.1, "error_threshold": 1.0, "test_failure_threshold": 0.0})
    UPDATED.clear()


def make_manager():
    bus = UnifiedEventBus()
    mgr = SelfCodingManager(
        SelfCodingEngine(),
        ModelAutomationPipeline(),
        bot_name="alpha",
        data_bot=DataBot(),
        bot_registry=BotRegistry(),
        event_bus=bus,
    )
    return mgr


def test_roi_threshold_tightens_on_improvement():
    reset()
    mgr = make_manager()
    for r in (1.0, 1.2, 1.4):
        mgr.baseline_tracker.update(roi=r, errors=0.0, tests_failed=0.0)
    mgr._refresh_thresholds()
    assert mgr.roi_drop_threshold > -0.1
    assert UPDATED[-1][1]["roi_drop"] == mgr.roi_drop_threshold


def test_error_threshold_relaxes_on_degradation():
    reset()
    mgr = make_manager()
    for e in (0.0, 1.0, 2.0):
        mgr.baseline_tracker.update(roi=0.0, errors=e, tests_failed=0.0)
    mgr._refresh_thresholds()
    assert mgr.error_rate_threshold > 1.0
    assert UPDATED[-1][1]["error_increase"] == mgr.error_rate_threshold
