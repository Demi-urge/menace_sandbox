import subprocess
import types
import shutil
import sys
from pathlib import Path

# Stub heavy vector_service modules before importing manager
vec_ctx = types.ModuleType("vector_service.context_builder")
vec_ctx.ContextBuilder = object  # placeholder
vec_ctx.ensure_fresh_weights = lambda builder: None
vec_ctx.record_failed_tags = lambda *a, **k: None
vec_ctx.load_failed_tags = lambda: None
sys.modules["vector_service.context_builder"] = vec_ctx

vec_ret = types.ModuleType("vector_service.retriever")
vec_ret.Retriever = object
vec_ret.PatchRetriever = object
vec_ret.FallbackResult = object
sys.modules["vector_service.retriever"] = vec_ret

vec_log = types.ModuleType("vector_service.patch_logger")
vec_log._VECTOR_RISK = None
sys.modules["vector_service.patch_logger"] = vec_log

# Stub test harness to avoid heavy sandbox imports
th = types.ModuleType("menace_sandbox.sandbox_runner.test_harness")
class TestHarnessResult(types.SimpleNamespace):
    pass
def run_tests(repo, path, backend="venv"):
    return TestHarnessResult(success=True, stdout="1 passed", duration=0.1)
th.run_tests = run_tests
th.TestHarnessResult = TestHarnessResult
sys.modules["menace_sandbox.sandbox_runner.test_harness"] = th

# Minimal stubs for heavy internal modules
sc_eng = types.ModuleType("menace_sandbox.self_coding_engine")
class SelfCodingEngine:
    def __init__(self, *a, **k):
        self.cognition_layer = types.SimpleNamespace()
sc_eng.SelfCodingEngine = SelfCodingEngine
sys.modules["menace_sandbox.self_coding_engine"] = sc_eng

pipeline_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")
class AutomationResult(types.SimpleNamespace):
    pass
class ModelAutomationPipeline:
    def __init__(self, *a, **k):
        pass
    def run(self, bot, energy=1):
        return AutomationResult()
pipeline_mod.AutomationResult = AutomationResult
pipeline_mod.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace_sandbox.model_automation_pipeline"] = pipeline_mod

data_mod = types.ModuleType("menace_sandbox.data_bot")
class DataBot:
    pass
data_mod.DataBot = DataBot
sys.modules["menace_sandbox.data_bot"] = data_mod

error_mod = types.ModuleType("menace_sandbox.error_bot")
class ErrorDB:
    pass
error_mod.ErrorDB = ErrorDB
sys.modules["menace_sandbox.error_bot"] = error_mod

adv_mod = types.ModuleType("menace_sandbox.advanced_error_management")
class FormalVerifier:
    pass
class AutomatedRollbackManager:
    pass
adv_mod.FormalVerifier = FormalVerifier
adv_mod.AutomatedRollbackManager = AutomatedRollbackManager
sys.modules["menace_sandbox.advanced_error_management"] = adv_mod

mut_mod = types.ModuleType("menace_sandbox.mutation_logger")
def log_mutation(*a, **k):
    pass
mut_mod.log_mutation = log_mutation
mut_mod.record_mutation_outcome = lambda *a, **k: None
sys.modules["menace_sandbox.mutation_logger"] = mut_mod

roll_mod = types.ModuleType("menace_sandbox.rollback_manager")
class RollbackManager:
    def rollback(self, *a, **k):
        pass
roll_mod.RollbackManager = RollbackManager
sys.modules["menace_sandbox.rollback_manager"] = roll_mod

code_mod = types.ModuleType("menace_sandbox.code_database")
class PatchRecord:
    pass
code_mod.PatchRecord = PatchRecord
sys.modules["menace_sandbox.code_database"] = code_mod

# Stub self_improvement package with minimal baseline tracker and target region
si_pkg = types.ModuleType("menace_sandbox.self_improvement")
si_pkg.__path__ = []
sys.modules["menace_sandbox.self_improvement"] = si_pkg
bt_mod = types.ModuleType("menace_sandbox.self_improvement.baseline_tracker")
class BaselineTracker:
    def __init__(self, window=5, metrics=None):
        pass
    def update(self, **kw):
        pass
    def get(self, key):
        return 0.0
    def std(self, key):
        return 0.0
bt_mod.BaselineTracker = BaselineTracker
sys.modules["menace_sandbox.self_improvement.baseline_tracker"] = bt_mod
tr_mod = types.ModuleType("menace_sandbox.self_improvement.target_region")
class TargetRegion:
    pass
tr_mod.TargetRegion = TargetRegion
sys.modules["menace_sandbox.self_improvement.target_region"] = tr_mod

pp_mod = types.ModuleType("menace_sandbox.patch_provenance")
pp_mod.record_patch_metadata = lambda *a, **k: None
sys.modules["menace_sandbox.patch_provenance"] = pp_mod

# Stub coding bot interface and thresholds
cbi_mod = types.ModuleType("menace_sandbox.coding_bot_interface")
def manager_generate_helper(manager, description, *, context_builder, **kwargs):
    return ""
cbi_mod.manager_generate_helper = manager_generate_helper
sys.modules["menace_sandbox.coding_bot_interface"] = cbi_mod
sct_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")
def update_thresholds(*a, **k):
    pass
sct_mod.update_thresholds = update_thresholds
sct_mod.get_thresholds = lambda bot: types.SimpleNamespace(
    roi_drop=0.0, error_threshold=0.0, test_failure_threshold=0.0
)
sys.modules["menace_sandbox.self_coding_thresholds"] = sct_mod

import menace_sandbox.self_coding_manager as scm


def test_baseline_updates_after_patch(tmp_path, monkeypatch):
    # create simple git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    file_path = tmp_path / "mod.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda builder: None)
    monkeypatch.setattr(scm, "ContextBuilder", lambda: types.SimpleNamespace())

    # fake git operations
    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["git", "clone"]:
            src, dest = cmd[2], cmd[3]
            shutil.copytree(src, dest, dirs_exist_ok=True)
        return types.SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(scm.subprocess, "check_output", lambda *a, **k: b"deadbeef")

    class DummyQuickFix:
        def __init__(self, *a, **k):
            pass
        def apply_validated_patch(self, module_path, desc, ctx_meta):
            return True, 1, []
        def validate_patch(self, module_path, desc, repo_root=None):
            return True, []

    class DummyDataBot:
        def __init__(self):
            self.roi_value = 1.0
            self.errors_value = 1.0
            self.failures_value = 0.0
            self.collected = []
        def roi(self, _name):
            return self.roi_value
        def average_errors(self, _name):
            return self.errors_value
        def average_test_failures(self, _name):
            return self.failures_value
        def get_thresholds(self, _bot):
            return types.SimpleNamespace(
                roi_drop=0.0, error_threshold=0.0, test_failure_threshold=0.0
            )
        def reload_thresholds(self, _bot):
            return self.get_thresholds(_bot)
        def check_degradation(self, *a, **k):
            return True
        def log_evolution_cycle(self, *a, **k):
            pass
        def record_test_failure(self, *a, **k):
            pass
        def forecast_roi_drop(self):
            return 0.0
        def collect(self, bot, revenue=0.0, errors=0, tests_failed=0, tests_run=0, **kw):
            self.roi_value = revenue
            self.errors_value = float(errors)
            self.failures_value = float(tests_failed)
            self.collected.append((revenue, errors, tests_failed, tests_run))

    class DummyPipeline:
        def run(self, bot, energy=1):
            data_bot.roi_value = 2.0
            data_bot.errors_value = 0.0
            return types.SimpleNamespace(roi=None)

    class DummyRegistry:
        def __init__(self):
            self.graph = {}
        def register_bot(self, name):
            self.graph.setdefault(name, {})
        def update_bot(self, name, module_path, *, patch_id=None, commit=None):
            self.graph.setdefault(name, {})
            self.graph[name].update({"module": module_path})
        def record_interaction_metadata(self, *a, **k):
            pass
        def record_heartbeat(self, *a, **k):
            pass

    data_bot = DummyDataBot()
    manager = scm.SelfCodingManager(
        types.SimpleNamespace(cognition_layer=types.SimpleNamespace(context_builder=types.SimpleNamespace())),
        DummyPipeline(),
        bot_name="mod",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=DummyQuickFix(),
    )
    manager.event_bus = None
    manager.bot_registry.hot_swap_bot = lambda *a, **k: None
    manager.bot_registry.register_interaction = lambda *a, **k: None
    manager.bot_registry.record_heartbeat = lambda *a, **k: None
    manager.bot_registry.record_interaction_metadata = lambda *a, **k: None
    manager.bot_registry.health_check_bot = lambda *a, **k: None
    manager.bot_registry.save = lambda *a, **k: None
    manager._refresh_thresholds = lambda: None
    manager.scan_repo = lambda: None

    manager.run_patch(file_path, "desc")

    assert data_bot.collected
    assert data_bot.roi("mod") == 2.0
    assert data_bot.average_errors("mod") == 0.0
