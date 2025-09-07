import types
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import pytest
from dynamic_path_router import resolve_path

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

# Minimal menace package to avoid executing heavy __init__
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules.setdefault("menace", menace_pkg)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Stubs for modules referenced by SelfCodingManager
sce_mod = types.ModuleType("menace.self_coding_engine")
sce_mod.SelfCodingEngine = object
sys.modules.setdefault("menace.self_coding_engine", sce_mod)

mapl_mod = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    def __init__(self, roi=None):
        self.roi = roi
class ModelAutomationPipeline:
    def __init__(self, *a, **k):
        pass

    def run(self, *a, **k):
        return AutomationResult()
mapl_mod.AutomationResult = AutomationResult
mapl_mod.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules.setdefault("menace.model_automation_pipeline", mapl_mod)

adv_mod = types.ModuleType("menace.advanced_error_management")
class FormalVerifier:
    def verify(self, path: Path) -> bool:
        return True
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
    def rollback(self, *a, **k):
        pass
rb_mod.RollbackManager = RollbackManager
sys.modules.setdefault("menace.rollback_manager", rb_mod)

db_mod = types.ModuleType("menace.data_bot")
class DataBot:
    pass
db_mod.DataBot = DataBot
sys.modules.setdefault("menace.data_bot", db_mod)

thr_mod = types.ModuleType("menace.sandbox_runner.test_harness")
class HarnessResult:
    def __init__(self, success: bool, failure=None, stdout="", stderr="", duration=0.0):
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
from patch_suggestion_db import PatchSuggestionDB
from menace.sandbox_runner.test_harness import TestHarnessResult as HarnessResultClass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_failed_tags_recorded(monkeypatch, tmp_path):
    file_path = tmp_path / resolve_path("sample.py")
    file_path.write_text("def x():\n    pass\n")

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=None)
            self.last_prompt_text = ""

        def apply_patch(self, path: Path, desc: str, **kwargs):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    engine = Engine()
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    pipeline = ModelAutomationPipeline(context_builder=builder)
    mgr = scm.SelfCodingManager(engine, pipeline)

    # ensure clone path exists and file copied during git clone
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)
        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    # failing test harness result to trigger ErrorParser tag extraction
    failure_result = HarnessResultClass(
        success=False,
        failure=None,
        stdout="",
        stderr="Traceback (most recent call last):\nValueError: boom",
        duration=0.0,
    )
    monkeypatch.setattr(
        scm, "run_tests", lambda repo, path, *, backend="venv": failure_result
    )

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "add", max_attempts=2)

    assert "value_error" in engine.patch_suggestion_db.failed_strategy_tags()
