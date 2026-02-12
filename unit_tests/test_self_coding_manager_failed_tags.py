# flake8: noqa
import types
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import sqlite3
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
    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return 0.0

    def get_thresholds(self, _bot: str):
        return types.SimpleNamespace(
            roi_drop=-1.0, error_threshold=1.0, test_failure_threshold=1.0
        )


db_mod.DataBot = DataBot
sys.modules.setdefault("menace.data_bot", db_mod)

err_mod = types.ModuleType("menace.error_bot")
class ErrorDB:
    pass
err_mod.ErrorDB = ErrorDB
sys.modules.setdefault("menace.error_bot", err_mod)

# QuickFixEngine stub
qfe_mod = types.ModuleType("menace.quick_fix_engine")
class QuickFixEngine:
    def __init__(self, *a, **k):
        self.context_builder = None
    def apply_validated_patch(self, *a, **k):
        return True, 1, []
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

    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.last_prompt_text = ""

        def apply_patch(self, path: Path, desc: str, **kwargs):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    engine = Engine()
    pipeline = ModelAutomationPipeline(context_builder=builder)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        data_bot=scm.DataBot(),
        bot_registry=scm.BotRegistry(),
    )

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


def test_patch_requires_quick_fix_engine(monkeypatch, tmp_path):
    file_path = tmp_path / resolve_path("sample.py")
    file_path.write_text("def x():\n    pass\n")

    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.last_prompt_text = ""

        def apply_patch(self, path: Path, desc: str, **kwargs):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    engine = Engine()
    pipeline = ModelAutomationPipeline(context_builder=builder)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        data_bot=scm.DataBot(),
        bot_registry=scm.BotRegistry(),
    )

    monkeypatch.setattr(scm, "QuickFixEngine", None, raising=False)

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

    success_result = HarnessResultClass(
        success=True,
        failure=None,
        stdout="",
        stderr="",
        duration=0.0,
    )
    monkeypatch.setattr(
        scm, "run_tests", lambda repo, path, *, backend="venv": success_result
    )

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "add")


def test_objective_integrity_breach_halts_loop_and_rolls_back(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    return 1\n", encoding="utf-8")

    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.last_prompt_text = ""

    engine = Engine()
    pipeline = ModelAutomationPipeline(context_builder=builder)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        data_bot=scm.DataBot(),
        bot_registry=types.SimpleNamespace(register_bot=lambda *a, **k: None, graph=types.SimpleNamespace(nodes={})),
    )

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
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: HarnessResultClass(True, None, "", "", 0.0),
    )

    monkeypatch.setattr(scm, "create_context_builder", lambda: builder)
    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda *_a, **_k: None)

    mgr.data_bot.check_degradation = lambda *a, **k: True
    mgr.refresh_quick_fix_context = lambda: None
    mgr._ensure_quick_fix_engine = lambda *_a, **_k: None
    mgr._enforce_objective_guard = lambda _path: None
    monkeypatch.setattr(mgr.objective_guard, "assert_integrity", lambda: None)
    mgr._last_commit_hash = "commit123"
    mgr._last_patch_id = 77
    monkeypatch.setattr(
        scm,
        "get_patch_by_commit",
        lambda commit: {"commit": commit, "source": "patch_ledger"},
    )

    mgr.quick_fix = types.SimpleNamespace(
        validate_patch=lambda *a, **k: (_ for _ in ()).throw(
            scm.ObjectiveGuardViolation(
                "objective_integrity_breach",
                details={"changed_files": ["reward_dispatcher.py"]},
            )
        )
    )

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    mgr.event_bus = Bus()

    rollbacks: list[str] = []

    class RB:
        def rollback(self, pid: str, requesting_bot: str | None = None) -> None:
            rollbacks.append(pid)

    monkeypatch.setattr(scm, "RollbackManager", lambda: RB())

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.run_patch(file_path, "add", provenance_token="token")

    assert rollbacks == ["commit123"]
    assert mgr._self_coding_paused is True
    breach_payloads = [
        payload for name, payload in events if name == "self_coding:objective_integrity_breach"
    ]
    assert breach_payloads
    assert breach_payloads[0]["changed_objective_files"] == ["reward_dispatcher.py"]

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.run_patch(file_path, "add", provenance_token="token")
    assert rollbacks == ["commit123"]


def test_objective_integrity_breach_partial_rollback_emits_critical(monkeypatch):
    mgr = object.__new__(scm.SelfCodingManager)
    mgr.engine = types.SimpleNamespace(audit_trail=None)
    mgr.bot_name = "bot"
    mgr.logger = types.SimpleNamespace(critical=lambda *a, **k: None, exception=lambda *a, **k: None)
    mgr._objective_breach_lock = scm.threading.Lock()
    mgr._objective_breach_handled = False
    mgr._self_coding_paused = False
    mgr._self_coding_disabled_reason = None
    mgr._last_commit_hash = "commit999"
    mgr._last_patch_id = 88
    mgr.bot_registry = types.SimpleNamespace(graph=types.SimpleNamespace(nodes={"bot": {}}))

    monkeypatch.setattr(
        scm,
        "get_patch_by_commit",
        lambda commit: {"commit": commit, "source": "patch_ledger"},
    )

    calls: list[str] = []

    class RB:
        def rollback(self, pid: str, requesting_bot: str | None = None) -> None:
            calls.append(pid)
            raise RuntimeError("rollback failed")

    monkeypatch.setattr(scm, "RollbackManager", lambda: RB())

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    mgr.event_bus = Bus()

    violation = scm.ObjectiveGuardViolation(
        "objective_integrity_breach",
        details={"changed_files": ["kpi_reward_core.py"]},
    )

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr._handle_objective_integrity_breach(
            violation=violation,
            path=Path("kpi_reward_core.py"),
            stage="test",
        )

    assert calls == ["commit999"]
    assert any(
        name == "self_coding:objective_integrity_breach_rollback_failed"
        for name, _ in events
    )

    mgr._handle_objective_integrity_breach(
        violation=violation,
        path=Path("kpi_reward_core.py"),
        stage="repeat",
    )
    assert calls == ["commit999"]
    assert any(
        name == "self_coding:objective_integrity_breach" and payload.get("already_handled")
        for name, payload in events
    )




def test_objective_integrity_lock_blocks_until_reset_and_manifest_rotation(monkeypatch):
    mgr = object.__new__(scm.SelfCodingManager)
    mgr.bot_name = "bot"
    mgr.logger = types.SimpleNamespace(exception=lambda *a, **k: None)
    mgr.objective_guard = types.SimpleNamespace()
    mgr._objective_breach_lock = scm.threading.Lock()
    mgr._objective_breach_handled = True
    mgr._self_coding_paused = True
    mgr._self_coding_disabled_reason = "objective_integrity_breach"
    mgr._objective_lock_requires_manifest_refresh = True
    mgr._objective_lock_manifest_sha_at_breach = "sha-before"
    mgr.bot_registry = types.SimpleNamespace(graph=types.SimpleNamespace(nodes={"bot": {}}))
    mgr._persist_registry_state = lambda: None
    mgr._record_audit_event = lambda payload: None
    mgr._clear_objective_integrity_lock = lambda: None

    with pytest.raises(RuntimeError, match=r"operator reset \+ objective hash baseline refresh required"):
        mgr._ensure_self_coding_active()

    monkeypatch.setattr(scm, "verify_objective_hash_lock", lambda guard: {"ok": True})
    monkeypatch.setattr(mgr, "_read_manifest_sha", lambda: "sha-before")

    with pytest.raises(RuntimeError, match="refresh objective hash baseline"):
        mgr.reset_objective_integrity_lock(operator_id="ops", reason="try too early")

    monkeypatch.setattr(mgr, "_read_manifest_sha", lambda: "sha-after")
    mgr.reset_objective_integrity_lock(operator_id="ops", reason="manifest rotated")

    assert mgr._self_coding_paused is False
    assert mgr._objective_lock_requires_manifest_refresh is False

def test_auto_run_patch_breach_rolls_back_by_patch_id(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.last_prompt_text = ""
            self.audit_trail = None

    mgr = scm.SelfCodingManager(
        Engine(),
        ModelAutomationPipeline(context_builder=builder),
        data_bot=scm.DataBot(),
        bot_registry=types.SimpleNamespace(register_bot=lambda *a, **k: None, graph=types.SimpleNamespace(nodes={})),
        evolution_orchestrator=types.SimpleNamespace(provenance_token="token"),
    )
    mgr.event_bus = types.SimpleNamespace(publish=lambda *_a, **_k: None)
    mgr._last_patch_id = 501
    mgr._last_commit_hash = None

    calls: list[str] = []

    class RB:
        def rollback(self, pid: str, requesting_bot: str | None = None) -> None:
            calls.append(pid)

    monkeypatch.setattr(scm, "RollbackManager", lambda: RB())

    def raise_breach(*_a, **_k):
        raise scm.ObjectiveGuardViolation(
            "objective_integrity_breach", details={"changed_files": ["objective_guard.py"]}
        )

    monkeypatch.setattr(mgr, "run_patch", raise_breach)

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.auto_run_patch(tmp_path / "sample.py", "desc")

    assert calls == ["501"]
    assert mgr._self_coding_paused is True


def test_idle_cycle_breach_halts_without_second_attempt(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)

    class Engine:
        def __init__(self):
            self.patch_suggestion_db = PatchSuggestionDB(tmp_path / "s.db")
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.last_prompt_text = ""
            self.audit_trail = None

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE suggestions (id INTEGER PRIMARY KEY, module TEXT, description TEXT)"
    )
    conn.execute(
        "INSERT INTO suggestions(id, module, description) VALUES (1, ?, 'first')",
        (str(tmp_path / "first.py"),),
    )
    conn.execute(
        "INSERT INTO suggestions(id, module, description) VALUES (2, ?, 'second')",
        (str(tmp_path / "second.py"),),
    )
    conn.commit()

    events: list[str] = []
    mgr = scm.SelfCodingManager(
        Engine(),
        ModelAutomationPipeline(context_builder=builder),
        data_bot=scm.DataBot(),
        bot_registry=types.SimpleNamespace(register_bot=lambda *a, **k: None, graph=types.SimpleNamespace(nodes={})),
        suggestion_db=types.SimpleNamespace(conn=conn),
        event_bus=types.SimpleNamespace(publish=lambda name, _payload: events.append(name)),
    )
    mgr._last_patch_id = 77
    mgr._last_commit_hash = None

    attempts: list[str] = []

    def raise_once(_path, desc):
        attempts.append(desc)
        raise scm.ObjectiveGuardViolation(
            "objective_integrity_breach",
            details={"changed_files": ["config/objective_hash_lock.json"]},
        )

    monkeypatch.setattr(mgr, "auto_run_patch", raise_once)

    rollbacks: list[str] = []

    class RB:
        def rollback(self, pid: str, requesting_bot: str | None = None) -> None:
            rollbacks.append(pid)

    monkeypatch.setattr(scm, "RollbackManager", lambda: RB())

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.idle_cycle()

    assert attempts == ["first"]
    assert rollbacks == ["77"]
    assert "self_coding:objective_integrity_trip" in events
