import importlib.machinery
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("MENACE_LOCAL_DB_PATH", "/tmp/menace_local.db")
os.environ.setdefault("MENACE_SHARED_DB_PATH", "/tmp/menace_shared.db")
os.environ.setdefault("SANDBOX_REPO_PATH", str(Path.cwd()))

def _fake_connection(*_args, **_kwargs):
    return types.SimpleNamespace(fetchone=lambda: None)


def _fake_get_connection(*_args, **_kwargs):
    return types.SimpleNamespace(execute=lambda *_a, **_k: _fake_connection())


db_router_stub = types.ModuleType("db_router")
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.init_db_router = lambda *a, **k: types.SimpleNamespace(
    get_connection=_fake_get_connection
)
db_router_stub.DBRouter = object
db_router_stub.LOCAL_TABLES = set()
db_router_stub.__spec__ = importlib.machinery.ModuleSpec("db_router", loader=None)
sys.modules.pop("db_router", None)
sys.modules["db_router"] = db_router_stub
sys.modules.pop("menace_sandbox.db_router", None)
sys.modules["menace_sandbox.db_router"] = db_router_stub

_bootstrap_stub = types.ModuleType("sandbox_runner.bootstrap")
_bootstrap_stub.resolve_path = lambda *a, **k: Path(".")
_bootstrap_stub.repo_root = lambda: Path(".")
_bootstrap_stub.path_for_prompt = lambda value: str(value)
_bootstrap_stub.initialize_autonomous_sandbox = lambda *a, **k: None
_bootstrap_stub.__spec__ = importlib.machinery.ModuleSpec(
    "sandbox_runner.bootstrap", loader=None
)
sys.modules.pop("sandbox_runner.bootstrap", None)
sys.modules["sandbox_runner.bootstrap"] = _bootstrap_stub
sys.modules.pop("menace_sandbox.sandbox_runner.bootstrap", None)
sys.modules["menace_sandbox.sandbox_runner.bootstrap"] = _bootstrap_stub

_task_stub = types.ModuleType("task_handoff_bot")
_task_stub.WorkflowDB = None
_task_stub.WorkflowRecord = None
_task_stub.__spec__ = importlib.machinery.ModuleSpec("task_handoff_bot", loader=None)
sys.modules.pop("task_handoff_bot", None)
sys.modules["task_handoff_bot"] = _task_stub
sys.modules.pop("menace_sandbox.task_handoff_bot", None)
sys.modules["menace_sandbox.task_handoff_bot"] = _task_stub

_summary_stub = types.ModuleType("self_improvement.workflow_summary_db")
_summary_stub.WorkflowSummaryDB = object
_summary_stub.__spec__ = importlib.machinery.ModuleSpec(
    "self_improvement.workflow_summary_db", loader=None
)
sys.modules.pop("self_improvement.workflow_summary_db", None)
sys.modules.pop("workflow_summary_db", None)
sys.modules["self_improvement.workflow_summary_db"] = _summary_stub
sys.modules["workflow_summary_db"] = _summary_stub
sys.modules.pop("menace_sandbox.workflow_summary_db", None)
sys.modules["menace_sandbox.workflow_summary_db"] = _summary_stub

try:
    from self_improvement import engine as engine_mod
except Exception as exc:  # pragma: no cover - optional dependency missing
    pytest.skip(
        f"self_improvement.engine unavailable: {exc}", allow_module_level=True
    )


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class FakeQuickFix:
    def __init__(self):
        self.calls: list[tuple[str, str, dict | None, dict]] = []

    def apply_validated_patch(self, module_path: str, description: str, context_meta=None, **kwargs):
        self.calls.append((module_path, description, context_meta, kwargs))
        return True, 101, []


class DummySandboxSettings:
    sandbox_score_db = "sandbox.db"

    def __init__(self):
        self.sandbox_score_db = "sandbox.db"


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("print('hello')\n", encoding="utf-8")
    return repo


def test_patch_validation_runs_self_tests_and_rolls_back(monkeypatch, temp_repo):
    engine = engine_mod.SelfImprovementEngine.__new__(engine_mod.SelfImprovementEngine)
    engine.logger = DummyLogger()
    engine._memory_summaries = lambda module: []
    engine.next_prompt_strategy = lambda: None
    engine.strategy_confidence = {}
    engine.baseline_tracker = types.SimpleNamespace(current=lambda _: 0.0)
    engine._save_state = lambda: None
    engine.data_bot = None
    engine.patch_db = None
    engine.metrics_db = None
    engine.module_index = 0
    engine.gpt_memory = object()
    engine.local_knowledge = object()
    engine._record_memory_outcome = lambda *a, **k: None
    engine.bot_name = "bot"
    engine.quick_fix = FakeQuickFix()
    engine.self_coding_engine = types.SimpleNamespace(
        llm_client=None,
        context_builder=object(),
        quick_fix=None,
        data_bot=None,
    )

    log_records: list[tuple] = []
    monkeypatch.setattr(engine_mod, "log_with_tags", lambda *a, **k: None)

    def _fake_log_prompt_attempt(prompt, success, exec_res, roi_meta, **kwargs):
        log_records.append((prompt, success, exec_res, roi_meta, kwargs))

    monkeypatch.setattr(engine_mod, "log_prompt_attempt", _fake_log_prompt_attempt)
    monkeypatch.setattr(engine_mod, "snapshot_tracker", types.SimpleNamespace(save_checkpoint=lambda *a, **k: None))
    monkeypatch.setattr(engine_mod, "SandboxSettings", DummySandboxSettings)
    monkeypatch.setattr(engine_mod, "get_latest_sandbox_score", lambda _db: 0.0)
    monkeypatch.setattr(engine_mod, "capture_snapshot", lambda **_: {})
    monkeypatch.setattr(
        engine_mod,
        "snapshot_delta",
        lambda _pre, _post: {
            "roi": 1.0,
            "sandbox_score": 1.0,
            "entropy": 1.0,
            "call_graph_complexity": 1.0,
            "token_diversity": 1.0,
        },
    )
    monkeypatch.setattr(engine_mod, "apply_patch", lambda patch_id, repo: ("newhash", "diff"))
    monkeypatch.setattr(engine_mod, "_repo_path", lambda: temp_repo)
    monkeypatch.setattr(
        engine_mod,
        "resolve_path",
        lambda path: (temp_repo / path) if not Path(path).is_absolute() else Path(path),
    )

    services: list[dict] = []

    class DummySelfTestService:
        def __init__(self, **kwargs):
            services.append(kwargs)

        def run_once(self):
            return (
                {
                    "passed": 0,
                    "failed": 1,
                    "coverage": 0.5,
                    "runtime": 1.0,
                    "retry_errors": {"suite": [{"test": "tests/test_sample.py::test_failure"}]},
                    "orphan_failed_modules": ["module"],
                },
                [],
            )

    monkeypatch.setattr(engine_mod, "SelfTestService", DummySelfTestService)

    commands: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        commands.append(list(cmd))

        class Result:
            def __init__(self, stdout: str = "") -> None:
                self.stdout = stdout
                self.stderr = ""
                self.returncode = 0

        if cmd[:2] == ["git", "rev-parse"]:
            return Result(stdout="oldhash\n")
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            cloned_module = dest / "module.py"
            cloned_module.write_text("print('hello')\n", encoding="utf-8")
            return Result()
        if cmd[:3] == ["git", "reset", "--hard"]:
            return Result()
        return Result()

    monkeypatch.setattr(engine_mod.subprocess, "run", _fake_run)

    patch_id = engine._generate_patch_with_memory("module", "improve")

    assert patch_id is None, "validation failure should roll back the patch"
    assert engine.quick_fix.calls, "quick fix validation should be invoked"
    assert services, "self tests should be executed"
    assert any(cmd[:3] == ["git", "reset", "--hard"] for cmd in commands)
    assert log_records, "validation results should be logged"

    _prompt, success, exec_res, roi_meta, extras = log_records[-1]
    assert not success
    assert exec_res.get("validation", {}).get("self_tests", {}).get("failed") == 1
    assert roi_meta.get("tests_passed") is False
    assert roi_meta.get("validation", {}).get("quick_fix", {}).get("passed") is True
