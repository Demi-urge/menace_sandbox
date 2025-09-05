import os
import sys
import types
from pathlib import Path
import importlib.util

import pytest
import yaml

# ---------------------------------------------------------------------------
# Stub modules required for importing SelfCodingScheduler
wsr_stub = types.ModuleType("sandbox_runner.workflow_sandbox_runner")


class WorkflowSandboxRunner:
    def run(self, func, safe_mode=False):
        func()
        class Result:
            modules = []
        return Result()


wsr_stub.WorkflowSandboxRunner = WorkflowSandboxRunner
sys.modules.setdefault("sandbox_runner.workflow_sandbox_runner", wsr_stub)

scm_stub = types.ModuleType("menace.self_coding_manager")


class SelfCodingManager:  # pragma: no cover - stub
    pass


scm_stub.SelfCodingManager = SelfCodingManager
sys.modules.setdefault("menace.self_coding_manager", scm_stub)

db_stub = types.ModuleType("menace.data_bot")


class DataBot:  # pragma: no cover - stub
    pass


db_stub.DataBot = DataBot
sys.modules.setdefault("menace.data_bot", db_stub)

aem_stub = types.ModuleType("menace.advanced_error_management")


class AutomatedRollbackManager:  # pragma: no cover - stub
    pass


aem_stub.AutomatedRollbackManager = AutomatedRollbackManager
sys.modules.setdefault("menace.advanced_error_management", aem_stub)

settings_stub = types.ModuleType("menace.sandbox_settings")


class SandboxSettings:  # pragma: no cover - stub
    self_coding_interval = 1
    self_coding_roi_drop = -1.0
    self_coding_error_increase = 1.0


settings_stub.SandboxSettings = SandboxSettings
sys.modules.setdefault("menace.sandbox_settings", settings_stub)

error_stub = types.ModuleType("menace.error_parser")


class ErrorParser:  # pragma: no cover - stub
    @staticmethod
    def parse_failure(trace: str):
        return {"strategy_tag": ""}


error_stub.ErrorParser = ErrorParser
sys.modules.setdefault("menace.error_parser", error_stub)

cms_stub = types.ModuleType("menace.cross_model_scheduler")
cms_stub._SimpleScheduler = None
cms_stub.BackgroundScheduler = None
sys.modules.setdefault("menace.cross_model_scheduler", cms_stub)

# Load real dynamic_path_router before importing scheduler
root_dir = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dynamic_path_router", root_dir / "dynamic_path_router.py")
dpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpr)
sys.modules["dynamic_path_router"] = dpr

from menace.self_coding_scheduler import SelfCodingScheduler


# ---------------------------------------------------------------------------
def test_scheduler_resolves_paths_from_menace_roots(monkeypatch, tmp_path):
    repo_a = tmp_path / "repo_a"
    (repo_a / ".git").mkdir(parents=True)
    repo_b = tmp_path / "repo_b"
    (repo_b / ".git").mkdir(parents=True)
    helper = repo_b / "auto_helpers.py"
    helper.write_text("# helper\n")
    metrics = repo_b / "sandbox_metrics.yaml"
    metrics.write_text("extra_metrics: {}\n")

    monkeypatch.setenv("MENACE_ROOTS", os.pathsep.join([str(repo_a), str(repo_b)]))
    dpr.clear_cache()
    resolved = dpr.resolve_path("auto_helpers.py")
    assert resolved == helper.resolve()
    assert dpr.path_for_prompt("auto_helpers.py") == helper.resolve().as_posix()

    class DummyManager:
        bot_name = "bot"
        engine = types.SimpleNamespace(patch_db=None)

    class DummyDataBot:
        def __init__(self):
            self.db = types.SimpleNamespace(fetch=lambda n: [])

        def roi(self, name: str) -> float:
            return 1.0

    settings = types.SimpleNamespace(
        self_coding_interval=1, self_coding_roi_drop=-1.0, self_coding_error_increase=1.0
    )
    scheduler = SelfCodingScheduler(DummyManager(), DummyDataBot(), settings=settings)
    assert scheduler.patch_path == helper.resolve()

    scheduler._record_cycle_metrics(True, 0)
    data = yaml.safe_load(metrics.read_text())
    assert data["extra_metrics"]["self_coding_cycle_success"] == 1.0


# ---------------------------------------------------------------------------
def test_resolve_path_missing_file_raises(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("MENACE_ROOTS", str(repo))
    dpr.clear_cache()
    with pytest.raises(FileNotFoundError):
        dpr.resolve_path("not_here.txt")
