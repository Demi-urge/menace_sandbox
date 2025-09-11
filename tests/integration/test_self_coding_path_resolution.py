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
    def __init__(self, engine=None, pipeline=None, *, bot_name="bot", data_bot=None, event_bus=None):
        self.engine = engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.event_bus = event_bus
        self._last_patch_id = None

    def register_patch_cycle(self, description, context_meta=None):
        roi = self.data_bot.roi(self.bot_name) if self.data_bot else 0.0
        errors = self.data_bot.average_errors(self.bot_name) if self.data_bot else 0.0
        patch_db = getattr(self.engine, "patch_db", None)
        if patch_db:
            self._last_patch_id = patch_db.add(
                filename=f"{self.bot_name}.cycle",
                description=description,
                roi_before=roi,
                roi_after=roi,
                errors_before=int(errors),
                errors_after=int(errors),
                source_bot=self.bot_name,
            )
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered",
                {
                    "bot": self.bot_name,
                    "description": description,
                    "roi_before": roi,
                    "errors_before": errors,
                },
            )


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
spec = importlib.util.spec_from_file_location(
    "dynamic_path_router", root_dir / "dynamic_path_router.py"  # path-ignore
)
dpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpr)
sys.modules["dynamic_path_router"] = dpr

from menace.self_coding_scheduler import SelfCodingScheduler  # noqa: E402


# ---------------------------------------------------------------------------
def test_scheduler_resolves_paths_from_menace_roots(monkeypatch, tmp_path):
    repo_a = tmp_path / "repo_a"
    (repo_a / ".git").mkdir(parents=True)
    repo_b = tmp_path / "repo_b"
    (repo_b / ".git").mkdir(parents=True)
    helper = repo_b / "auto_helpers.py"  # path-ignore
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


# ---------------------------------------------------------------------------
def test_register_patch_cycle_logs_history(tmp_path):
    class StubPatchDB:
        def __init__(self, path):
            import sqlite3

            self.conn = sqlite3.connect(path)
            self.conn.execute(
                """
                CREATE TABLE patch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    roi_before REAL,
                    roi_after REAL,
                    errors_before INTEGER,
                    errors_after INTEGER,
                    source_bot TEXT
                )
                """
            )

        def add(self, **rec):
            cur = self.conn.execute(
                """
                INSERT INTO patch_history (
                    description, roi_before, roi_after, errors_before, errors_after, source_bot
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["description"],
                    rec["roi_before"],
                    rec["roi_after"],
                    rec["errors_before"],
                    rec["errors_after"],
                    rec["source_bot"],
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    patch_db = StubPatchDB(tmp_path / "p.db")
    engine = types.SimpleNamespace(patch_db=patch_db)

    class DummyDataBot:
        def roi(self, name: str) -> float:
            return 1.5

        def average_errors(self, name: str) -> float:
            return 2.5

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic: str, payload: dict) -> None:
            events.append((topic, payload))

    mgr = SelfCodingManager(
        engine, None, bot_name="bot", data_bot=DummyDataBot(), event_bus=Bus()
    )
    mgr.register_patch_cycle("cycle", {})

    rows = list(
        patch_db.conn.execute(
            "SELECT description, roi_before, errors_before, source_bot FROM patch_history"
        )
    )
    assert rows and rows[0][0] == "cycle"
    assert rows[0][1] == pytest.approx(1.5)
    assert rows[0][2] == 2
    assert rows[0][3] == "bot"
    assert events and events[0][0] == "self_coding:cycle_registered"
