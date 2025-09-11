# flake8: noqa
import pytest
from dynamic_path_router import resolve_path

resolve_path("README.md")

pytest.importorskip("networkx")
pytest.importorskip("pandas")

import sys
import types
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db
sys.modules["data_bot"] = db
sys.modules["menace"].RAISE_ERRORS = False
ns = types.ModuleType("neurosales")
ns.add_message = lambda *a, **k: None
ns.get_history = lambda *a, **k: []
ns.get_recent_messages = lambda *a, **k: []
ns.list_conversations = lambda *a, **k: []
sys.modules.setdefault("neurosales", ns)
mapl_stub = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi
class ModelAutomationPipeline: ...
mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub
sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sys.modules["menace.self_coding_engine"] = sce_stub
prb_stub = types.ModuleType("menace.pre_execution_roi_bot")
class ROIResult:
    def __init__(self, roi, errors, proi, perr, risk):
        self.roi = roi
        self.errors = errors
        self.predicted_roi = proi
        self.predicted_errors = perr
        self.risk = risk
prb_stub.ROIResult = ROIResult
sys.modules["menace.pre_execution_roi_bot"] = prb_stub
error_bot_stub = types.ModuleType("menace.error_bot")
error_bot_stub.ErrorDB = object
sys.modules.setdefault("menace.error_bot", error_bot_stub)
aem_stub = types.ModuleType("menace.advanced_error_management")
aem_stub.FormalVerifier = object
aem_stub.AutomatedRollbackManager = object
sys.modules.setdefault("menace.advanced_error_management", aem_stub)
rm_stub = types.ModuleType("menace.rollback_manager")
rm_stub.RollbackManager = object
sys.modules.setdefault("menace.rollback_manager", rm_stub)
mutation_logger_stub = types.ModuleType("menace.mutation_logger")
mutation_logger_stub.log_mutation = lambda *a, **k: None
sys.modules.setdefault("menace.mutation_logger", mutation_logger_stub)
sr_pkg = types.ModuleType("menace.sandbox_runner")
th_stub = types.ModuleType("menace.sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th_stub.TestHarnessResult = types.SimpleNamespace
sr_pkg.test_harness = th_stub
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)
import menace.self_coding_manager as scm
import menace.model_automation_pipeline as mapl
import menace.pre_execution_roi_bot as prb
from menace.evolution_history_db import EvolutionHistoryDB
from pathlib import Path
import subprocess
import tempfile
import shutil
import logging


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str, **_: object):
        self.calls.append((path, desc))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPipeline:
    def __init__(self):
        self.calls = []

    def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
        self.calls.append((model, energy))
        return mapl.AutomationResult(
            package=None,
            roi=prb.ROIResult(1.0, 0.5, 1.0, 0.5, 0.1),
        )


def test_run_patch_logs_evolution(monkeypatch, tmp_path):
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb, evolution_db=hist)
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    calls: list[tuple] = []

    def run_tests_stub(repo, path, *, backend="venv"):
        calls.append((repo, path, backend))
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    res = mgr.run_patch(file_path, "add")
    assert engine.calls
    assert pipeline.calls
    assert calls
    assert "# patched" in file_path.read_text()
    rows = hist.fetch()
    assert any(r[0].startswith("self_coding") for r in rows)
    assert isinstance(res, mapl.AutomationResult)


def test_run_patch_logging_error(monkeypatch, tmp_path, caplog):
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb, evolution_db=hist)
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path: types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        ),
    )

    def fail(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(data_bot, "log_evolution_cycle", fail)
    caplog.set_level(logging.ERROR)
    mgr.run_patch(file_path, "add")
    assert "failed to log evolution cycle" in caplog.text


def test_approval_logs_audit_failure(monkeypatch, tmp_path, caplog):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    class DummyRollback:
        def log_healing_action(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )
    policy = scm.PatchApprovalPolicy(
        verifier=DummyVerifier(),
        rollback_mgr=DummyRollback(),
        bot_name="bot",
    )
    caplog.set_level(logging.ERROR)
    file_path = tmp_path / "x.py"  # path-ignore
    file_path.write_text("x = 1\n")
    assert policy.approve(file_path)
    assert "failed to log healing action" in caplog.text


def test_run_patch_records_patch_outcome(monkeypatch, tmp_path):
    builder = types.SimpleNamespace()

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                calls=[], context_builder=builder
            )

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def __init__(self):
            self._vals = iter([1.0, 2.0])

        def roi(self, _bot: str) -> float:
            return next(self._vals)

        def log_evolution_cycle(self, *a, **k):
            pass
        def average_errors(self, _bot: str) -> float:  # pragma: no cover - simple
            return 0.0

    engine = DummyEngine()

    def record_patch_outcome(session_id, success, contribution=0.0):
        engine.cognition_layer.calls.append((session_id, success, contribution))

    engine.cognition_layer.record_patch_outcome = record_patch_outcome
    pipeline = DummyPipeline()
    data_bot = DummyDataBot()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    mgr.run_patch(
        file_path, "add", context_meta={"retrieval_session_id": "sid"}
    )
    assert engine.cognition_layer.calls == [("sid", True, pytest.approx(1.0))]


def test_generate_and_patch_delegates(monkeypatch, tmp_path):
    calls: list[tuple] = []

    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(context_builder=None)

        def generate_helper(self, *a, **k):
            calls.append(("gen", a, k))

        def apply_patch(self, path: Path, desc: str, **_: object):
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def fake_run_patch(path, desc, **kw):
        calls.append(("patch", path, desc, kw.get("context_builder")))
        return mapl.AutomationResult(None, prb.ROIResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(mgr, "run_patch", fake_run_patch)
    builder = object()
    mgr.generate_and_patch(file_path, "fix", context_builder=builder)
    assert any(c[0] == "gen" for c in calls)
    assert any(c[0] == "patch" and c[1] == file_path and c[3] is builder for c in calls)


def test_generate_and_patch_failure(tmp_path):
    class BadEngine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(context_builder=None)

        def generate_helper(self, *a, **k):
            raise RuntimeError("boom")

        def apply_patch(self, path: Path, desc: str, **_: object):
            return 1, False, 0.0

    engine = BadEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")
    with pytest.raises(scm.HelperGenerationError):
        mgr.generate_and_patch(file_path, "fix")
