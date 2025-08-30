import pytest

pytest.importorskip("networkx")
pytest.importorskip("pandas")

import sys
import types
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)
import menace.self_coding_manager as scm
import menace.model_automation_pipeline as mapl
import menace.pre_execution_roi_bot as prb
import menace.data_bot as db
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
    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")

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
        if cmd[0] == "pytest":
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    class DummyRunner:
        def run(self, workflow, *, safe_mode=False, **kw):
            workflow()
            return types.SimpleNamespace(modules=[types.SimpleNamespace(result=True)])

    monkeypatch.setattr(scm, "WorkflowSandboxRunner", lambda: DummyRunner())

    res = mgr.run_patch(file_path, "add")
    assert engine.calls
    assert pipeline.calls
    assert "# patched" in file_path.read_text()
    rows = hist.fetch()
    assert rows and rows[0][0] == "self_coding"
    assert isinstance(res, mapl.AutomationResult)


def test_run_patch_logging_error(monkeypatch, tmp_path, caplog):
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb, evolution_db=hist)
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")

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
        if cmd[0] == "pytest":
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    class DummyRunner:
        def run(self, workflow, *, safe_mode=False, **kw):
            workflow()
            return types.SimpleNamespace(modules=[types.SimpleNamespace(result=True)])

    monkeypatch.setattr(scm, "WorkflowSandboxRunner", lambda: DummyRunner())

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
    file_path = tmp_path / "x.py"
    file_path.write_text("x = 1\n")
    assert policy.approve(file_path)
    assert "failed to log healing action" in caplog.text
