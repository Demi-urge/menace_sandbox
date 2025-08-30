import os
from pathlib import Path
import types
import tempfile
import shutil

import menace.self_coding_manager as scm
import menace.model_automation_pipeline as mapl
import menace.pre_execution_roi_bot as prb


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str, **kwargs):
        self.calls.append(path)
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


def test_ephemeral_clone_removed(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    return 1\n")

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    clone_run = {}

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
            return types.SimpleNamespace(returncode=0)
        if cmd[0] == "pytest":
            clone_run["cwd"] = kw.get("cwd")
            return types.SimpleNamespace(returncode=0)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    class DummyRunner:
        def __init__(self):
            self.safe_mode = None
            self.calls = 0

        def run(self, workflow, *, safe_mode=False, **kwargs):
            self.safe_mode = safe_mode
            self.calls += 1
            workflow()
            return types.SimpleNamespace(modules=[types.SimpleNamespace(result=True)])

    runner = DummyRunner()
    monkeypatch.setattr(
        scm, "WorkflowSandboxRunner", lambda: runner
    )

    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mgr.run_patch(file_path, "add")
    finally:
        os.chdir(cwd)

    assert runner.calls == 1
    assert runner.safe_mode is True
    assert clone_run.get("cwd") == str(tmpdir_path)
    assert not tmpdir_path.exists()
    assert "# patched" in file_path.read_text()
