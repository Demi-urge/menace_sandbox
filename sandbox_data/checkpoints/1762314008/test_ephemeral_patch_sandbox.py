import os
import sys
from pathlib import Path
import types
import tempfile
import shutil


db_stub = types.ModuleType("menace.data_bot")
db_stub.DataBot = object
db_stub.MetricsDB = object
sys.modules["menace.data_bot"] = db_stub
sys.modules["data_bot"] = db_stub

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sys.modules["menace.self_coding_engine"] = sce_stub
pp_stub = types.ModuleType("menace.patch_provenance")
pp_stub.record_patch_metadata = lambda *a, **k: None
sys.modules["menace.patch_provenance"] = pp_stub

mapl_stub = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi
class ModelAutomationPipeline:
    def run(self, model: str, energy: int = 1):
        return AutomationResult()
mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub

prb_stub = types.ModuleType("menace.pre_execution_roi_bot")
class ROIResult:
    def __init__(self, roi, confidence=1.0, errors=0.0, proi=0.0, perr=0.0, risk=0.0):
        self.roi = roi
        self.confidence = confidence
        self.errors = errors
        self.predicted_roi = proi
        self.predicted_errors = perr
        self.risk = risk
prb_stub.ROIResult = ROIResult
sys.modules["menace.pre_execution_roi_bot"] = prb_stub

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
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    return 1\n")

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
            return types.SimpleNamespace(returncode=0)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    run_calls: list[tuple] = []

    def run_tests_stub(repo, path, *, backend="venv"):
        run_calls.append((repo, path, backend))
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mgr.run_patch(file_path, "add")
    finally:
        os.chdir(cwd)

    assert len(run_calls) == 1
    assert not tmpdir_path.exists()
    assert "# patched" in file_path.read_text()
