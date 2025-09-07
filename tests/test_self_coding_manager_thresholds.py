import os
import pytest
import sys
import types
import subprocess
import tempfile
from pathlib import Path
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

vs = types.ModuleType("vector_service")
class DummyBuilder:
    def __init__(self, *a, **k):
        pass
    def refresh_db_weights(self):
        pass
vs.ContextBuilder = DummyBuilder
vs.CognitionLayer = object
sys.modules["vector_service"] = vs

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path(__file__).resolve().parent.parent)]
menace_pkg.RAISE_ERRORS = False
sys.modules["menace"] = menace_pkg
sys.path.append(str(Path(__file__).resolve().parent.parent))
import shutil
from pathlib import Path

pytest.importorskip("pandas")

# Reuse stubs from the main self_coding_manager tests
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db  # noqa: E402
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


class ModelAutomationPipeline:
    def run(self, model: str, energy: int = 1) -> AutomationResult:
        return AutomationResult(
            package=None,
            roi=prb.ROIResult(1.0, 0.5, 1.0, 0.5, 0.2),
        )


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

import menace.self_coding_manager as scm  # noqa: E402
import menace.model_automation_pipeline as mapl  # noqa: E402
import menace.pre_execution_roi_bot as prb  # noqa: E402
from menace.evolution_history_db import EvolutionHistoryDB  # noqa: E402


class DummyEngine:
    def apply_patch(self, path: Path, desc: str, **_: object):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


def test_dynamic_confidence_threshold_blocks_merge(monkeypatch, tmp_path):
    hist = EvolutionHistoryDB(tmp_path / "e.db")
    mdb = db.MetricsDB(tmp_path / "m.db")
    data_bot = db.DataBot(mdb, evolution_db=hist)
    engine = DummyEngine()
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    pipeline = mapl.ModelAutomationPipeline(context_builder=builder)
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    mgr.baseline_tracker.update(confidence=0.9)

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

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    def run_tests_stub(repo, path, *, backend="venv"):
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    mgr.run_patch(file_path, "add", auto_merge=True, confidence_threshold=0.1)
    assert not any(cmd[:3] == ["git", "merge", "--no-ff"] for cmd in calls)
