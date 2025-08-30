import os
import types
import sys
from pathlib import Path
import subprocess
import shutil
import pytest

# Stub out environment_bootstrap if required
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)

# Data bot shim
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db
sys.modules["data_bot"] = db

# Misc optional deps
sys.modules["menace"].RAISE_ERRORS = False
ns = types.ModuleType("neurosales")
ns.add_message = lambda *a, **k: None
ns.get_history = lambda *a, **k: []
ns.get_recent_messages = lambda *a, **k: []
ns.list_conversations = lambda *a, **k: []
ns.push_chain = lambda *a, **k: None
ns.peek_chain = lambda *a, **k: None
sys.modules.setdefault("neurosales", ns)

# Model automation pipeline / ROI stubs
mapl_stub = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi
class ModelAutomationPipeline: ...
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


class DummyPatchLogger:
    def __init__(self):
        self.calls = []
    def track_contributors(self, *a, **k):
        self.calls.append((a, k))


class DummyContextBuilder:
    def __init__(self):
        self.calls = []
    def query(self, q, *, exclude_tags=None):
        self.calls.append(exclude_tags)
        return "ctx", "sid"


class DummyEngine:
    def __init__(self, builder, logger):
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_logger = logger
        self.calls = []
    def apply_patch(self, path: Path, desc: str, **kwargs):
        self.calls.append(kwargs.get("context_meta"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPipeline:
    def run(self, model: str, energy: int = 1) -> AutomationResult:
        roi = prb_stub.ROIResult(roi=1.0, confidence=0.9)
        return AutomationResult(package=None, roi=roi)


class DummyDataBot:
    def __init__(self):
        self._vals = [0.0, 1.0]
        self.logged = []
    def roi(self, _name):
        return self._vals.pop(0)
    def log_evolution_cycle(self, *a, **k):
        self.logged.append(k)


class DummyRunner:
    def __init__(self):
        self.calls = 0
        self.safe_mode = None
    def run(self, workflow, *, safe_mode=False, **kw):
        self.safe_mode = safe_mode
        self.calls += 1
        if self.calls == 1:
            mod = types.SimpleNamespace(success=False, exception="AssertionError: boom")
            return types.SimpleNamespace(modules=[mod])
        workflow()
        mod = types.SimpleNamespace(success=True, exception=None)
        return types.SimpleNamespace(modules=[mod])


@pytest.mark.parametrize("backend", ["venv", "docker"])
def test_closed_loop_patch(monkeypatch, tmp_path, backend):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    pushes = []
    def fake_run(cmd, *a, cwd=None, check=None, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[:2] == ["git", "push"]:
            pushes.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    tmpdir_path = tmp_path / "clone"
    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)
        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)
    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    runner = DummyRunner()
    monkeypatch.setattr(scm, "WorkflowSandboxRunner", lambda: runner)
    monkeypatch.setattr(scm.ErrorParser, "parse", staticmethod(lambda t: {"tags": ["boom"]}))
    monkeypatch.setattr(scm.MutationLogger, "log_mutation", lambda *a, **k: 1)
    monkeypatch.setattr(scm.MutationLogger, "record_mutation_outcome", lambda *a, **k: None)

    builder = DummyContextBuilder()
    logger = DummyPatchLogger()
    engine = DummyEngine(builder, logger)
    pipeline = DummyPipeline()
    data_bot = DummyDataBot()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)

    mgr.run_patch(file_path, "change")

    assert builder.calls == [["boom"]]
    assert pushes and pushes[-1][-1].endswith("main")
    assert data_bot.logged and data_bot.logged[0]["roi_delta"] == 1.0
    assert logger.calls and logger.calls[0][1]["contribution"] == 1.0
    assert runner.safe_mode is True
