import types
import sys
from pathlib import Path
import subprocess
import shutil

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
ns.push_chain = lambda *a, **k: None
ns.peek_chain = lambda *a, **k: None
sys.modules.setdefault("neurosales", ns)

op_stub = types.ModuleType("menace.operational_monitor_bot")
op_stub.OperationalMonitoringBot = object
sys.modules.setdefault("menace.operational_monitor_bot", op_stub)

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
    def __init__(self, roi, errors=0.0, proi=0.0, perr=0.0, risk=0.0):
        self.roi = roi
        self.errors = errors
        self.predicted_roi = proi
        self.predicted_errors = perr
        self.risk = risk
prb_stub.ROIResult = ROIResult
sys.modules["menace.pre_execution_roi_bot"] = prb_stub

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sys.modules["menace.self_coding_engine"] = sce_stub
pp_stub = types.ModuleType("menace.patch_provenance")
pp_stub.record_patch_metadata = lambda *a, **k: None
sys.modules["menace.patch_provenance"] = pp_stub

import menace.self_coding_manager as scm
from menace.self_coding_manager import SelfCodingManager

class DummyPatchLogger:
    def __init__(self):
        self.calls = []
    def track_contributors(self, *args, **kwargs):
        self.calls.append((args, kwargs))

class DummyEngine:
    def __init__(self, logger):
        self.patch_logger = logger
    def apply_patch(self, path: Path, desc: str, **_: object):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0

class _AutomationResult:
    def __init__(self, roi):
        self.roi = roi

class _ROIResult:
    def __init__(self, confidence):
        self.roi = 1.0
        self.confidence = confidence

class DummyPipeline:
    def __init__(self, confidence):
        self.confidence = confidence
    def run(self, *_args, **_kw):
        return _AutomationResult(_ROIResult(self.confidence))

class DummyDataBot:
    def __init__(self):
        self._vals = [0.0, 1.0]
    def roi(self, _name):
        return self._vals.pop(0)
    def log_evolution_cycle(self, *a, **k):
        pass
    def average_errors(self, _name):  # pragma: no cover - simple
        return 0.0


def setup(monkeypatch, tmp_path, confidence):
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("x=1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    pushes = []
    def fake_run(cmd, *a, cwd=None, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        elif cmd[:2] == ["git", "push"]:
            pushes.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(scm.MutationLogger, "log_mutation", lambda *a, **k: 1)
    monkeypatch.setattr(scm.MutationLogger, "record_mutation_outcome", lambda *a, **k: None)

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

    patch_logger = DummyPatchLogger()
    engine = DummyEngine(patch_logger)
    pipeline = DummyPipeline(confidence)
    data_bot = DummyDataBot()
    mgr = SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)
    return mgr, file_path, patch_logger, pushes, run_calls


def test_pushes_to_review_branch(monkeypatch, tmp_path):
    mgr, file_path, patch_logger, pushes, calls = setup(monkeypatch, tmp_path, confidence=0.4)
    mgr.run_patch(file_path, "change", confidence_threshold=0.5)
    assert any("review/1" in cmd[-1] for cmd in pushes)
    assert patch_logger.calls and patch_logger.calls[0][1]["contribution"] == 1.0
    assert len(calls) == 1


def test_merges_to_main(monkeypatch, tmp_path):
    mgr, file_path, patch_logger, pushes, calls = setup(monkeypatch, tmp_path, confidence=0.9)
    mgr.run_patch(file_path, "change", confidence_threshold=0.5, auto_merge=True)
    assert any("review/1" in cmd[-1] for cmd in pushes)
    assert any(cmd[-1].endswith("main") for cmd in pushes)
    assert patch_logger.calls and patch_logger.calls[0][1]["contribution"] == 1.0
    assert len(calls) == 1
