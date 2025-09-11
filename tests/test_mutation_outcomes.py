import importlib
import importlib.util
import pathlib
import sys
import logging
import tempfile
import shutil
import types
import subprocess
import os

# Load package to allow relative imports
spec = importlib.util.spec_from_file_location(
    'menace', pathlib.Path(__file__).resolve().parents[1] / '__init__.py'  # path-ignore
)
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules['menace'] = menace_pkg
spec.loader.exec_module(menace_pkg)
db_stub = types.ModuleType("menace.data_bot")
db_stub.DataBot = object
db_stub.MetricsDB = object
sys.modules["menace.data_bot"] = db_stub
sys.modules["data_bot"] = db_stub

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

sie_module = types.ModuleType("menace.self_improvement")
class SelfImprovementEngine:
    def _optimize_self(self):
        patch_id, _, perf = self.self_coding_engine.apply_patch(None)
        event = mutation_logger.log_mutation()
        mutation_logger.record_mutation_outcome(event, perf, perf, perf)
        return patch_id
sie_module.SelfImprovementEngine = SelfImprovementEngine
sys.modules["menace.self_improvement"] = sie_module

scm_module = importlib.import_module('menace.self_coding_manager')
mutation_logger = importlib.import_module('menace.mutation_logger')


class DummyPipeline:
    def run(self, bot_name: str, energy: int = 1):
        class Result:
            roi = type('Roi', (), {'roi': 1.5})()
        return Result()


class DummyDataBot:
    def __init__(self):
        self.calls = 0

    def roi(self, bot_name: str) -> float:
        self.calls += 1
        return [0.5, 1.5][self.calls - 1]

    def log_evolution_cycle(self, *args, **kwargs):
        pass
    def average_errors(self, bot_name: str) -> float:  # pragma: no cover - simple
        return 0.0


class DummyCodingEngine:
    patch_db = None

    def apply_patch(self, *args, **kwargs):
        return 1, False, 1.0


def test_run_patch_records_outcome(monkeypatch, tmp_path):
    recorded = []

    monkeypatch.setattr(mutation_logger, 'log_mutation', lambda **_: 99)
    def capture(event_id, after_metric, roi, performance):
        recorded.append((event_id, after_metric, roi, performance))
    monkeypatch.setattr(mutation_logger, 'record_mutation_outcome', capture)

    manager = scm_module.SelfCodingManager(
        DummyCodingEngine(),
        DummyPipeline(),
        data_bot=DummyDataBot(),
    )

    file_path = tmp_path / 'x.py'  # path-ignore
    file_path.write_text('x = 1\n')

    tmpdir_path = tmp_path / 'clone'

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, 'TemporaryDirectory', lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ['git', 'clone']:
            dst = pathlib.Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm_module.subprocess, 'run', fake_run)

    monkeypatch.setattr(
        scm_module,
        'run_tests',
        lambda repo, path: types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        ),
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        manager.run_patch(file_path, 'test')
    finally:
        os.chdir(cwd)

    assert recorded == [(99, 1.5, 1.5, 1.0)]


class DummySelfCodingEngine:
    patch_db = None

    def apply_patch(self, *args, **kwargs):
        return 5, False, 2.0


def test_optimize_self_records_outcome(monkeypatch):
    recorded = []

    monkeypatch.setattr(mutation_logger, 'log_mutation', lambda **_: 42)
    def capture(event_id, after_metric, roi, performance):
        recorded.append((event_id, after_metric, roi, performance))
    monkeypatch.setattr(mutation_logger, 'record_mutation_outcome', capture)

    engine = sie_module.SelfImprovementEngine.__new__(sie_module.SelfImprovementEngine)
    engine.self_coding_engine = DummySelfCodingEngine()
    engine._last_patch_id = None
    engine._last_mutation_id = None
    engine.logger = logging.getLogger('test')

    engine._optimize_self()

    assert recorded == [(42, 2.0, 2.0, 2.0)]
