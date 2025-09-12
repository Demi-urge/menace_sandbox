# flake8: noqa
import pytest

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
db_router_stub = types.ModuleType("db_router")
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.LOCAL_TABLES = set()


class DummyRouter:
    def __init__(self, *a, **k):
        pass

    class _Conn:
        def execute(self, *a, **k):
            return types.SimpleNamespace(fetchall=lambda: [])

        def commit(self):
            pass

    def get_connection(self, *_a, **_k):
        return self._Conn()


def init_db_router(*a, **k):
    return DummyRouter()


db_router_stub.DBRouter = DummyRouter
db_router_stub.init_db_router = init_db_router
sys.modules.setdefault("db_router", db_router_stub)
dpr = types.SimpleNamespace(
    resolve_path=lambda p: __import__("pathlib").Path(p),
    repo_root=lambda: __import__("pathlib").Path("."),
    path_for_prompt=lambda p: str(p),
)
sys.modules["dynamic_path_router"] = dpr
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
code_db_stub = types.ModuleType("menace.code_database")
class PatchRecord:
    pass
class PatchHistoryDB:
    pass
code_db_stub.PatchRecord = PatchRecord
code_db_stub.PatchHistoryDB = PatchHistoryDB
sys.modules["menace.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub
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


class DummyRegistry:
    def register_bot(self, name: str) -> None:
        pass


class DummyDataBot:
    def __init__(self) -> None:
        self.failures = 0
        self.db = types.SimpleNamespace(log_eval=lambda *a, **k: None)

    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return self.failures

    def get_thresholds(self, _bot: str):
        return types.SimpleNamespace(
            roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
        )

    def log_evolution_cycle(self, *a, **k) -> None:  # pragma: no cover - simple
        pass


def test_run_patch_logs_evolution(monkeypatch, tmp_path):
    hist = EvolutionHistoryDB(tmp_path / "e.db")

    class LocalDataBot(DummyDataBot):
        def log_evolution_cycle(self, *a, **k) -> None:
            hist.log_cycle("self_coding", {})

    data_bot = LocalDataBot()
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
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

    class LocalDataBot(DummyDataBot):
        def log_evolution_cycle(self, *a, **k) -> None:
            hist.log_cycle("self_coding", {})

    data_bot = LocalDataBot()
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
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

        def average_test_failures(self, _bot: str) -> float:  # pragma: no cover - simple
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

    engine = DummyEngine()

    def record_patch_outcome(session_id, success, contribution=0.0):
        engine.cognition_layer.calls.append((session_id, success, contribution))

    engine.cognition_layer.record_patch_outcome = record_patch_outcome
    pipeline = DummyPipeline()
    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
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


def test_registry_update_and_hot_swap(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace()
            )

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def check_degradation(self, *_):
            return True

        def log_evolution_cycle(self, *a, **k) -> None:  # pragma: no cover - simple
            pass

    class Graph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def __contains__(self, name: str) -> bool:
            return name in self.nodes

    class DummyRegistry:
        def __init__(self) -> None:
            self.graph = Graph()
            self.update_args: tuple | None = None
            self.hot_swapped = False

        def record_heartbeat(self, _name: str) -> None:  # pragma: no cover - simple
            pass

        def register_interaction(self, *_a, **_k) -> None:  # pragma: no cover - simple
            pass

        def record_interaction_metadata(self, *a, **k) -> None:  # pragma: no cover - simple
            pass

        def register_bot(self, name: str) -> None:
            self.graph.nodes.setdefault(name, {})

        def update_bot(self, name: str, module: str, *, patch_id=None, commit=None) -> None:
            self.update_args = (name, module, patch_id, commit)
            self.graph.nodes.setdefault(name, {})["version"] = 1

        def hot_swap_bot(self, name: str) -> None:
            self.hot_swapped = True

    engine = DummyEngine()
    pipeline = DummyPipeline()
    registry = DummyRegistry()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=types.SimpleNamespace(
            context_builder=None, apply_validated_patch=lambda *a, **k: (True, None)
        ),
    )
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
    monkeypatch.setattr(scm.subprocess, "check_output", lambda *a, **k: b"deadbeef")
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, *, backend="venv": types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )

    mgr.run_patch(file_path, "add")
    assert registry.update_args == (
        "bot",
        scm.path_for_prompt(file_path),
        None,
        "deadbeef",
    )
    assert registry.hot_swapped


def test_generate_and_patch_delegates(monkeypatch, tmp_path):
    calls: list[tuple] = []

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def fake_run_patch(path, desc, **kw):
        calls.append(("patch", path, desc, kw.get("context_builder")))
        return mapl.AutomationResult(None, prb.ROIResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(mgr, "run_patch", fake_run_patch)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())
    mgr.generate_and_patch(file_path, "fix", context_builder=builder)
    assert any(c[0] == "patch" and c[1] == file_path and c[3] is builder for c in calls)


def test_generate_and_patch_failure(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def bad_run_patch(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mgr, "run_patch", bad_run_patch)
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())

    class DummyBuilder:
        def refresh_db_weights(self) -> None:
            return None

    monkeypatch.setattr(scm, "ContextBuilder", DummyBuilder)

    with pytest.raises(RuntimeError):
        mgr.generate_and_patch(file_path, "fix")


def test_generate_and_patch_refreshes_builder(monkeypatch, tmp_path):
    calls: list[int] = []

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):  # pragma: no cover - stub
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def fake_run_patch(path, desc, **kw):
        return mapl.AutomationResult(None, prb.ROIResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(mgr, "run_patch", fake_run_patch)
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())

    class Builder:
        def refresh_db_weights(self) -> None:
            calls.append(1)

    builder = Builder()

    mgr.generate_and_patch(file_path, "fix", context_builder=builder)
    mgr.generate_and_patch(file_path, "fix", context_builder=builder)

    assert len(calls) == 2


def test_should_refactor_on_test_failures_only(monkeypatch):
    class DummyEngine:
        patch_suggestion_db = None

    class DummyPipeline:
        pass

    class DummyDataBot:
        def __init__(self) -> None:
            self.failures = 0

        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return self.failures

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    mgr._last_errors = data_bot.average_errors("bot")
    assert not mgr.should_refactor()
    data_bot.failures = 5
    assert mgr.should_refactor()


class PredictingDataBot:
    def __init__(self) -> None:
        self.thresholds = types.SimpleNamespace(
            roi_drop=0.0, error_threshold=0.0, test_failure_threshold=0.0
        )
        self.current_roi = 1.0
        self.current_err = 3.0
        self.current_fail = 2.0
        self.anomaly_sensitivity = 1.0
        self.confidence = 0.1

    def roi(self, _bot: str) -> float:
        return self.current_roi

    def average_errors(self, _bot: str) -> float:
        return self.current_err

    def average_test_failures(self, _bot: str) -> float:
        return self.current_fail

    def get_thresholds(self, _bot: str):
        return self.thresholds

    def reload_thresholds(self, _bot: str):  # pragma: no cover - simple stub
        return self.thresholds

    def update_thresholds(
        self,
        _bot: str,
        *,
        roi_drop: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        forecast: dict | None = None,
    ) -> None:
        self.thresholds = types.SimpleNamespace(
            roi_drop=roi_drop,
            error_threshold=error_threshold,
            test_failure_threshold=test_failure_threshold,
        )
        self.forecast = forecast

    def check_degradation(
        self, _bot: str, _roi: float, _errors: float, _failures: float
    ) -> bool:
        t = self.thresholds
        conf = self.confidence
        return (t.roi_drop or 0.0) < -conf or (t.error_threshold or 0.0) > conf or (
            t.test_failure_threshold or 0.0
        ) > conf


def test_should_refactor_with_negative_prediction(monkeypatch):
    class DummyEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(
                refresh_db_weights=lambda: None, session_id=""
            )
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    class DummyPipeline:
        pass

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    mgr.baseline_tracker.update(roi=3.0, errors=1.0, tests_failed=0.0)
    mgr.baseline_tracker.update(roi=2.0, errors=2.0, tests_failed=1.0)
    assert mgr.should_refactor()
    assert data_bot.thresholds.roi_drop < 0.0
    assert data_bot.thresholds.error_threshold > 0.0
    assert data_bot.thresholds.test_failure_threshold > 0.0


def test_should_refactor_ignores_positive_prediction(monkeypatch):
    class DummyEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(
                refresh_db_weights=lambda: None, session_id=""
            )
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    class DummyPipeline:
        pass

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    # Positive ROI and stable errors/failures should not trigger refactor
    mgr.baseline_tracker.update(roi=1.0, errors=1.0, tests_failed=0.0)
    mgr.baseline_tracker.update(roi=2.0, errors=0.5, tests_failed=0.0)
    data_bot.current_roi = 3.0
    data_bot.current_err = 0.4
    data_bot.current_fail = 0.0
    assert not mgr.should_refactor()


def test_ema_forecast_reduces_false_positive(monkeypatch):
    class DummyDataBot:
        def __init__(self) -> None:
            self.rois = [1.0, 1.0, 1.0, 0.95]
            self.idx = 0
            self.thresholds = types.SimpleNamespace(
                roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=1.0
            )

        def roi(self, _bot: str) -> float:
            v = self.rois[self.idx]
            self.idx += 1
            return v

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return self.thresholds

        def update_thresholds(self, _bot: str, *, roi_drop=None, error_threshold=None, test_failure_threshold=None, forecast=None):
            self.thresholds = types.SimpleNamespace(
                roi_drop=roi_drop,
                error_threshold=error_threshold,
                test_failure_threshold=test_failure_threshold,
            )
            self.forecast = forecast

        def check_degradation(self, _bot, roi, _err, _fail):
            delta = roi - 1.0
            return delta <= (self.thresholds.roi_drop or 0.0)

    class LocalEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    mgr = scm.SelfCodingManager(
        LocalEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )
    for _ in range(3):
        assert not mgr.should_refactor()
    assert not mgr.should_refactor()
    assert mgr.data_bot.forecast["roi"]

def test_init_requires_helpers():
    engine = DummyEngine()
    pipeline = DummyPipeline()
    with pytest.raises(ValueError):
        scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            data_bot=DummyDataBot(),
        )
    with pytest.raises(ValueError):
        scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            bot_registry=DummyRegistry(),
        )
