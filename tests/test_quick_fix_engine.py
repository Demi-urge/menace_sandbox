import sys
import types
import logging
import pytest
from pathlib import Path

# Minimal stubs to allow importing the manager without heavy dependencies
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)

db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
db_stub.DataBot = object
sys.modules.setdefault("data_bot", db_stub)
sys.modules.setdefault("menace.data_bot", db_stub)

db_router_stub = types.ModuleType("db_router")
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.LOCAL_TABLES = set()


class DummyRouter:
    def __init__(self, *a, **k) -> None:
        pass

    class _Conn:
        def execute(self, *a, **k):
            return types.SimpleNamespace(fetchall=lambda: [])

        def commit(self) -> None:
            pass

    def get_connection(self, *_a, **_k):
        return self._Conn()


def init_db_router(*a, **k):
    return DummyRouter()


db_router_stub.DBRouter = DummyRouter
db_router_stub.init_db_router = init_db_router
sys.modules.setdefault("db_router", db_router_stub)

dpr = types.SimpleNamespace(
    resolve_path=lambda p: Path(p),
    repo_root=lambda: Path("."),
    path_for_prompt=lambda p: str(p),
    get_project_root=lambda: Path("."),
)
sys.modules["dynamic_path_router"] = dpr

sr_pkg = types.ModuleType("menace.sandbox_runner")
th_stub = types.ModuleType("menace.sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th_stub.TestHarnessResult = types.SimpleNamespace
sr_pkg.test_harness = th_stub
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)

mapl_stub = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    pass
class ModelAutomationPipeline:
    pass
mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object

class _DummyCtx:
    def set(self, _val):
        return None

    def reset(self, _tok) -> None:
        pass

    def get(self):
        return None

sce_stub.MANAGER_CONTEXT = _DummyCtx()
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

code_db_stub = types.ModuleType("menace.code_database")
class PatchRecord:
    pass
code_db_stub.PatchRecord = PatchRecord
code_db_stub.PatchHistoryDB = object
sys.modules["menace.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub

import menace.self_coding_manager as scm


class DummyBuilder:
    def refresh_db_weights(self) -> None:
        pass


class DummyCognitionLayer:
    def __init__(self) -> None:
        self.context_builder = DummyBuilder()


class DummyEngine:
    def __init__(self) -> None:
        self.cognition_layer = DummyCognitionLayer()


class DummyDataBot:
    def get_thresholds(self, _bot: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            roi_drop=0.1, error_threshold=0.1, test_failure_threshold=0.1
        )

    def reload_thresholds(self, _bot: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            roi_drop=0.1, error_threshold=0.1, test_failure_threshold=0.1
        )


class DummyPipeline:
    pass


class DummyRegistry:
    def register_bot(self, *_a, **_k) -> None:
        pass


class DummyErrorDB:
    pass


def make_manager(skip: bool = False) -> scm.SelfCodingManager:
    return scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        skip_quick_fix_validation=skip,
    )


@pytest.fixture(autouse=True)
def _patch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda _b: None)
    monkeypatch.setattr(scm, "ErrorDB", DummyErrorDB)


def test_missing_quick_fix_engine_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(scm, "QuickFixEngine", None)
    mgr = make_manager()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            mgr._ensure_quick_fix_engine()
    assert "pip install menace[quickfix]" in caplog.text


def test_skip_quick_fix_engine(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(scm, "QuickFixEngine", None)
    mgr = make_manager(skip=True)
    with caplog.at_level(logging.WARNING):
        assert mgr._ensure_quick_fix_engine() is None
    text = caplog.text
    assert "pip install menace[quickfix]" in text
    assert "skipping validation" in text


def test_patch_uses_refreshed_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating DB weights should result in a fresh builder per patch."""

    weights = {"value": 0}

    class Builder:
        def __init__(self) -> None:
            self.weight = weights["value"]

        def refresh_db_weights(self) -> None:  # pragma: no cover - no-op
            pass

    monkeypatch.setattr(scm, "create_context_builder", lambda: Builder())
    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda _b: None)

    seen = []

    def fake_helper(_mgr, _desc, **kw):
        cb = kw.get("context_builder")
        seen.append(getattr(cb, "weight", -1))
        return ""

    monkeypatch.setattr(scm, "_BASE_MANAGER_GENERATE_HELPER", fake_helper)
    monkeypatch.setattr(scm, "QuickFixEngine", lambda *a, **k: types.SimpleNamespace())

    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
    )
    import menace.quick_fix_engine as qfe

    weights["value"] = 1
    qfe.manager_generate_helper(mgr, "first")
    weights["value"] = 2
    qfe.manager_generate_helper(mgr, "second")

    assert seen == [1, 2]

