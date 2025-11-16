import asyncio
import threading
from types import SimpleNamespace
import sys
from pathlib import Path
import importlib.util
import types

from dynamic_path_router import resolve_path

sys.path.append(str(Path(__file__).resolve().parents[2]))

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules["menace"] = menace_pkg
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.__path__ = [str(resolve_path("self_improvement"))]
sys.modules["menace.self_improvement"] = si_pkg
# Stub out dependencies required by meta_planning during import
logging_utils = types.ModuleType("menace.logging_utils")
logging_utils.get_logger = lambda name=None: types.SimpleNamespace(
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    exception=lambda *a, **k: None,
    info=lambda *a, **k: None,
)
logging_utils.log_record = lambda **k: k
sys.modules["menace.logging_utils"] = logging_utils

sandbox_settings_mod = types.ModuleType("menace.sandbox_settings")


class _SandboxSettings:
    sandbox_data_dir = "/tmp"
    sandbox_repo_path = "/tmp"
    synergy_weight_file = "/tmp/synergy.json"
    sandbox_central_logging = False
    meta_mutation_rate = 1.0
    meta_roi_weight = 1.0
    meta_domain_penalty = 1.0
    meta_entropy_threshold = None
    meta_entropy_weight = 0.0
    meta_search_depth = 1
    meta_beam_width = 1
    max_allowed_errors = 0
    overfitting_entropy_threshold = 1.0


sandbox_settings_mod.SandboxSettings = _SandboxSettings
sandbox_settings_mod.DEFAULT_SEVERITY_SCORE_MAP = {
    "critical": 100.0,
    "crit": 100.0,
    "fatal": 100.0,
    "high": 75.0,
    "error": 75.0,
    "warn": 50.0,
    "warning": 50.0,
    "medium": 50.0,
    "low": 25.0,
    "info": 0.0,
}

sys.modules["menace.sandbox_settings"] = sandbox_settings_mod

ws_db_mod = types.ModuleType("menace.workflow_stability_db")


class _WSDB:
    def record_metrics(self, *a, **k):
        pass


ws_db_mod.WorkflowStabilityDB = _WSDB
sys.modules["menace.workflow_stability_db"] = ws_db_mod

roi_db_mod = types.ModuleType("menace.roi_results_db")


class _ROIDB:
    def log_result(self, *a, **k):
        pass


roi_db_mod.ROIResultsDB = _ROIDB
sys.modules["menace.roi_results_db"] = roi_db_mod

lock_utils_mod = types.ModuleType("menace.lock_utils")
lock_utils_mod.SandboxLock = lambda *a, **k: types.SimpleNamespace(
    __enter__=lambda self: self,
    __exit__=lambda self, exc_type, exc, tb: False,
)
lock_utils_mod.Timeout = Exception
lock_utils_mod.LOCK_TIMEOUT = 1
sys.modules["menace.lock_utils"] = lock_utils_mod

error_logger_mod = types.ModuleType("menace.error_logger")
error_logger_mod.TelemetryEvent = object
sys.modules["menace.error_logger"] = error_logger_mod

init_mod = types.ModuleType("menace.self_improvement.init")
init_mod.settings = _SandboxSettings()
sys.modules["menace.self_improvement.init"] = init_mod
spec = importlib.util.spec_from_file_location(
    "menace.self_improvement.meta_planning",
    resolve_path("self_improvement/meta_planning.py"),  # path-ignore
)
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)
sys.modules["menace.self_improvement.meta_planning"] = mp


def _setup(monkeypatch):
    log_messages = []

    class Logger:
        def debug(self, msg, *, extra=None, exc_info=None):
            log_messages.append((msg, extra))

        def exception(self, msg, *, extra=None, exc_info=None):
            log_messages.append((msg, extra))

        warning = debug

    logger = Logger()
    monkeypatch.setattr(mp, "get_logger", lambda name=None: logger)
    monkeypatch.setattr(mp, "log_record", lambda **k: k)

    cfg = SimpleNamespace(
        enable_meta_planner=False,
        meta_mutation_rate=1.0,
        meta_roi_weight=1.0,
        meta_domain_penalty=1.0,
        max_allowed_errors=0,
        overfitting_entropy_threshold=1.0,
    )
    monkeypatch.setattr(mp, "_init", SimpleNamespace(settings=cfg))
    monkeypatch.setattr(mp, "_get_entropy_threshold", lambda c, t: 0)
    monkeypatch.setattr(
        mp, "get_stable_workflows", lambda: SimpleNamespace(record_metrics=lambda *a, **k: None)
    )
    return log_messages


def test_cycle_logs_error_on_exception(monkeypatch):
    log_messages = _setup(monkeypatch)

    class ErrorPlanner:
        cluster_map = {}
        roi_db = None
        stability_db = None

        def discover_and_persist(self, workflows):
            raise RuntimeError("boom")
    monkeypatch.setattr(mp, "MetaWorkflowPlanner", ErrorPlanner)
    stop_event = threading.Event()

    def evaluate_cycle(tracker, error_log):
        stop_event.set()
        return "run", {}

    asyncio.run(
        mp.self_improvement_cycle(
            {"w": lambda: None},
            interval=0,
            stop_event=stop_event,
            evaluate_cycle=evaluate_cycle,
        )
    )
    assert any(
        msg == "cycle" and extra.get("outcome") == "error"
        for msg, extra in log_messages
    )


def test_cycle_logs_skipped_on_empty_records(monkeypatch):
    log_messages = _setup(monkeypatch)

    class EmptyPlanner:
        cluster_map = {}
        roi_db = None
        stability_db = None

        def discover_and_persist(self, workflows):
            return []
    monkeypatch.setattr(mp, "MetaWorkflowPlanner", EmptyPlanner)
    stop_event = threading.Event()

    def evaluate_cycle(tracker, error_log):
        stop_event.set()
        return "run", {}

    asyncio.run(
        mp.self_improvement_cycle(
            {"w": lambda: None},
            interval=0,
            stop_event=stop_event,
            evaluate_cycle=evaluate_cycle,
        )
    )
    assert any(
        msg == "cycle"
        and extra.get("outcome") == "skipped"
        and extra.get("reason") == "no_records"
        for msg, extra in log_messages
    )
