import importlib.util
import logging
import threading
import sys
import time
import types
from pathlib import Path

vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "vector_service")]
sys.modules.setdefault("vector_service", vector_service_pkg)

_lazy_bootstrap_spec = importlib.util.spec_from_file_location(
    "vector_service.lazy_bootstrap",
    Path(__file__).resolve().parents[1] / "vector_service" / "lazy_bootstrap.py",
)
assert _lazy_bootstrap_spec is not None and _lazy_bootstrap_spec.loader is not None
lazy_bootstrap = importlib.util.module_from_spec(_lazy_bootstrap_spec)
sys.modules["vector_service.lazy_bootstrap"] = lazy_bootstrap
_lazy_bootstrap_spec.loader.exec_module(lazy_bootstrap)


def _get_warmup_summary(caplog):
    return next(record.warmup for record in caplog.records if hasattr(record, "warmup"))


def test_model_download_times_out(caplog, monkeypatch, tmp_path):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._MODEL_READY = False

    def slow_download(*, logger=None, warmup=False):  # noqa: ARG001
        time.sleep(0.2)
        return tmp_path / "model.tar"

    monkeypatch.setattr(lazy_bootstrap, "ensure_embedding_model", slow_download)

    budget_calls = 0

    def check_budget():
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls > 2:
            raise TimeoutError("budget expired")

    lazy_bootstrap.warmup_vector_service(
        download_model=True, check_budget=check_budget, logger=logging.getLogger("test"), probe_model=False
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-budget"


def test_handler_hydration_times_out(caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    class SlowSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            time.sleep(0.2)

        def vectorise(self, *_args, **_kwargs):
            return {"vector": [1.0]}

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = SlowSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    budget_calls = 0

    def check_budget():
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls > 3:
            raise TimeoutError("budget expired")

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        check_budget=check_budget,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-budget"


def test_stage_skipped_when_budget_below_estimate(caplog):
    caplog.set_level(logging.INFO)

    lazy_bootstrap.warmup_vector_service(
        download_model=True,
        budget_remaining=lambda: 5.0,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-estimate"


def test_background_hook_invoked_for_estimate_deferral(caplog):
    caplog.set_level(logging.INFO)
    deferred: set[str] = set()

    lazy_bootstrap.warmup_vector_service(
        download_model=True,
        budget_remaining=lambda: 5.0,
        logger=logging.getLogger("test"),
        background_hook=deferred.update,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-estimate"
    assert deferred == {"model"}


def test_handler_vectorise_budget_deferral_uses_background_hook(caplog):
    caplog.set_level(logging.INFO)
    deferred: set[str] = set()

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        run_vectorise=True,
        warmup_lite=False,
        budget_remaining=lambda: 1.0,
        logger=logging.getLogger("test"),
        background_hook=deferred.update,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"].startswith("deferred")
    assert warmup_summary["vectorise"].startswith("deferred")
    assert deferred == {"handlers", "vectorise"}


def test_default_stage_timeouts_enforced_without_budget(caplog, monkeypatch, tmp_path):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._MODEL_READY = False

    monkeypatch.setattr(
        lazy_bootstrap,
        "_CONSERVATIVE_STAGE_TIMEOUTS",
        {"model": 0.05, "handlers": 0.05, "vectorise": 0.05},
    )

    def slow_download(*, logger=None, warmup=False, stop_event=None):  # noqa: ARG001
        time.sleep(0.1)
        return tmp_path / "model.tar"

    monkeypatch.setattr(lazy_bootstrap, "ensure_embedding_model", slow_download)

    lazy_bootstrap.warmup_vector_service(
        download_model=True,
        check_budget=None,
        logger=logging.getLogger("test"),
        probe_model=False,
        stage_timeouts=None,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-timeout"


def test_timeout_defers_remaining_stages(caplog, monkeypatch, tmp_path):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._MODEL_READY = False
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()

    monkeypatch.setattr(
        lazy_bootstrap,
        "_CONSERVATIVE_STAGE_TIMEOUTS",
        {"model": 0.05, "handlers": 0.05, "vectorise": 0.05},
    )

    def slow_download(*, logger=None, warmup=False, stop_event=None):  # noqa: ARG001
        time.sleep(0.1)
        if stop_event is not None:
            stop_event.set()
        return tmp_path / "model.tar"

    monkeypatch.setattr(lazy_bootstrap, "ensure_embedding_model", slow_download)

    deferred: set[str] = set()

    lazy_bootstrap.warmup_vector_service(
        download_model=True,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
        check_budget=None,
        logger=logging.getLogger("test"),
        probe_model=False,
        stage_timeouts=None,
        background_hook=deferred.update,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-timeout"
    assert warmup_summary["handlers"] == "deferred-budget"
    assert warmup_summary["scheduler"] == "deferred-budget"
    assert warmup_summary.get("vectorise") == "deferred-budget"
    assert deferred.issuperset({"model", "handlers", "scheduler", "vectorise"})


def test_handler_timeout_stops_background(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    monkeypatch.setattr(
        lazy_bootstrap,
        "_CONSERVATIVE_STAGE_TIMEOUTS",
        {"model": 0.1, "handlers": 0.05, "vectorise": 0.05},
    )

    stop_seen = threading.Event()
    finished = threading.Event()

    class SlowSharedVectorService:
        def __init__(self, *, stop_event=None, budget_check=None, **_kwargs):  # noqa: ARG002
            stop_seen.set()
            while stop_event is not None and not stop_event.is_set():
                if budget_check is not None:
                    try:
                        budget_check(stop_event)
                    except TimeoutError:
                        if stop_event is not None:
                            stop_event.set()
                        break
                time.sleep(0.01)
            finished.set()

        def vectorise(self, *_args, **_kwargs):
            return []

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = SlowSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        stage_timeouts=None,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-timeout"
    assert stop_seen.is_set()
    assert finished.wait(0.5)


def test_vectorise_timeout_aborts_work(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    monkeypatch.setattr(
        lazy_bootstrap,
        "_CONSERVATIVE_STAGE_TIMEOUTS",
        {"model": 0.1, "handlers": 0.2, "vectorise": 0.05},
    )

    started = threading.Event()
    stopped = threading.Event()

    class CooperativeVectorService:
        def __init__(self, *_args, **_kwargs):
            pass

        def vectorise(self, *_args, stop_event=None, **_kwargs):
            started.set()
            while stop_event is not None and not stop_event.is_set():
                time.sleep(0.01)
            stopped.set()
            return []

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = CooperativeVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        run_vectorise=True,
        stage_timeouts=None,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary.get("vectorise") == "deferred-timeout"
    assert started.is_set()
    assert stopped.wait(0.5)


def test_upfront_budget_deferral_memoised(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()

    instantiations = 0

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class CountingSharedVectorService:
        def __init__(self, *_args, **_kwargs):
            nonlocal instantiations
            instantiations += 1

        def vectorise(self, *_args, **_kwargs):
            raise AssertionError("vectorise should be deferred before starting")

    vectorizer_stub.SharedVectorService = CountingSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    warmup_kwargs = dict(
        hydrate_handlers=True,
        run_vectorise=True,
        warmup_lite=False,
        stage_timeouts={"handlers": 0.1, "vectorise": 0.1},
        logger=logging.getLogger("test"),
    )

    lazy_bootstrap.warmup_vector_service(**warmup_kwargs)

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-budget"
    assert warmup_summary["vectorise"] == "deferred-budget"
    assert instantiations == 0
    assert "handlers" in warmup_summary.get("background", "")

    caplog.clear()

    lazy_bootstrap.warmup_vector_service(**warmup_kwargs)

    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"] == "deferred-budget"
    assert retry_summary["vectorise"] == "deferred-budget"
    assert instantiations == 0
