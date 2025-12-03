import importlib.util
import json
import logging
import sys
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
    summaries = [record.warmup for record in caplog.records if hasattr(record, "warmup")]
    assert summaries, "no warmup summary emitted"
    return summaries[-1]


def _reset_state():
    lazy_bootstrap._clear_warmup_cache()
    lazy_bootstrap._SCHEDULER = None
    lazy_bootstrap._MODEL_READY = False


def _clear_cache_only():
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_CACHE_LOADED = False
    lazy_bootstrap._SCHEDULER = None
    lazy_bootstrap._MODEL_READY = False


def test_handler_deferrals_exposed_in_summary(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()

    deferral_payload = {"foo": {"reason": "budget", "remaining_budget": 0.5}}

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class StubSharedVectorService:
        handler_deferrals = deferral_payload
        handler_hydration_deferred = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

    vectorizer_stub.SharedVectorService = StubSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=False,
        run_vectorise=False,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert json.loads(warmup_summary["handler_deferrals"]) == deferral_payload


def test_warmup_lite_defers_heavy_stages(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()

    scheduler_calls = []

    def fake_scheduler(**_kwargs):
        scheduler_calls.append("scheduler")

    monkeypatch.setattr(lazy_bootstrap, "ensure_scheduler_started", fake_scheduler)

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("SharedVectorService should be deferred during warmup-lite")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(logger=logging.getLogger("test"))

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-lite-noop"
    assert warmup_summary["scheduler"] == "deferred-lite"
    assert warmup_summary["vectorise"] == "deferred-lite-noop"
    assert {"handlers", "scheduler", "vectorise"}.issubset(set(warmup_summary["deferred"].split(",")))
    assert not scheduler_calls


def test_bootstrap_deferral_memoised(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    scheduler_calls = []

    def fake_scheduler(**_kwargs):
        scheduler_calls.append("scheduler")

    monkeypatch.setattr(lazy_bootstrap, "ensure_scheduler_started", fake_scheduler)

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("SharedVectorService should be deferred during bootstrap")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-bootstrap"
    assert warmup_summary["scheduler"] == "deferred-bootstrap"
    assert warmup_summary["vectorise"] == "deferred-bootstrap"
    assert {"handlers", "scheduler", "vectorise"}.issubset(set(warmup_summary["deferred"].split(",")))
    assert not scheduler_calls

    caplog.clear()

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
    )

    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"] == "deferred-bootstrap"
    assert retry_summary["scheduler"] == "deferred-bootstrap"
    assert retry_summary["vectorise"] == "deferred-bootstrap"
    assert {"handlers", "scheduler", "vectorise"}.issubset(set(retry_summary["deferred"].split(",")))
    assert not scheduler_calls


def test_warmup_cache_reused(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("VECTOR_WARMUP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    _reset_state()

    scheduler_calls: list[str] = []

    def fake_scheduler(**_kwargs):
        scheduler_calls.append("scheduler")

    monkeypatch.setattr(lazy_bootstrap, "ensure_scheduler_started", fake_scheduler)

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("SharedVectorService should be deferred during bootstrap")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
    )

    cache_file = next(cache_dir.glob("*.json"))
    assert cache_file.exists()

    _clear_cache_only()

    deferred: set[str] = set()
    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
        background_hook=deferred.update,
    )

    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"] == "deferred-bootstrap"
    assert retry_summary["scheduler"] == "deferred-bootstrap"
    assert retry_summary["vectorise"] == "deferred-bootstrap"
    assert {"handlers", "scheduler", "vectorise"}.issubset(set(retry_summary["deferred"].split(",")))
    assert not scheduler_calls
    assert deferred.issuperset({"handlers", "scheduler", "vectorise"})


def test_ceiling_deferral_persisted(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("VECTOR_WARMUP_CACHE_DIR", str(tmp_path))
    _reset_state()

    calls: list[str] = []

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            calls.append("handlers")
            raise AssertionError("SharedVectorService should be deferred under ceiling")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        run_vectorise=True,
        stage_timeouts={"handlers": 0.5},
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-ceiling"
    assert warmup_summary["vectorise"] == "deferred-ceiling"
    assert not calls

    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_CACHE_LOADED = False
    caplog.clear()

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        run_vectorise=True,
        stage_timeouts={"handlers": 0.5},
    )

    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"] == "deferred-ceiling"
    assert retry_summary["vectorise"] == "deferred-ceiling"
    assert not calls
