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
    lazy_bootstrap._LAZY_BOOTSTRAP_SATISFIED = False
    lazy_bootstrap._LAZY_BOOTSTRAP_SUMMARY = None


def _clear_cache_only():
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_CACHE_LOADED = False
    lazy_bootstrap._SCHEDULER = None
    lazy_bootstrap._MODEL_READY = False
    lazy_bootstrap._LAZY_BOOTSTRAP_SATISFIED = False
    lazy_bootstrap._LAZY_BOOTSTRAP_SUMMARY = None


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
    assert warmup_summary["handlers"] == "deferred-bootstrap-hard-gate"
    assert warmup_summary["scheduler"] == "deferred-bootstrap-hard-gate"
    assert warmup_summary["vectorise"] == "deferred-bootstrap-hard-gate"
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
    assert retry_summary["handlers"] == "deferred-bootstrap-hard-gate"
    assert retry_summary["scheduler"] == "deferred-bootstrap-hard-gate"
    assert retry_summary["vectorise"] == "deferred-bootstrap-hard-gate"
    assert {"handlers", "scheduler", "vectorise"}.issubset(set(retry_summary["deferred"].split(",")))
    assert not scheduler_calls


def test_bootstrap_fast_defers_handlers_and_reports_background(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()
    monkeypatch.setenv("MENACE_BOOTSTRAP_FAST", "1")

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("SharedVectorService should be deferred during bootstrap-fast")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    deferred_calls: list[set[str]] = []

    def _background_hook(stages, _budget=None):
        deferred_calls.append(set(stages))

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        warmup_lite=False,
        hydrate_handlers=True,
        start_scheduler=False,
        run_vectorise=True,
        background_hook=_background_hook,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"].startswith("deferred")
    assert "handlers" in warmup_summary["deferred_stages"].split(",")
    assert "vectorise" in warmup_summary["deferred_stages"].split(",")
    assert deferred_calls and {"handlers", "vectorise"}.issubset(deferred_calls[-1])


def test_bootstrap_presence_only_defers_heavy_work(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    download_calls: list[str] = []
    scheduler_calls: list[str] = []
    deferred_calls: list[set[str]] = []
    recorded_hints: list[dict[str, float | None]] = []

    def fake_model_download(**_kwargs):
        download_calls.append("model")
        raise AssertionError("Model download should be deferred during bootstrap")

    monkeypatch.setattr(lazy_bootstrap, "ensure_embedding_model", fake_model_download)
    monkeypatch.setattr(
        lazy_bootstrap,
        "ensure_scheduler_started",
        lambda **_kwargs: scheduler_calls.append("scheduler"),
    )

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class NoOpSharedVectorService:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("SharedVectorService should be deferred during bootstrap")

    vectorizer_stub.SharedVectorService = NoOpSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    def _background_hook(stages, budget_hints=None):
        deferred_calls.append(set(stages))
        hints = budget_hints or {}
        recorded_hints.append({stage: hints.get(stage) for stage in stages})

    stage_timeouts = {"model": 3.0, "handlers": 2.0, "scheduler": 1.0, "vectorise": 1.5}

    warmup_summary = lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        download_model=True,
        probe_model=True,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
        warmup_lite=False,
        background_hook=_background_hook,
        stage_timeouts=stage_timeouts,
    )

    assert warmup_summary["model"] == "deferred-bootstrap-hard-gate"
    assert warmup_summary["handlers"] == "deferred-bootstrap-hard-gate"
    assert warmup_summary["scheduler"] == "deferred-bootstrap-hard-gate"
    assert warmup_summary["vectorise"] == "deferred-bootstrap-hard-gate"
    assert set(warmup_summary.get("deferred", "").split(",")) == {
        "model",
        "handlers",
        "scheduler",
        "vectorise",
    }
    assert not download_calls
    assert not scheduler_calls
    assert deferred_calls and deferred_calls[-1] == {"model", "handlers", "scheduler", "vectorise"}
    assert recorded_hints[-1] == {stage: stage_timeouts[stage] for stage in deferred_calls[-1]}
    assert lazy_bootstrap._WARMUP_STAGE_MEMO["model"] == "deferred-bootstrap-hard-gate"

    caplog.clear()
    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        download_model=True,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
        warmup_lite=False,
        background_hook=_background_hook,
        stage_timeouts=stage_timeouts,
    )

    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["model"] == "deferred-bootstrap-hard-gate"
    assert not download_calls
    assert not scheduler_calls


def test_bootstrap_presence_only_queues_background(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _reset_state()
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    def _stub_shared_vector_service(*_args, **_kwargs):  # pragma: no cover - guard
        raise AssertionError("Handler hydration should be deferred during bootstrap")

    vectorizer_stub.SharedVectorService = _stub_shared_vector_service
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    scheduled: list[tuple[set[str], Mapping[str, float | None] | None]] = []

    def _record_schedule(stages, **kwargs):
        scheduled.append((set(stages), kwargs.get("timeouts")))

    monkeypatch.setattr(lazy_bootstrap, "_schedule_background_warmup", _record_schedule)

    summary = lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        hydrate_handlers=True,
        run_vectorise=True,
    )

    assert summary["handlers"].startswith("deferred-bootstrap")
    assert summary["vectorise"].startswith("deferred-bootstrap")
    assert scheduled and {"handlers", "vectorise"}.issubset(scheduled[-1][0])
    timeout_hints = scheduled[-1][1] or {}
    assert "handlers" in timeout_hints and "vectorise" in timeout_hints


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
    assert retry_summary["handlers"] == "deferred-bootstrap-hard-gate"
    assert retry_summary["scheduler"] == "deferred-bootstrap-hard-gate"
    assert retry_summary["vectorise"] == "deferred-bootstrap-hard-gate"
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
