from __future__ import annotations

import importlib.util
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
    return next(record.warmup for record in caplog.records if hasattr(record, "warmup"))


def test_handler_chain_deferred_when_ceiling_below_estimate(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    deferred: set[str] = set()

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class StubSharedVectorService:
        def __init__(self, *_args, **_kwargs):
            pass

        def vectorise(self, *_args, **_kwargs):  # pragma: no cover - deferral expected
            raise AssertionError("vectorise should be deferred")

    vectorizer_stub.SharedVectorService = StubSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        run_vectorise=True,
        warmup_lite=False,
        stage_timeouts={"budget": 5.0},
        background_hook=deferred.update,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"] == "deferred-ceiling"
    assert warmup_summary["vectorise"] == "deferred-ceiling"
    assert deferred == {"handlers", "vectorise"}


def test_vectorise_alone_deferred_when_ceiling_below_estimate(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    deferred: set[str] = set()

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class StubSharedVectorService:
        def __init__(self, *_args, **_kwargs):
            pass

        def vectorise(self, *_args, **_kwargs):  # pragma: no cover - deferral expected
            raise AssertionError("vectorise should be deferred")

    vectorizer_stub.SharedVectorService = StubSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=False,
        run_vectorise=True,
        warmup_lite=False,
        stage_timeouts={"budget": 2.0},
        background_hook=deferred.update,
        logger=logging.getLogger("test"),
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["vectorise"] == "deferred-ceiling"
    assert deferred == {"vectorise"}


def test_model_download_honours_stage_timeout(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = object  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    called: list[str] = []

    def fake_download(**_kwargs):  # noqa: ARG001
        called.append("model")
        raise AssertionError("model download should be deferred")

    monkeypatch.setattr(lazy_bootstrap, "ensure_embedding_model", fake_download)

    background: list[set[str]] = []
    budget_hints_list: list[dict[str, float | None]] = []

    def capture_hook(stages, budget_hints=None):  # type: ignore[override]
        background.append(set(stages))
        sanitized: dict[str, float | None] = {}
        if isinstance(budget_hints, dict):
            sanitized.update({k: v for k, v in budget_hints.items() if isinstance(k, str)})
        budget_hints_list.append(sanitized)

    lazy_bootstrap.warmup_vector_service(
        logger=logging.getLogger("test"),
        download_model=True,
        warmup_lite=False,
        stage_timeouts={"model": 0.5},
        background_hook=capture_hook,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"] == "deferred-ceiling"
    assert not called
    assert background and {"model"}.issubset(background[0])
    assert budget_hints_list and budget_hints_list[0].get("model") == 0.5


def test_handler_timebox_queues_background_and_memoises(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()

    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")

    hook_calls: list[tuple[set[str], dict[str, float | None]]] = []
    instantiations = 0

    class SlowSharedVectorService:
        def __init__(self, *_args, **_kwargs):
            nonlocal instantiations
            instantiations += 1

        def vectorise(self, *_args, stop_event=None, **_kwargs):
            if stop_event is not None:
                stop_event.wait(0.2)
            return []

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")
    vectorizer_stub.SharedVectorService = SlowSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    def capture_hook(stages: set[str], budget_hints=None):  # type: ignore[override]
        hints: dict[str, float | None] = {}
        if isinstance(budget_hints, dict):
            hints.update({k: v for k, v in budget_hints.items() if isinstance(k, str)})
        hook_calls.append((set(stages), hints))

    warmup_kwargs = dict(
        hydrate_handlers=True,
        run_vectorise=True,
        warmup_lite=False,
        stage_timeouts={"handlers": 0.02, "vectorise": 0.02},
        background_hook=capture_hook,
        logger=logging.getLogger("test"),
    )

    lazy_bootstrap.warmup_vector_service(**warmup_kwargs)

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"].startswith("deferred")
    assert warmup_summary["vectorise"].startswith("deferred")
    assert warmup_summary.get("handlers_queued") == "queued"
    assert warmup_summary.get("vectorise_queued") == "queued"
    assert hook_calls and {"handlers", "vectorise"}.issubset(hook_calls[0][0])
    assert hook_calls[0][1].get("handlers") == 0.02
    assert hook_calls[0][1].get("vectorise") == 0.02
    assert instantiations == 1

    caplog.clear()
    lazy_bootstrap.warmup_vector_service(**warmup_kwargs)
    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"].startswith("deferred")
    assert retry_summary["vectorise"].startswith("deferred")
    assert instantiations == 1

