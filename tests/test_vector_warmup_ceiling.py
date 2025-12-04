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
    assert warmup_summary["handlers"].startswith("deferred")
    assert warmup_summary["vectorise"].startswith("deferred")
    assert {"handlers", "vectorise"}.issubset(deferred)


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
    assert warmup_summary["vectorise"].startswith("deferred")
    assert "vectorise" in deferred


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
    assert warmup_summary["model"].startswith("deferred")
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
    assert instantiations == 0

    caplog.clear()
    lazy_bootstrap.warmup_vector_service(**warmup_kwargs)
    retry_summary = _get_warmup_summary(caplog)
    assert retry_summary["handlers"].startswith("deferred")
    assert retry_summary["vectorise"].startswith("deferred")
    assert instantiations == 0


def test_legacy_warmup_backgrounds_heavy_without_budget(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_STAGE_META.clear()

    background_calls: list[tuple[set[str], dict[str, float | None]]] = []

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class StubSharedVectorService:
        def __init__(self, *_args, **_kwargs):
            pass

        def vectorise(self, *_args, **_kwargs):  # pragma: no cover - background deferral expected
            raise AssertionError("legacy vectorise should be deferred")

    vectorizer_stub.SharedVectorService = StubSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    def capture_hook(stages: set[str], budget_hints=None):  # type: ignore[override]
        hints: dict[str, float | None] = {}
        if isinstance(budget_hints, dict):
            hints.update({k: v for k, v in budget_hints.items() if isinstance(k, str)})
        background_calls.append((set(stages), hints))

    lazy_bootstrap.warmup_vector_service(
        download_model=True,
        hydrate_handlers=True,
        start_scheduler=True,
        run_vectorise=True,
        logger=logging.getLogger("test"),
        background_hook=capture_hook,
        warmup_lite=True,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["model"].startswith("deferred")
    assert warmup_summary["handlers"].startswith("deferred")
    assert warmup_summary["scheduler"].startswith("deferred")
    assert warmup_summary["vectorise"].startswith("deferred")
    assert warmup_summary.get("handlers_queued") == "queued"
    assert warmup_summary.get("vectorise_queued") == "queued"

    assert background_calls, "background hook should be invoked for deferrals"
    stages, hints = background_calls[0]
    assert {"handlers", "scheduler", "vectorise", "model"}.issubset(stages)
    assert all(
        hint is None or hint <= lazy_bootstrap._HEAVY_STAGE_CEILING for hint in hints.values()
    )


def test_heavy_warmup_without_budgets_defers_at_default_ceiling(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_STAGE_META.clear()

    monkeypatch.delenv("MENACE_BOOTSTRAP", raising=False)
    monkeypatch.delenv("MENACE_BOOTSTRAP_FAST", raising=False)
    monkeypatch.delenv("MENACE_BOOTSTRAP_MODE", raising=False)

    ceiling = 0.05
    monkeypatch.setattr(lazy_bootstrap, "_HEAVY_STAGE_CEILING", ceiling)
    monkeypatch.setattr(lazy_bootstrap, "_BOOTSTRAP_STAGE_TIMEOUT", ceiling)

    background_calls: list[tuple[set[str], dict[str, float | None]]] = []

    def capture_hook(stages, budget_hints=None):  # type: ignore[override]
        hints: dict[str, float | None] = {}
        if isinstance(budget_hints, dict):
            hints.update({k: v for k, v in budget_hints.items() if isinstance(k, str)})
        background_calls.append((set(stages), hints))

    vectorizer_stub = types.ModuleType("vector_service.vectorizer")

    class SlowSharedVectorService:
        def __init__(self, *_args, stop_event=None, **_kwargs):
            self._stop_event = stop_event
            if stop_event is not None:
                stop_event.wait(0.1)

        def vectorise(self, *_args, stop_event=None, **_kwargs):
            target = stop_event or self._stop_event
            if target is not None:
                target.wait(0.1)
            return []

    vectorizer_stub.SharedVectorService = SlowSharedVectorService
    monkeypatch.setitem(sys.modules, "vector_service.vectorizer", vectorizer_stub)

    lazy_bootstrap.warmup_vector_service(
        hydrate_handlers=True,
        run_vectorise=True,
        warmup_lite=False,
        logger=logging.getLogger("test"),
        background_hook=capture_hook,
    )

    warmup_summary = _get_warmup_summary(caplog)
    assert warmup_summary["handlers"].startswith("deferred")
    assert warmup_summary["vectorise"].startswith("deferred")
    assert warmup_summary.get("handlers_queued") == "queued"
    assert warmup_summary.get("vectorise_queued") == "queued"

    assert background_calls, "background hook should receive ceiling deferrals"
    stages, hints = background_calls[0]
    assert {"handlers", "vectorise"}.issubset(stages)
    assert hints.get("handlers") == ceiling
    assert hints.get("vectorise") == ceiling

