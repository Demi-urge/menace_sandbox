import logging
import sys
import time
import types
import threading
from contextlib import contextmanager

import importlib.util
from pathlib import Path


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_logs(logger: logging.Logger, level: int = logging.INFO):
    handler = _ListHandler()
    original_level = logger.level
    original_propagate = logger.propagate
    original_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    logger.disabled = False
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.disabled = original_disabled


def _load_registry():
    path = Path(__file__).resolve().parent.parent / "vector_service" / "registry.py"
    spec = importlib.util.spec_from_file_location("vector_service.registry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("vector_service.registry", module)
    spec.loader.exec_module(module)
    return module


def test_load_handlers_times_out(monkeypatch):
    registry = _load_registry()

    class SlowVectorizer:
        def __init__(self, *_, **__):
            time.sleep(0.2)

        def transform(self, _record):  # pragma: no cover - behaviour only
            return [1.0]

    module = types.ModuleType("vector_service.slow_vectorizer")
    module.SlowVectorizer = SlowVectorizer
    monkeypatch.setitem(sys.modules, "vector_service.slow_vectorizer", module)
    monkeypatch.setattr(
        registry,
        "_VECTOR_REGISTRY",
        {"slow": ("vector_service.slow_vectorizer", "SlowVectorizer", None, None)},
    )

    with _capture_logs(registry.logger) as records:
        handlers = registry.load_handlers(handler_timeouts=0.05, bootstrap_fast=False)

    assert "slow" in handlers
    assert getattr(handlers["slow"], "is_patch_stub", False)
    assert handlers.deferral_statuses.get("slow") == "timeout"
    assert any(getattr(record, "reason", None) == "timeout" for record in records)


def test_budget_cap_defers_without_import(monkeypatch):
    registry = _load_registry()

    called = False

    def _fail_import(name):
        nonlocal called
        called = True
        raise AssertionError(f"import_module called for {name}")

    monkeypatch.setattr(registry, "importlib", types.SimpleNamespace(import_module=_fail_import))
    monkeypatch.setattr(
        registry,
        "_VECTOR_REGISTRY",
        {"budgeted": ("vector_service.missing", "Missing", None, None)},
    )

    with _capture_logs(registry.logger) as records:
        handlers = registry.load_handlers(handler_timeouts={"budget": 0.0}, bootstrap_fast=False)

    assert "budgeted" in handlers
    assert getattr(handlers["budgeted"], "is_patch_stub", False)
    assert handlers.deferral_statuses.get("budgeted") == "budget"
    assert handlers.deferral_budgets.get("budgeted") is not None
    assert not called
    assert any(getattr(record, "reason", None) == "budget" for record in records)


def test_global_budget_short_circuits(monkeypatch):
    registry = _load_registry()

    def _import(name):
        module = types.ModuleType(name)

        class SlowVectorizer:
            def __init__(self, *_, **__):
                time.sleep(0.02)

            def transform(self, _record):  # pragma: no cover - behaviour only
                return [2.0]

        module.SlowVectorizer = SlowVectorizer
        return module

    monkeypatch.setattr(registry, "importlib", types.SimpleNamespace(import_module=_import))
    monkeypatch.setattr(
        registry,
        "_VECTOR_REGISTRY",
        {
            "first": ("vector_service.first", "SlowVectorizer", None, None),
            "second": ("vector_service.second", "SlowVectorizer", None, None),
        },
    )

    stop_event = threading.Event()
    with _capture_logs(registry.logger) as records:
        handlers = registry.load_handlers(
            handler_timeouts={"budget": 0.01}, stop_event=stop_event, bootstrap_fast=False
        )

    assert stop_event.is_set()
    assert set(handlers.keys()) == {"first", "second"}
    assert handlers.deferral_statuses.get("first") in {"timeout", "budget", "loaded"}
    assert handlers.deferral_statuses.get("second") == "budget"
    assert handlers.deferral_budgets.get("second") is not None
    assert any(getattr(record, "reason", None) == "budget" for record in records)


def test_stop_event_cancels_without_error_log(monkeypatch):
    registry = _load_registry()

    called = False

    def _fail_import(name):
        nonlocal called
        called = True
        raise AssertionError(f"import_module called for {name}")

    monkeypatch.setattr(registry, "importlib", types.SimpleNamespace(import_module=_fail_import))
    monkeypatch.setattr(
        registry,
        "_VECTOR_REGISTRY",
        {
            "fast": ("vector_service.fast_vectorizer", "FastVectorizer", None, None),
            "faster": ("vector_service.faster_vectorizer", "FastVectorizer", None, None),
        },
    )

    stop_event = threading.Event()
    stop_event.set()

    with _capture_logs(registry.logger) as records:
        handlers = registry.load_handlers(stop_event=stop_event, bootstrap_fast=False)

    assert set(handlers.keys()) == {"fast", "faster"}
    assert all(getattr(handler, "is_patch_stub", False) for handler in handlers.values())
    assert handlers.deferral_statuses.get("fast") == "cancelled"
    assert handlers.deferral_statuses.get("faster") == "cancelled"
    assert any(getattr(record, "reason", None) == "cancelled" for record in records)
    assert not any("vector_registry.handler.error" in record.getMessage() for record in records)
    assert not called
