import logging
import sys
import time
import types

import importlib.util
from pathlib import Path


def _load_registry():
    path = Path(__file__).resolve().parent.parent / "vector_service" / "registry.py"
    spec = importlib.util.spec_from_file_location("vector_service.registry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("vector_service.registry", module)
    spec.loader.exec_module(module)
    return module


def test_load_handlers_times_out(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
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

    handlers = registry.load_handlers(handler_timeouts=0.05, bootstrap_fast=False)

    assert "slow" in handlers
    assert getattr(handlers["slow"], "is_patch_stub", False)
    assert any(getattr(record, "reason", None) == "timeout" for record in caplog.records)


def test_budget_cap_defers_without_import(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
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

    handlers = registry.load_handlers(handler_timeouts={"budget": 0.0}, bootstrap_fast=False)

    assert "budgeted" in handlers
    assert getattr(handlers["budgeted"], "is_patch_stub", False)
    assert not called
    assert any(getattr(record, "reason", None) == "budget" for record in caplog.records)
