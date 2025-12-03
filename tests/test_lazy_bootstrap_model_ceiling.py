import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path

import pytest


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


def _join_background_thread(timeout: float = 1.0) -> None:
    if lazy_bootstrap._MODEL_BACKGROUND_THREAD is not None:
        lazy_bootstrap._MODEL_BACKGROUND_THREAD.join(timeout=timeout)
        lazy_bootstrap._MODEL_BACKGROUND_THREAD = None


def _reset_model_state() -> None:
    lazy_bootstrap._MODEL_READY = False
    lazy_bootstrap._MODEL_BACKGROUND_THREAD = None
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    lazy_bootstrap._WARMUP_STAGE_META.clear()


def test_embedding_model_stage_deadline_hits_ceiling(monkeypatch, tmp_path):
    _reset_model_state()
    monkeypatch.setattr(
        lazy_bootstrap,
        "_model_bundle_path",
        lambda: tmp_path / "tiny-distilroberta-base.tar.xz",
    )

    def fail_bundle(*_, **__):  # pragma: no cover - defensive guard
        raise AssertionError("bundle should not run when ceiling is hit")

    monkeypatch.setitem(
        sys.modules,
        "vector_service.download_model",
        types.SimpleNamespace(bundle=fail_bundle),
    )

    stop_event = threading.Event()
    setattr(stop_event, "_stage_deadline", time.monotonic() + 0.01)

    result = lazy_bootstrap.ensure_embedding_model(
        logger=logging.getLogger("test"),
        warmup=True,
        warmup_lite=True,
        stop_event=stop_event,
        download_timeout=0.01,
    )

    _join_background_thread()

    assert result == (None, "deferred-ceiling")
    assert lazy_bootstrap._WARMUP_STAGE_MEMO.get("model") == "deferred-ceiling"
    assert (
        lazy_bootstrap._WARMUP_STAGE_META.get("model", {}).get("background_timeout")
        == pytest.approx(0.01, rel=0.05)
    )


def test_download_timeout_defers_to_background(monkeypatch, tmp_path):
    _reset_model_state()
    monkeypatch.setenv("EMBEDDER_BOOTSTRAP_WAIT", "0.02")
    monkeypatch.setattr(
        lazy_bootstrap,
        "_model_bundle_path",
        lambda: tmp_path / "tiny-distilroberta-base.tar.xz",
    )

    called = threading.Event()

    def fail_bundle(*_, **__):  # pragma: no cover - defensive guard
        called.set()
        raise AssertionError("bundle should not run when deferred")

    monkeypatch.setitem(
        sys.modules,
        "vector_service.download_model",
        types.SimpleNamespace(bundle=fail_bundle),
    )

    result = lazy_bootstrap.ensure_embedding_model(
        logger=logging.getLogger("test"),
        warmup=True,
        warmup_lite=True,
        stop_event=None,
        download_timeout=0.001,
    )

    _join_background_thread()

    assert result == (None, "deferred-ceiling")
    assert lazy_bootstrap._WARMUP_STAGE_MEMO.get("model") == "deferred-ceiling"
    assert (
        lazy_bootstrap._WARMUP_STAGE_META.get("model", {}).get("background_timeout")
        == pytest.approx(0.001, rel=0.1)
    )
    assert called.is_set() is False
