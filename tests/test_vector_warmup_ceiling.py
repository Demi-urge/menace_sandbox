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


def test_handler_chain_deferred_when_ceiling_below_estimate(caplog):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    deferred: set[str] = set()

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


def test_vectorise_alone_deferred_when_ceiling_below_estimate(caplog):
    caplog.set_level(logging.INFO)
    lazy_bootstrap._WARMUP_STAGE_MEMO.clear()
    deferred: set[str] = set()

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

