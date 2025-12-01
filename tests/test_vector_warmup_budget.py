import logging
import sys
import time
import types

import vector_service.lazy_bootstrap as lazy_bootstrap


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
    assert warmup_summary["model"] == "skipped-budget"


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
    assert warmup_summary["handlers"] == "skipped-budget"
