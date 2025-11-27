import logging
import threading
import time

import pytest

import coding_bot_interface as cbi


class _DummyPipeline:
    def __init__(self, **_kwargs: object) -> None:
        return


def test_prepare_timeout_logs_standard_hints(monkeypatch, caplog):
    stop_event = threading.Event()
    stop_event.set()

    monkeypatch.setenv("MENACE_BOOTSTRAP_WAIT_SECS", "30")
    monkeypatch.setattr(cbi, "enforce_bootstrap_timeout_policy", lambda logger=None: {})

    with caplog.at_level(logging.WARNING, logger=cbi.logger.name):
        with pytest.raises(TimeoutError) as excinfo:
            cbi._prepare_pipeline_for_bootstrap_impl(
                pipeline_cls=_DummyPipeline,
                context_builder=object(),
                bot_registry=object(),
                data_bot=object(),
                stop_event=stop_event,
                timeout=30.0,
                deadline=time.perf_counter() + 30.0,
            )

    combined_output = caplog.text + str(excinfo.value)

    assert "MENACE_BOOTSTRAP_WAIT_SECS=240" in combined_output
    assert "BOOTSTRAP_VECTOR_STEP_TIMEOUT=240" in combined_output
    assert "Stagger concurrent bootstraps" in combined_output
