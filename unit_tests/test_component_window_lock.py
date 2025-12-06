import logging
import threading
import time

import pytest

import bootstrap_timeout_policy as btp


def test_component_window_lock_diagnostics(monkeypatch, caplog):
    monkeypatch.setattr(btp, "_COMPONENT_WINDOW_LOCK_WARN_AFTER", 0.05)
    monkeypatch.setenv(btp._COMPONENT_WINDOW_LOCK_MAX_WAIT_ENV, "0.15")
    coordinator = btp.SharedTimeoutCoordinator(total_budget=5)

    release_event = threading.Event()

    def _holder():
        with coordinator._component_window_lock(label="holder", requested=1):
            release_event.wait(timeout=0.3)

    holder_thread = threading.Thread(target=_holder, name="holder-thread")
    holder_thread.start()
    time.sleep(0.01)

    caplog.set_level(logging.WARNING)
    with pytest.raises(TimeoutError):
        with coordinator._component_window_lock(label="waiter", requested=2):
            pass

    release_event.set()
    holder_thread.join(timeout=1)

    lock_warnings = [r for r in caplog.records if "component window lock" in r.message]
    assert lock_warnings, "expected lock acquisition warning"

    shared_payload = getattr(lock_warnings[-1], "shared_timeout", {})
    assert shared_payload.get("waited_for_lock") is not None
    holder_info = shared_payload.get("lock_holder") or {}
    assert holder_info.get("thread_name") == "holder-thread"
    assert "caller" in holder_info

