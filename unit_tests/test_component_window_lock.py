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


def test_component_window_lock_times_out_without_env(monkeypatch):
    monkeypatch.delenv(btp._COMPONENT_WINDOW_LOCK_MAX_WAIT_ENV, raising=False)
    monkeypatch.setattr(btp, "_DEFAULT_COMPONENT_LOCK_MAX_WAIT", 0.1)
    monkeypatch.setattr(btp, "_COMPONENT_WINDOW_LOCK_WARN_AFTER", 0.05)
    coordinator = btp.SharedTimeoutCoordinator(total_budget=5)

    release_event = threading.Event()

    def _holder():
        with coordinator._component_window_lock(label="holder", requested=1):
            release_event.wait(timeout=0.3)

    holder_thread = threading.Thread(target=_holder, name="holder-thread")
    holder_thread.start()
    time.sleep(0.01)

    with pytest.raises(TimeoutError):
        with coordinator._component_window_lock(label="waiter", requested=2):
            pass

    release_event.set()
    holder_thread.join(timeout=1)


def test_component_window_lock_timeout_reports_wait_and_holder(monkeypatch, caplog):
    monkeypatch.setenv(btp._COMPONENT_WINDOW_LOCK_MAX_WAIT_ENV, "0.05")
    monkeypatch.setattr(btp, "_COMPONENT_WINDOW_LOCK_WARN_AFTER", 0.01)
    coordinator = btp.SharedTimeoutCoordinator(total_budget=1)

    release_event = threading.Event()

    def _holder():
        with coordinator._component_window_lock(label="holder", requested=0.5):
            release_event.wait(timeout=0.2)

    holder_thread = threading.Thread(target=_holder, name="holder-thread")
    holder_thread.start()
    time.sleep(0.005)

    caplog.set_level(logging.WARNING)
    with pytest.raises(TimeoutError) as excinfo:
        with coordinator._component_window_lock(label="waiter", requested=0.5):
            pass

    release_event.set()
    holder_thread.join(timeout=1)

    message = str(excinfo.value)
    assert "after" in message and "holder-thread" in message

    timeout_logs = [
        r for r in caplog.records if getattr(r, "shared_timeout", {}).get("event") == "lock-timeout"
    ]
    assert timeout_logs, "expected timeout warning for diagnostic payload"
    payload = timeout_logs[-1].shared_timeout
    assert payload.get("waited_for_lock") is not None
    assert payload.get("lock_wait_budget") == coordinator._component_lock_max_wait
    holder_info = payload.get("lock_holder") or {}
    assert holder_info.get("thread_name") == "holder-thread"


def test_component_window_lock_warning_surfaces_wait_and_holder(monkeypatch, caplog):
    monkeypatch.setenv(btp._COMPONENT_WINDOW_LOCK_MAX_WAIT_ENV, "0.06")
    monkeypatch.setattr(btp, "_COMPONENT_WINDOW_LOCK_WARN_AFTER", 0.005)
    coordinator = btp.SharedTimeoutCoordinator(total_budget=1)

    release_event = threading.Event()

    def _holder():
        with coordinator._component_window_lock(label="holder", requested=0.5):
            release_event.wait(timeout=0.2)

    holder_thread = threading.Thread(target=_holder, name="holder-thread")
    holder_thread.start()
    time.sleep(0.003)

    caplog.set_level(logging.WARNING)
    with pytest.raises(TimeoutError):
        with coordinator._component_window_lock(label="waiter", requested=0.5):
            pass

    release_event.set()
    holder_thread.join(timeout=1)

    warning_messages = [r.message for r in caplog.records if "lock acquisition" in r.message]
    assert warning_messages, "expected warning that surfaces waited duration"
    assert any("after" in msg and "holder-thread" in msg for msg in warning_messages)


def test_component_window_lock_same_thread_reentry(monkeypatch, caplog):
    coordinator = btp.SharedTimeoutCoordinator(total_budget=5)

    caplog.set_level(logging.WARNING)
    with coordinator._component_window_lock(label="outer", requested=1):
        with coordinator._component_window_lock(label="inner", requested=1):
            pass

    reentry_logs = [r for r in caplog.records if "lock reentry" in r.message]
    assert reentry_logs, "expected reentry warning to aid debugging"

