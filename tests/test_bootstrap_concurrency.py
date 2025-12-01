import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lock_utils import Timeout
from tests.test_bootstrap_orchestrator import _reload_bootstrap


class _BlockingBootstrapper:
    def __init__(self, start_event: threading.Event, finish_event: threading.Event):
        self.start_event = start_event
        self.finish_event = finish_event
        self.calls: list[dict] = []

    def bootstrap(self, **kwargs):
        self.calls.append(kwargs)
        self.start_event.set()
        self.finish_event.wait(timeout=2)


def test_ensure_bootstrapped_allows_single_execution(monkeypatch, tmp_path):
    eb = _reload_bootstrap(monkeypatch, tmp_path)

    start = threading.Event()
    finish = threading.Event()
    bootstrapper = _BlockingBootstrapper(start, finish)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(eb.ensure_bootstrapped, bootstrapper=bootstrapper)
            for _ in range(5)
        ]

        start.wait(timeout=1)
        finish.set()
        results = [f.result() for f in futures]

    assert len(bootstrapper.calls) == 1
    assert results.count(results[0]) == len(results)


def test_ensure_bootstrapped_raises_on_lock_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCK_TIMEOUT", "0.2")
    eb = _reload_bootstrap(monkeypatch, tmp_path)

    lock = eb.SandboxLock(tmp_path / "lock")
    guard = lock.acquire(timeout=0.2)
    try:
        finish = threading.Event()
        bootstrapper = _BlockingBootstrapper(threading.Event(), finish)
        with pytest.raises(Timeout):
            eb.ensure_bootstrapped(bootstrapper=bootstrapper)
    finally:
        guard.__exit__(None, None, None)

    assert not bootstrapper.calls
