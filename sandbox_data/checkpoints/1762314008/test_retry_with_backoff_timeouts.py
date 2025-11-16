import logging
import asyncio
import pytest
import requests

from resilience import retry_with_backoff, RetryError


def test_retry_with_backoff_handles_requests_timeout(caplog):
    calls = {"count": 0}

    def func():
        calls["count"] += 1
        raise requests.Timeout("boom")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RetryError):
            retry_with_backoff(func, attempts=2, delay=0)
    assert calls["count"] == 2
    assert any("timeout on attempt 1/2" in r.message for r in caplog.records)


def test_retry_with_backoff_handles_asyncio_timeout(caplog):
    calls = {"count": 0}

    def func():
        calls["count"] += 1
        raise asyncio.TimeoutError()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RetryError):
            retry_with_backoff(func, attempts=2, delay=0)
    assert calls["count"] == 2
    assert any("timeout on attempt 1/2" in r.message for r in caplog.records)
