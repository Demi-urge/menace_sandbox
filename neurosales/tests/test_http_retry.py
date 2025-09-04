import os
import sys
import logging
import pytest
import requests
from unittest.mock import patch, MagicMock
import importlib.util
import types

from dynamic_path_router import resolve_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

spec = importlib.util.spec_from_file_location(
    "neurosales.http_retry", str(resolve_path("neurosales/http_retry.py"))
)
http_retry = importlib.util.module_from_spec(spec)
sys.modules.setdefault("neurosales", types.ModuleType("neurosales"))
sys.modules["neurosales.http_retry"] = http_retry
spec.loader.exec_module(http_retry)
sys.modules["neurosales"].http_retry = http_retry
ResilientRequester = http_retry.ResilientRequester


def test_get_retries_on_failure():
    req = ResilientRequester(max_retries=2, base_backoff=0)
    resp_fail = MagicMock()
    resp_fail.status_code = 500
    resp_ok = MagicMock()
    resp_ok.status_code = 200
    with patch("neurosales.http_retry.requests.Session.request", side_effect=[resp_fail, resp_ok]) as p, \
         patch("neurosales.http_retry.time.sleep") as ts:
        resp = req.get("http://x")
    assert resp.status_code == 200
    assert p.call_count == 2
    assert ts.called


def test_post_raises_after_retries():
    req = ResilientRequester(max_retries=1, base_backoff=0)
    resp_fail = MagicMock()
    resp_fail.status_code = 400
    with patch("neurosales.http_retry.requests.Session.request", return_value=resp_fail), \
         patch("neurosales.http_retry.time.sleep"):
        try:
            req.post("http://x")
        except Exception:
            caught = True
        else:
            caught = False
    assert caught


def test_request_logs_errors(caplog):
    req = ResilientRequester(max_retries=1, base_backoff=0)
    with patch(
        "neurosales.http_retry.requests.Session.request",
        side_effect=requests.RequestException("bad"),
    ), patch("neurosales.http_retry.time.sleep"), caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            req.get("http://x")
    assert any("HTTP GET request failed" in r.getMessage() for r in caplog.records)
