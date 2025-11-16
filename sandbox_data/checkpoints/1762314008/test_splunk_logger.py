import logging
import types
import menace.splunk_logger as sl


def test_network_error_logged(monkeypatch, caplog):
    hec = sl.SplunkHEC(token="t")

    def post(url, headers=None, json=None, timeout=1):
        raise RuntimeError("boom")

    monkeypatch.setattr(sl, "requests", types.SimpleNamespace(post=post))
    caplog.set_level(logging.ERROR)
    hec.add("123", {"x": 1})
    assert "failed" in caplog.text


def test_retry_success(monkeypatch):
    hec = sl.SplunkHEC(token="t")

    attempts = []

    def post(url, headers=None, json=None, timeout=1):
        attempts.append(True)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(sl, "requests", types.SimpleNamespace(post=post))
    hec.add("123", {"x": 1})
    assert len(attempts) == 3


def test_retry_failure_logged(monkeypatch, caplog):
    hec = sl.SplunkHEC(token="t")

    attempts = []

    def post(url, headers=None, json=None, timeout=1):
        attempts.append(True)
        raise RuntimeError("boom")

    monkeypatch.setattr(sl, "requests", types.SimpleNamespace(post=post))
    caplog.set_level(logging.ERROR)
    hec.add("123", {"x": 1})
    assert len(attempts) == 3
    assert "failed" in caplog.text
