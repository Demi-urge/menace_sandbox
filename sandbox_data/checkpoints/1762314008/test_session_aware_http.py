import types
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import menace.allocator_service as svc
import menace.session_vault as sv
import menace.session_aware_http as sah
from menace.session_aware_http import SessionAwareHTTP


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


def _server(port: int):
    server = HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_wrapper(tmp_path):
    svc._vault = sv.SessionVault(path=tmp_path / "s.db")
    svc._vault.add(
        "localhost",
        sv.SessionData(cookies={}, user_agent="ua", fingerprint="fp", last_seen=0),
    )

    server = _server(8765)
    try:
        http = SessionAwareHTTP(base_url="http://testserver")
        # monkeypatch testclient to our FastAPI app
        from fastapi.testclient import TestClient
        client = TestClient(svc.app)

        orig_post = http._request_new_session

        def _request_new(domain: str):
            resp = client.post("/session/get", json={"domain": domain})
            data = resp.json()
            http.current = sv.SessionData(
                cookies=data["cookies"],
                user_agent=data["user_agent"],
                fingerprint=data["fingerprint"],
                last_seen=0,
                session_id=data["session_id"],
                domain=domain,
            )
        http._request_new_session = _request_new

        resp = http.get("http://localhost:8765")
        assert resp.status_code == 200
    finally:
        server.shutdown()


def test_report_failure_logged(monkeypatch, caplog):
    http = SessionAwareHTTP(base_url="http://x", report_attempts=2, report_backoff=0)
    http.current = sv.SessionData(cookies={}, user_agent="ua", fingerprint="fp", last_seen=0, session_id=1, domain="x")

    attempts = []

    def post(url, json=None, timeout=5):
        attempts.append(True)
        raise RuntimeError("boom")

    monkeypatch.setattr(sah, "requests", types.SimpleNamespace(post=post))
    caplog.set_level(logging.WARNING)
    http._report("success")
    assert len(attempts) == 2
    assert "session report failed" in caplog.text


def test_report_retry_success(monkeypatch):
    http = SessionAwareHTTP(base_url="http://x", report_attempts=3, report_backoff=0)
    http.current = sv.SessionData(cookies={}, user_agent="ua", fingerprint="fp", last_seen=0, session_id=1, domain="x")

    attempts = []

    def post(url, json=None, timeout=5):
        attempts.append(True)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(sah, "requests", types.SimpleNamespace(post=post))
    http._report("success")
    assert len(attempts) == 3
