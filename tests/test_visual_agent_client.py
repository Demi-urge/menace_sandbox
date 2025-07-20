import importlib
import os
import threading
import time
import types
import filelock
import pytest


pkg_path = os.path.join(os.path.dirname(__file__), "..")
pkg_spec = importlib.util.spec_from_file_location(
    "menace", os.path.join(pkg_path, "__init__.py"), submodule_search_locations=[pkg_path]
)
menace_pkg = importlib.util.module_from_spec(pkg_spec)
import sys
sys.modules["menace"] = menace_pkg
pkg_spec.loader.exec_module(menace_pkg)


def _reload_client(monkeypatch, prefix=None):
    if prefix is None:
        monkeypatch.delenv("VA_MESSAGE_PREFIX", raising=False)
    else:
        monkeypatch.setenv("VA_MESSAGE_PREFIX", prefix)
    mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))
    return mod


def _capture_prompt(monkeypatch, vac_mod):
    captured = {}

    def fake_send(self, base, prompt):
        captured["prompt"] = prompt
        return True, "ok"

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_send", fake_send)
    client = vac_mod.VisualAgentClient(urls=["http://x"])
    client.ask([{"content": "hello"}])
    return captured["prompt"], vac_mod.SELF_IMPROVEMENT_PREFIX


def test_default_prefix(monkeypatch):
    vac_mod = _reload_client(monkeypatch)
    prompt, prefix = _capture_prompt(monkeypatch, vac_mod)
    assert prompt.startswith(prefix)
    assert prefix == vac_mod.DEFAULT_MESSAGE_PREFIX


def test_env_prefix(monkeypatch):
    vac_mod = _reload_client(monkeypatch, "CUSTOM")
    prompt, _ = _capture_prompt(monkeypatch, vac_mod)
    assert prompt.startswith("CUSTOM")


def _setup_stubbed_client(monkeypatch, vac_mod, delay=0.0):
    """Return a VisualAgentClient instance with networking stubs."""

    class Resp:
        def __init__(self, status_code=202, data=None, text=""):
            self.status_code = status_code
            self._data = data or {}
            self.text = text

        def json(self):
            return self._data

    def fake_post(url, headers=None, json=None, timeout=10):
        return Resp(202)

    def fake_get(url, timeout=10):
        return Resp(200, {"active": False, "status": "ok"})

    monkeypatch.setattr(
        vac_mod,
        "requests",
        types.SimpleNamespace(post=fake_post, get=fake_get),
    )
    monkeypatch.setattr(vac_mod, "log_event", lambda *a, **k: "id")

    def fake_poll(self, base):
        if delay:
            time.sleep(delay)
        return True, "ok"

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_poll", fake_poll)
    return vac_mod.VisualAgentClient(urls=["http://x"])


def test_ask_overlap_error(monkeypatch):
    vac_mod = _reload_client(monkeypatch)
    client = _setup_stubbed_client(monkeypatch, vac_mod, delay=0.1)

    results: list[dict] = []
    errors: list[Exception] = []

    def call():
        try:
            results.append(client.ask([{"content": "hi"}]))
        except Exception as exc:  # pragma: no cover - thread errors
            errors.append(exc)

    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "busy" in str(errors[0])


def test_sequential_asks_succeed(monkeypatch):
    vac_mod = _reload_client(monkeypatch)
    client = _setup_stubbed_client(monkeypatch, vac_mod)

    resp1 = client.ask([{"content": "a"}])
    resp2 = client.ask([{"content": "b"}])

    assert resp1["choices"][0]["message"]["content"] == "ok"
    assert resp2["choices"][0]["message"]["content"] == "ok"


def test_token_refresh_retry(monkeypatch, caplog):
    vac_mod = _reload_client(monkeypatch)
    calls: list[int] = []

    def fake_run(cmd, shell=True, text=True, capture_output=True):
        calls.append(1)
        if len(calls) < 2:
            return types.SimpleNamespace(returncode=1, stdout="bad", stderr="err")
        return types.SimpleNamespace(returncode=0, stdout="NEW", stderr="")

    monkeypatch.setattr(vac_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(vac_mod.time, "sleep", lambda *a: None)
    caplog.set_level("WARNING")
    client = vac_mod.VisualAgentClient(urls=["http://x"], token_refresh_cmd="cmd")
    assert client._refresh_token()
    assert client.token == "NEW"
    assert len(calls) == 2
    assert "bad" in caplog.text or "err" in caplog.text


def test_token_refresh_failure(monkeypatch, caplog):
    vac_mod = _reload_client(monkeypatch)

    def fake_run(cmd, shell=True, text=True, capture_output=True):
        return types.SimpleNamespace(returncode=1, stdout="oops", stderr="fail")

    monkeypatch.setattr(vac_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(vac_mod.time, "sleep", lambda *a: None)
    caplog.set_level("WARNING")
    client = vac_mod.VisualAgentClient(urls=["http://x"], token_refresh_cmd="cmd")
    assert not client._refresh_token()
    assert "oops" in caplog.text or "fail" in caplog.text


def test_global_lock_across_clients(monkeypatch):
    """Concurrent clients should respect the global lock."""

    class DummyTimeout(Exception):
        pass

    shared = threading.Lock()

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, timeout=0):
            if not shared.acquire(blocking=False):
                raise DummyTimeout()

        def release(self):
            if shared.locked():
                shared.release()

        @property
        def is_locked(self):
            return shared.locked()

    monkeypatch.setattr(filelock, "FileLock", DummyLock)
    monkeypatch.setattr(filelock, "Timeout", DummyTimeout)

    vac_mod = _reload_client(monkeypatch)
    client1 = _setup_stubbed_client(monkeypatch, vac_mod, delay=0.5)
    client2 = _setup_stubbed_client(monkeypatch, vac_mod)

    # Manually hold the global lock to simulate an active run
    vac_mod._global_lock.acquire(timeout=0)
    try:
        with pytest.raises(RuntimeError):
            client2.ask([{"content": "hi"}])
    finally:
        vac_mod._global_lock.release()


def test_send_exception_releases_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "lock"
    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(lock_path))
    vac_mod = _reload_client(monkeypatch)

    def bad_post(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        vac_mod,
        "requests",
        types.SimpleNamespace(post=bad_post),
    )
    client = vac_mod.VisualAgentClient(urls=["http://x"])
    ok, _ = client._send("http://x", "p")
    assert not ok
    assert not lock_path.exists()


def test_revert_exception_releases_lock(monkeypatch, tmp_path):
    lock_path = tmp_path / "lock2"
    monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(lock_path))
    vac_mod = _reload_client(monkeypatch)

    def bad_post(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        vac_mod,
        "requests",
        types.SimpleNamespace(post=bad_post),
    )
    client = vac_mod.VisualAgentClient(urls=["http://x"])
    ok, _ = client._send_revert("http://x")
    assert not ok
    assert not lock_path.exists()

