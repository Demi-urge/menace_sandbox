import importlib
import os
import threading
import time
import types
import runpy
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


def test_calls_are_queued(monkeypatch):
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

    assert len(results) == 2
    assert not errors


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
            shared.acquire()

        def release(self):
            if shared.locked():
                shared.release()

        @property
        def is_locked(self):
            return shared.locked()

    monkeypatch.setattr(filelock, "FileLock", DummyLock)
    monkeypatch.setattr(filelock, "Timeout", DummyTimeout)

    vac_mod = _reload_client(monkeypatch)
    _setup_stubbed_client(monkeypatch, vac_mod, delay=0.5)
    client2 = _setup_stubbed_client(monkeypatch, vac_mod)

    vac_mod._global_lock.acquire(timeout=0)

    results: list[dict] = []

    def call():
        results.append(client2.ask([{"content": "hi"}]))

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.1)
    vac_mod._global_lock.release()
    t.join()

    assert results[0]["choices"][0]["message"]["content"] == "ok"


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
    with pytest.raises(RuntimeError):
        client.ask([{"content": "p"}])
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
    with pytest.raises(RuntimeError):
        client.revert()
    assert not lock_path.exists()


def test_calls_process_sequentially(monkeypatch):
    vac_mod = _reload_client(monkeypatch)

    active = 0
    overlap = []

    def fake_send(self, base, prompt):
        nonlocal active, overlap
        active += 1
        if active > 1:
            overlap.append(True)
        time.sleep(0.1)
        active -= 1
        return True, "ok"

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_send", fake_send)
    client = vac_mod.VisualAgentClient(urls=["http://x"])

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

    assert len(results) == 2
    assert not errors
    assert not overlap


def test_refresh_serializes_with_run(monkeypatch):
    """A refresh should block other /run calls via the global lock."""

    class DummyTimeout(Exception):
        pass

    shared = threading.Lock()

    class DummyLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, timeout=0):
            shared.acquire()

        def release(self):
            if shared.locked():
                shared.release()

        @property
        def is_locked(self):
            return shared.locked()

    monkeypatch.setattr(filelock, "FileLock", DummyLock)
    monkeypatch.setattr(filelock, "Timeout", DummyTimeout)

    vac_mod = _reload_client(monkeypatch)

    events: list[tuple[str, float]] = []

    def fake_refresh(self):
        events.append(("refresh_start", time.time()))
        time.sleep(0.1)
        events.append(("refresh_end", time.time()))
        return True

    def fake_post(url, headers=None, json=None, timeout=10):
        if url.endswith("/run"):
            events.append(("run", time.time()))
        return types.SimpleNamespace(status_code=202)

    def fake_get(url, timeout=10):
        return types.SimpleNamespace(status_code=200, json=lambda: {"active": False, "status": "ok"})

    monkeypatch.setattr(vac_mod.VisualAgentClient, "_refresh_token", fake_refresh)
    monkeypatch.setattr(vac_mod, "requests", types.SimpleNamespace(post=fake_post, get=fake_get))
    monkeypatch.setattr(vac_mod, "log_event", lambda *a, **k: "id")
    monkeypatch.setattr(vac_mod.VisualAgentClient, "_poll", lambda self, base: (True, "ok"))

    client1 = vac_mod.VisualAgentClient(urls=["http://x"], token_refresh_cmd="cmd")
    client2 = vac_mod.VisualAgentClient(urls=["http://x"])

    fut = client1._refresh_token_async()
    time.sleep(0.02)
    client2.ask_async([{"content": "hi"}]).result()
    fut.result()

    refresh_end = next(t for n, t in events if n == "refresh_end")
    run_time = next(t for n, t in events if n == "run")

    assert run_time >= refresh_end



def test_overlapping_clients_sequential(monkeypatch, tmp_path):
    """Concurrent clients contacting a real server should run sequentially."""

    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json

    lock_path = tmp_path / "va.lock"

    events: list[float] = []
    running_lock = threading.Lock()
    active = {"val": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/run":
                if not running_lock.acquire(blocking=False):
                    self.send_response(409)
                    self.end_headers()
                    return
                events.append(time.time())
                active["val"] = True

                def worker():
                    time.sleep(0.2)
                    active["val"] = False
                    running_lock.release()
                    pass

                threading.Thread(target=worker, daemon=True).start()
                self.send_response(202)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/status":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"active": active["val"]}).encode())
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("localhost", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        monkeypatch.setenv("VISUAL_AGENT_LOCK_FILE", str(lock_path))
        vac_mod = _reload_client(monkeypatch)
        monkeypatch.setattr(vac_mod, "log_event", lambda *a, **k: "id")

        c1 = vac_mod.VisualAgentClient(
            urls=[f"http://localhost:{port}"], poll_interval=0.01
        )
        c2 = vac_mod.VisualAgentClient(
            urls=[f"http://localhost:{port}"], poll_interval=0.01
        )

        times: list[tuple[str, float]] = []

        def run1():
            c1.ask([{"content": "a"}])
            times.append(("end1", time.time()))

        def run2():
            c2.ask([{"content": "b"}])
            times.append(("end2", time.time()))

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join()
        t2.join()
    finally:
        server.shutdown()

    assert len(events) == 2
    assert events[1] - events[0] >= 0.19
    end1 = next(t for n, t in times if n == "end1")
    end2 = next(t for n, t in times if n == "end2")
    assert end2 >= end1


def test_main_runs_single_worker(monkeypatch):
    """Running menace_visual_agent_2 directly should use one worker."""

    import sys
    import types
    import uvicorn

    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    calls = {}

    def fake_run(*args, **kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    runpy.run_module("menace_visual_agent_2", run_name="__main__")

    assert calls.get("workers") == 1


def test_dataset_dir_env(monkeypatch, tmp_path):
    """The dataset directory should respect the VA_DATASET_DIR env var."""

    import sys
    import types

    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    data_dir = tmp_path / "dataset"
    monkeypatch.setenv("VA_DATASET_DIR", str(data_dir))

    va_mod = importlib.reload(importlib.import_module("menace_visual_agent_2"))

    assert va_mod.DATASET_DIR == str(data_dir)
    assert data_dir.exists()

