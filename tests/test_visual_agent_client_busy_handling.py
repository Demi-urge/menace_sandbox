import time
import threading
import types
import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
real_requests = pytest.importorskip("requests")

from tests.test_visual_agent_subprocess_recovery import _start_server


def test_busy_client_waits(monkeypatch, tmp_path):
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))

        class DummyLock:
            def acquire(self, timeout=None):
                class G:
                    def __enter__(self_inner):
                        return self_inner
                    def __exit__(self_inner, exc_type, exc, tb):
                        pass
                return G()

            def release(self):
                pass

            @property
            def is_locked(self):
                return False

        # Disable inter-client locking so requests overlap
        monkeypatch.setattr(vac_mod, "_global_lock", DummyLock())

        codes = []
        def record_post(*a, **k):
            resp = real_requests.post(*a, **k)
            codes.append(resp.status_code)
            return resp

        monkeypatch.setattr(
            vac_mod,
            "requests",
            types.SimpleNamespace(post=record_post, get=real_requests.get),
        )

        def patched_poll(self, base: str):
            observed = False
            while True:
                resp = real_requests.get(f"{base}/status", timeout=1)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("active"):
                        observed = True
                    elif observed:
                        return True, "done"
                time.sleep(0.05)

        monkeypatch.setattr(vac_mod.VisualAgentClient, "_poll", patched_poll)

        client1 = vac_mod.VisualAgentClient(urls=[url], poll_interval=0.05, token="tombalolosvisualagent123")
        client2 = vac_mod.VisualAgentClient(urls=[url], poll_interval=0.05, token="tombalolosvisualagent123")

        times = {}

        def run1():
            times["start1"] = time.time()
            client1.ask([{"content": "a"}])
            times["end1"] = time.time()

        def run2():
            times["start2"] = time.time()
            client2.ask([{"content": "b"}])
            times["end2"] = time.time()

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert any(code == 409 for code in codes)
        assert times["end2"] >= times["end1"]
        status = real_requests.get(f"{url}/status", timeout=1).json()
        assert status["queue"] == 0
        assert status["active"] is False
    finally:
        proc.terminate()
        proc.wait(timeout=5)
