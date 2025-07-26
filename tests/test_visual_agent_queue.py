import time
import threading
import types
import importlib
import subprocess

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
real_requests = pytest.importorskip("requests")

from tests.test_visual_agent_subprocess_recovery import _start_server


def test_queue_status_during_concurrent_runs(monkeypatch, tmp_path):
    proc, port = _start_server(tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        vac_mod = importlib.reload(importlib.import_module("menace.visual_agent_client"))

        class DummyLock:
            def acquire(self, timeout=0, poll_interval=0.05):
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

        # Disable inter-client lock so requests overlap
        monkeypatch.setattr(vac_mod, "_global_lock", DummyLock())

        codes: list[int] = []

        def record_post(*a, **k):
            resp = real_requests.post(*a, **k)
            codes.append(resp.status_code)
            return resp

        monkeypatch.setattr(
            vac_mod,
            "requests",
            types.SimpleNamespace(post=record_post, get=real_requests.get),
        )

        queue_sizes = []

        def patched_poll(self, base: str):
            while True:
                resp = real_requests.get(f"{base}/status", timeout=1)
                if resp.status_code == 200:
                    data = resp.json()
                    queue_sizes.append(data.get("queue"))
                    if not data.get("active"):
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
        time.sleep(0.01)
        t2.start()

        # Give the second client time to attempt while the first runs
        time.sleep(0.1)
        mid_status = real_requests.get(f"{url}/status", timeout=1).json()

        t1.join()
        t2.join()

        final_status = real_requests.get(f"{url}/status", timeout=1).json()

        assert all(code == 202 for code in codes)
        assert times["end2"] >= times["end1"]
        assert mid_status["queue"] == 1
        assert final_status["queue"] == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

