import threading
import time
import types
import sys
import pytest

if 'filelock' not in sys.modules:
    filelock_mod = types.ModuleType('filelock')
    class DummyFileLock:
        def __init__(self, *a, **k):
            pass
        def acquire(self, *a, **k):
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
    filelock_mod.FileLock = DummyFileLock
    filelock_mod.Timeout = RuntimeError
    sys.modules['filelock'] = filelock_mod

from menace.visual_agent_client import VisualAgentClientStub


def test_stub_single_connection():
    stub = VisualAgentClientStub()
    results = {}

    def first():
        stub.active = True
        time.sleep(0.1)
        results["first"] = True
        stub.active = False

    def second():
        with pytest.raises(RuntimeError) as exc:
            stub.ask([{"content": "b"}])
        results["second"] = "409" in str(exc.value)

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["first"] is True
    assert results["second"] is True
