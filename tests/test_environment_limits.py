import sys
import importlib.util
import types
from pathlib import Path


def _load_env():
    if 'filelock' not in sys.modules:
        class DummyLock:
            def __init__(self):
                self.is_locked = False
                self.lock_file = ''
            def acquire(self):
                self.is_locked = True
            def release(self):
                self.is_locked = False
            def __enter__(self):
                self.acquire()
                return self
            def __exit__(self, exc_type, exc, tb):
                self.release()
        sys.modules['filelock'] = types.SimpleNamespace(FileLock=lambda *a, **k: DummyLock())
    pkg_path = Path(__file__).resolve().parents[1] / 'sandbox_runner'
    if 'sandbox_runner' not in sys.modules:
        pkg = types.ModuleType('sandbox_runner')
        pkg.__path__ = [str(pkg_path)]
        sys.modules['sandbox_runner'] = pkg
    path = pkg_path / 'environment.py'  # path-ignore
    spec = importlib.util.spec_from_file_location('sandbox_runner.environment', path)
    env = importlib.util.module_from_spec(spec)
    sys.modules['sandbox_runner.environment'] = env
    assert spec and spec.loader
    spec.loader.exec_module(env)  # type: ignore[attr-defined]
    return env


def test_cgroup_create_cleanup(tmp_path):
    env = _load_env()
    base = tmp_path / 'cgroup'
    base.mkdir()
    env._CGROUP_BASE = base
    cg = env._create_cgroup('0.1', '4Mi')
    assert cg and cg.exists()
    assert (cg / 'cpu.max').read_text() == '10000 100000'
    assert (cg / 'memory.max').read_text() == str(4 * 1024 * 1024)
    env._cleanup_cgroup(cg)
    assert not cg.exists()


def test_apply_psutil_rlimits(monkeypatch):
    env = _load_env()

    class DummyProc:
        def __init__(self):
            self.calls = []
        def rlimit(self, res, limits):
            self.calls.append((res, limits))

    class DummyProcess(DummyProc):
        pass

    ps_stub = types.ModuleType('psutil')
    ps_stub.RLIMIT_CPU = 1
    ps_stub.RLIMIT_AS = 2
    ps_stub.Process = DummyProcess
    monkeypatch.setattr(env, 'psutil', ps_stub, raising=False)

    proc = DummyProc()
    env._apply_psutil_rlimits(proc, '2', '3Mi')
    assert (1, (20, 20)) in proc.calls
    assert (2, (3 * 1024 * 1024, 3 * 1024 * 1024)) in proc.calls

    proc2 = DummyProc()
    env._apply_psutil_rlimits(proc2, 'bad', 'bad')
    assert proc2.calls == []


def test_setup_and_cleanup_tc(monkeypatch):
    env = _load_env()

    class DummyIPRoute:
        calls = []
        def __init__(self):
            pass
        def link_lookup(self, ifname):
            self.calls.append(('link_lookup', ifname))
            return [1]
        def tc(self, action, parent, index, kind, **kw):
            self.calls.append((action, kind, kw))
        def close(self):
            self.calls.append(('close',))

    monkeypatch.setattr(env, 'IPRoute', DummyIPRoute)

    DummyIPRoute.calls = []
    ipr, idx = env._setup_tc_netem({'NETWORK_LATENCY_MS': '40', 'PACKET_DUPLICATION': '2'})
    assert isinstance(ipr, DummyIPRoute)
    assert ('add', 'netem', {'delay': '40ms', 'duplicate': 2.0}) in DummyIPRoute.calls

    DummyIPRoute.calls.clear()
    env._cleanup_tc(ipr, idx)
    assert any(c[0] == 'del' for c in DummyIPRoute.calls)
    assert ('close',) in DummyIPRoute.calls
