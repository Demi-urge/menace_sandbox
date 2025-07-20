import builtins
import types
import asyncio
import sandbox_runner.environment as env

class DummyPsutil(types.ModuleType):
    def __init__(self):
        super().__init__("psutil")
        self.calls = []
        self.RLIMIT_CPU = 0
        self.RLIMIT_AS = 9
    def Process(self, pid=None):
        return DummyProcess(pid or 0, self.calls)
    def net_io_counters(self):
        return types.SimpleNamespace(bytes_recv=0, bytes_sent=0)

class DummyProcess:
    def __init__(self, pid, calls):
        self.pid = pid
        self.calls = calls
        self.count = 0
    def rlimit(self, res, limits):
        self.calls.append((res, limits))
    def cpu_times(self):
        self.count += 1
        if self.count > 1:
            return types.SimpleNamespace(user=20.0, system=0.0)
        return types.SimpleNamespace(user=0.0, system=0.0)
    def memory_info(self):
        return types.SimpleNamespace(rss=0)
    def io_counters(self):
        return types.SimpleNamespace(read_bytes=0, write_bytes=0)

class DummyPopen:
    def __init__(self, *a, **k):
        if k.get("preexec_fn"):
            k["preexec_fn"]()
        self.pid = 1
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""
        self.killed = False
    def communicate(self, timeout=None):
        return ("", "")
    def poll(self):
        return None if not self.killed else self.returncode
    def kill(self):
        self.killed = True
        self.returncode = -9


def test_execute_local_psutil_limits(monkeypatch):
    dummy_ps = DummyPsutil()
    monkeypatch.setattr(env, "psutil", dummy_ps)
    monkeypatch.setattr(env, "resource", None)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)
    monkeypatch.setattr(env.subprocess, "Popen", DummyPopen)
    orig_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "docker":
            raise ImportError("no docker")
        return orig_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    asyncio.run(env._execute_in_container("print('hi')", {"CPU_LIMIT": "1", "MEMORY_LIMIT": "1Mi"}))
    assert (dummy_ps.RLIMIT_CPU, (10, 10)) in dummy_ps.calls
    assert (dummy_ps.RLIMIT_AS, (1048576, 1048576)) in dummy_ps.calls


def test_run_psutil_enforces_limits(monkeypatch):
    dummy_ps = DummyPsutil()
    monkeypatch.setattr(env, "psutil", dummy_ps)
    monkeypatch.setattr(env, "resource", None)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)

    class DonePopen(DummyPopen):
        def poll(self):
            return 0

    monkeypatch.setattr(env.subprocess, "Popen", DonePopen)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)
    res, _ = asyncio.run(env._section_worker("print('x')", {"CPU_LIMIT": "1", "MEMORY_LIMIT": "1Mi"}, 0.0))
    assert res["exit_code"] == 0
    assert (dummy_ps.RLIMIT_CPU, (10, 10)) in dummy_ps.calls
    assert (dummy_ps.RLIMIT_AS, (1048576, 1048576)) in dummy_ps.calls
