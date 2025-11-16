import builtins
import asyncio
import sandbox_runner.environment as env

class DummyIPRoute:
    calls = []
    def __init__(self):
        pass
    def link_lookup(self, ifname):
        self.calls.append(("link_lookup", ifname))
        return [1]
    def tc(self, action, parent, index, kind, **kw):
        self.calls.append((action, kind, kw))
    def close(self):
        self.calls.append(("close",))


def test_execute_local_tc_limits(monkeypatch):
    DummyIPRoute.calls = []
    monkeypatch.setattr(env, "IPRoute", DummyIPRoute)
    orig_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "docker":
            raise ImportError("no docker")
        return orig_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: True)
    monkeypatch.setattr(env.resource, "setrlimit", lambda *a, **k: None)
    asyncio.run(env._execute_in_container("print('x')", {"NETWORK_LATENCY_MS": "40", "PACKET_DUPLICATION": "2"}))
    assert ("add", "netem", {"delay": "40ms", "duplicate": 2.0}) in DummyIPRoute.calls
    assert any(c[0] == "del" for c in DummyIPRoute.calls)
    assert ("close",) in DummyIPRoute.calls
