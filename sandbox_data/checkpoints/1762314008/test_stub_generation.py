import sys
import types

th_stub = types.ModuleType("sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: None
th_stub.TestHarnessResult = object
sys.modules.setdefault("sandbox_runner.test_harness", th_stub)

menace_env = types.ModuleType("environment_generator")
menace_env._CPU_LIMITS = []
menace_env._MEMORY_LIMITS = []
menace_pkg = types.ModuleType("menace")
menace_pkg.environment_generator = menace_env
sys.modules.setdefault("menace.environment_generator", menace_env)
sys.modules.setdefault("menace", menace_pkg)

import sandbox_runner.generative_stub_provider as gsp


def sample_func(name: str, count: int, active: bool) -> None:
    """Sample function used for stub generation."""
    return None


def test_history_stub_generation(monkeypatch):
    """Stub generation falls back to historical values."""
    class HistDB:
        def recent(self, n):
            return [
                {"name": "foo", "count": 2, "active": True},
            ]

    monkeypatch.setattr(gsp, "_get_history_db", lambda: HistDB())
    gsp._CACHE.clear()

    stubs = gsp.generate_stubs(
        [{}], {"target": sample_func, "strategy": "history"}, context_builder=types.SimpleNamespace(build_prompt=lambda q, *, intent_metadata=None, **k: q)
    )[0]
    assert stubs == {"name": "foo", "count": 2, "active": True}
