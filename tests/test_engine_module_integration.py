import sys
import types

from tests.test_self_improvement_logging import _load_engine

# ensure logging utils stub has required functions
log_stub = sys.modules["menace.logging_utils"]

class _Log:
    def info(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass

log_stub.get_logger = lambda name=None: _Log()
log_stub.setup_logging = lambda *a, **k: None
log_stub.set_correlation_id = lambda *a, **k: None

# some modules expect this flag on the menace package
sys.modules["menace"].RAISE_ERRORS = False

# stub SelfTestService before loading engine
class DummySTS:
    def __init__(self, *a, **k):
        self.kwargs = k
        self.results = {}

    async def _run_once(self):
        self.results = {
            "integration": {
                "integrated": ["new_mod.py"],  # path-ignore
                "redundant": ["old_mod.py"],  # path-ignore
            },
            "failed": 0,
        }

sts_mod = types.ModuleType("menace.self_test_service")
sts_mod.SelfTestService = DummySTS
sys.modules["menace.self_test_service"] = sts_mod
sys.modules["self_test_service"] = sts_mod

# stub sandbox_runner.environment
calls: dict[str, object] = {}

def fake_auto_include(mods, recursive=False, **kwargs):
    calls["mods"] = list(mods)
    calls["recursive"] = recursive

env_mod = types.ModuleType("sandbox_runner.environment")
env_mod.auto_include_modules = fake_auto_include
sys.modules["sandbox_runner.environment"] = env_mod
sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.environment = env_mod
sys.modules["sandbox_runner"] = sandbox_runner_pkg

sie = _load_engine()
sys.modules["sandbox_settings"].SandboxSettings = lambda: types.SimpleNamespace(
    sandbox_data_dir="sandbox_data", recursive_isolated=True, auto_include_isolated=True
)
sie.SandboxSettings = sys.modules["sandbox_settings"].SandboxSettings

class DummyMetricsDB:
    def __init__(self):
        self.records = []

    def log_eval(self, cycle, metric, value):
        self.records.append((cycle, metric, value))

class DummyDataBot:
    def __init__(self):
        self.metrics_db = DummyMetricsDB()

class DummyLogger:
    def __init__(self):
        self.logs = []

    def info(self, msg, extra=None):
        self.logs.append((msg, extra))

    def exception(self, *a, **k):
        self.logs.append(("exception", a))

engine = types.SimpleNamespace(
    logger=DummyLogger(),
    data_bot=DummyDataBot(),
    orphan_traces={},
)


def test_engine_integrates_and_records():
    paths = ["new_mod.py", "old_mod.py"]  # path-ignore
    result = sie.SelfImprovementEngine._test_orphan_modules(engine, paths)
    assert result == {"new_mod.py"}  # path-ignore
    assert calls["mods"] == ["new_mod.py"]  # path-ignore
    assert calls["recursive"] is True
    assert engine.orphan_traces.get("old_mod.py", {}).get("redundant") is True  # path-ignore
    # ensure metrics recorded
    metrics = engine.data_bot.metrics_db.records
    passed = [m for m in metrics if m[1] == "self_test_passed"]
    redundant = [m for m in metrics if m[1] == "self_test_redundant"]
    assert passed and redundant
