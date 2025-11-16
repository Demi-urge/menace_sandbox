import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_flakiness_runs_settings_override(monkeypatch):
    from menace_sandbox.sandbox_settings import SandboxSettings

    monkeypatch.delenv("FLAKINESS_RUNS", raising=False)
    assert SandboxSettings().flakiness_runs == 5

    monkeypatch.setenv("FLAKINESS_RUNS", "7")
    assert SandboxSettings().flakiness_runs == 7


def test_self_test_service_container_retries(monkeypatch):
    monkeypatch.setenv("SELF_TEST_RETRIES", "5")
    vec_mod = types.ModuleType("vector_service.context_builder")
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            return "ctx", "s", {}

    vec_mod.ContextBuilder = DummyBuilder
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", vec_mod)
    import menace_sandbox.self_test_service as sts
    importlib.reload(sts)

    class DummyErrorLogger:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(sts, "ErrorLogger", DummyErrorLogger)
    svc = sts.SelfTestService(context_builder=DummyBuilder())
    assert svc.container_retries == 5

    monkeypatch.delenv("SELF_TEST_RETRIES", raising=False)
    importlib.reload(sts)
    monkeypatch.setattr(sts, "ErrorLogger", DummyErrorLogger)
    svc_default = sts.SelfTestService(
        container_retries=2, context_builder=DummyBuilder()
    )
    assert svc_default.container_retries == 2


def test_orphan_retry_settings_override(monkeypatch):
    from menace_sandbox.sandbox_settings import SandboxSettings

    monkeypatch.delenv("ORPHAN_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("ORPHAN_RETRY_DELAY", raising=False)
    cfg = SandboxSettings()
    assert cfg.orphan_retry_attempts == 3
    assert cfg.orphan_retry_delay == 0.1

    monkeypatch.setenv("ORPHAN_RETRY_ATTEMPTS", "6")
    monkeypatch.setenv("ORPHAN_RETRY_DELAY", "0.5")
    cfg_override = SandboxSettings()
    assert cfg_override.orphan_retry_attempts == 6
    assert cfg_override.orphan_retry_delay == 0.5
