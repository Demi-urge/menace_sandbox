import importlib
import sys
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
    import menace_sandbox.self_test_service as sts

    importlib.reload(sts)
    svc = sts.SelfTestService()
    assert svc.container_retries == 5

    monkeypatch.delenv("SELF_TEST_RETRIES", raising=False)
    importlib.reload(sts)
    svc_default = sts.SelfTestService(container_retries=2)
    assert svc_default.container_retries == 2
