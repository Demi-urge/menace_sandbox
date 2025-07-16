import subprocess
from menace.deployment_bot import DeploymentBot, DeploymentDB
from menace.unified_event_bus import UnifiedEventBus


def _make_fake_run(test_success=True):
    calls = []

    def fake_run(cmd, check=False):
        calls.append(cmd)
        if "pytest" in cmd[-1]:
            rc = 0 if test_success else 1
            return subprocess.CompletedProcess(cmd, rc)
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run, calls


def test_auto_update_success(monkeypatch):
    fake_run, calls = _make_fake_run(test_success=True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("nodes:update", lambda t, e: events.append(e))
    bot = DeploymentBot(DeploymentDB(":memory:"), event_bus=bus)
    bot.auto_update_nodes(["host"])

    assert ["ssh", "host", "git pull origin main && pytest -q"] in calls
    assert ["ssh", "host", "docker compose build --pull && docker compose up -d"] in calls
    assert events == [{"node": "host", "status": "success"}]


def test_auto_update_tests_fail(monkeypatch):
    fake_run, calls = _make_fake_run(test_success=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("nodes:update", lambda t, e: events.append(e))
    bot = DeploymentBot(DeploymentDB(":memory:"), event_bus=bus)
    bot.auto_update_nodes(["host"])

    assert ["ssh", "host", "git pull origin main && pytest -q"] in calls
    assert ["ssh", "host", "docker compose build --pull && docker compose up -d"] not in calls
    assert events == [{"node": "host", "status": "failed"}]
