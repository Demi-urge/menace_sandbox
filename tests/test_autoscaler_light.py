import importlib
import logging
import subprocess

import autoscaler as a


def test_light_mode_bootstrap(monkeypatch, caplog):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    importlib.reload(a)

    calls = []

    def fake_run(cmd, check=False, stdout=None, stderr=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    caplog.set_level(logging.INFO)
    boot = a.EnvironmentBootstrapper()
    boot.deploy_across_hosts(["h1", "h2"])

    expected = ["ssh", "h1", "python3", "-m", "menace.environment_bootstrap"]
    assert expected in calls
    assert "bootstrapping h1" in caplog.text
    assert "bootstrap succeeded on h1" in caplog.text


def test_light_mode_retry(monkeypatch, caplog):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    import importlib
    importlib.reload(a)

    calls = []

    def fake_run(cmd, check=False, stdout=None, stderr=None):
        calls.append(cmd)
        if cmd[0] == "ssh" and calls.count(cmd) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    import retry_utils
    monkeypatch.setattr(retry_utils.time, "sleep", lambda *_: None)
    caplog.set_level(logging.WARNING)
    boot = a.EnvironmentBootstrapper()
    boot.deploy_across_hosts(["h1"])

    assert calls.count(["ssh", "h1", "python3", "-m", "menace.environment_bootstrap"]) == 2
    assert "retry 1/3" in caplog.text


def test_light_deploy_runs_preparation(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("MENACE_OS_PACKAGES", "pkg")
    monkeypatch.setenv("BOOTSTRAP_SECRET_NAMES", "sec")
    import importlib
    importlib.reload(a)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))

    calls: list[str] = []

    monkeypatch.setattr(a.EnvironmentBootstrapper, "check_commands", lambda self, cmds, m="check_commands": calls.append(m))
    monkeypatch.setattr(a.EnvironmentBootstrapper, "check_os_packages", lambda self, pkgs, m="check_os_packages": calls.append(m))
    monkeypatch.setattr(a.EnvironmentBootstrapper, "export_secrets", lambda self, m="export_secrets": calls.append(m))
    monkeypatch.setattr(a.EnvironmentBootstrapper, "run_migrations", lambda self, m="run_migrations": calls.append(m))

    boot = a.EnvironmentBootstrapper()
    boot.deploy_across_hosts(["h1"])

    assert set(calls) == {"check_commands", "check_os_packages", "export_secrets", "run_migrations"}

