import subprocess

import pytest

from menace.external_dependency_provisioner import ExternalDependencyProvisioner


def test_detect_compose_command_prefers_docker_compose(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["docker", "compose", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Docker Compose version v2", "")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = ExternalDependencyProvisioner()

    assert prov._detect_compose_command() == ["docker", "compose"]
    assert calls == [["docker", "compose", "--version"]]


def test_detect_compose_command_falls_back_to_legacy(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd == ["docker", "compose", "--version"]:
            return subprocess.CompletedProcess(cmd, 1, "", "not supported")
        if cmd == ["docker-compose", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "docker-compose version 1.29", "")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = ExternalDependencyProvisioner()

    assert prov._detect_compose_command() == ["docker-compose"]


def test_detect_compose_command_reports_versions(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd in (["docker", "compose", "--version"], ["docker-compose", "--version"]):
            return subprocess.CompletedProcess(cmd, 1, "", "missing")
        if cmd == ["docker", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Docker version 25.0", "")
        return subprocess.CompletedProcess(cmd, 1, "", "n/a")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = ExternalDependencyProvisioner()

    with pytest.raises(RuntimeError) as exc_info:
        prov._detect_compose_command()

    msg = str(exc_info.value)
    assert "Attempted probes:" in msg
    assert "Detected versions:" in msg
    assert "MENACE_EXTERNAL_DEPS_MANAGED_EXTERNALLY=1" in msg


def test_provision_skips_when_external_management_enabled(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    monkeypatch.setenv("MENACE_EXTERNAL_DEPS_MANAGED_EXTERNALLY", "1")

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when provisioning is skipped")

    monkeypatch.setattr(subprocess, "run", fail_run)
    prov = ExternalDependencyProvisioner(str(compose))

    assert prov.provision() is False


def test_provision_uses_detected_command(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["docker", "compose", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Docker Compose version v2", "")
        if cmd[:2] == ["docker", "compose"] and cmd[-2:] == ["up", "-d"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prov = ExternalDependencyProvisioner(str(compose))

    assert prov.provision() is True
    assert ["docker", "compose", "-f", str(compose), "up", "-d"] in calls
