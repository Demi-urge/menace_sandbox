import subprocess
from menace.external_dependency_provisioner import ExternalDependencyProvisioner


def test_provision_runs_when_compose_exists(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("version: '3'")
    calls = []

    def fake_call(cmd, **kwargs):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    prov = ExternalDependencyProvisioner(str(compose))
    prov.provision()
    assert ["docker", "compose", "-f", str(compose), "up", "-d"] in calls


def test_provision_creates_default_compose(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    calls = []

    def fake_call(cmd, **kwargs):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    prov = ExternalDependencyProvisioner(str(compose))
    prov.provision()

    assert compose.exists()
    text = compose.read_text()
    assert "rabbitmq" in text
    assert ["docker", "compose", "-f", str(compose), "up", "-d"] in calls
