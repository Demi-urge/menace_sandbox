import subprocess
import time
from menace.local_infrastructure_provisioner import LocalInfrastructureProvisioner


def test_up_creates_default_compose(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    calls = []

    def fake_call(cmd, **kwargs):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    monkeypatch.setattr(LocalInfrastructureProvisioner, "_containers_running", lambda self: True)
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    prov = LocalInfrastructureProvisioner(str(compose))
    prov.up()

    content = compose.read_text()
    assert "rabbitmq" in content
    assert "postgres" in content
    assert "vault" in content
    assert ["docker", "compose", "-f", str(compose), "up", "-d"] in calls


def test_password_is_generated_and_file_rewritten(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"

    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    prov1 = LocalInfrastructureProvisioner(str(compose))
    prov1.up()
    text1 = compose.read_text()
    assert "POSTGRES_PASSWORD: pass" not in text1
    line = [l for l in text1.splitlines() if "POSTGRES_PASSWORD" in l][0]
    pwd1 = line.split(":", 1)[1].strip()
    assert len(pwd1) > 8

    prov2 = LocalInfrastructureProvisioner(str(compose))
    prov2.up()
    text2 = compose.read_text()
    assert text1 != text2


def test_down_stops_containers(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    calls = []

    def fake_call(cmd, **kwargs):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    prov = LocalInfrastructureProvisioner(str(compose))
    prov.ensure_compose_file()
    prov.down()

    assert ["docker", "compose", "-f", str(compose), "down"] in calls
