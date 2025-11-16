import sys
from pathlib import Path
import menace.service_installer as si


def test_generate_k8s(tmp_path, capsys):
    out = tmp_path / "k8s.yaml"
    env = tmp_path / ".env"
    env.write_text(
        "FOO=bar\nCPU_LIMIT=1\nMEMORY_LIMIT=128Mi\nMENACE_VOLUMES=/h:/c\n"
    )
    si._generate_k8s(out, env_file=str(env))
    data = out.read_text()
    assert "Deployment" in data
    assert "menace.service_supervisor" in data
    assert "FOO" in data
    assert "/c" in data
    assert "128Mi" in data
    msg = capsys.readouterr().out
    assert "kubectl apply" in msg


def test_cli_swarm(tmp_path, capsys):
    out = tmp_path / "compose.yml"
    env = tmp_path / ".env"
    env.write_text("BAR=baz\nMENACE_VOLUMES=/h2:/c2\n")
    si.main(
        ["--orchestrator", "swarm", "--output", str(out), "--env-file", str(env)]
    )
    data = out.read_text()
    assert "docker stack deploy" in capsys.readouterr().out
    assert "menace.service_supervisor" in data
    assert "BAR" in data
    assert "/c2" in data

