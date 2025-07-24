import json
import types
import sandbox_runner.environment as env


def test_recorded_containers_removed(monkeypatch, tmp_path):
    file = tmp_path / "active.json"
    monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_FILE", file)
    file.write_text(json.dumps(["a", "b"]))

    removed = []

    def fake_run(cmd, **kw):
        if "ps" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="")
        if "rm" in cmd:
            removed.append(cmd[-1])
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))

    env.purge_leftovers()

    assert set(removed) == {"a", "b"}
    assert file.exists()
    assert json.loads(file.read_text()) == []
