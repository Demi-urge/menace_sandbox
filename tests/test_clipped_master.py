import subprocess
from menace.clipped.clipped_master import run_script


def test_run_script_merges_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, check, env=None):
        captured['env'] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    script = tmp_path / "dummy.py"
    script.write_text("print('hi')")

    monkeypatch.setenv("DISPLAY", ":1")
    run_script(str(script), env={"CUSTOM": "1"}, args=[])
    assert captured["env"]["DISPLAY"] == ":1"
    assert captured["env"]["CUSTOM"] == "1"
