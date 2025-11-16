import subprocess
from menace.clipped import clipped_master
from menace.dynamic_path_router import resolve_path


def test_run_script_merges_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, check, env=None):
        captured['env'] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    script = tmp_path / "dummy.py"  # path-ignore
    script.write_text("print('hi')")

    monkeypatch.setenv("DISPLAY", ":1")
    clipped_master.run_script(str(script), env={"CUSTOM": "1"}, args=[])
    assert captured["env"]["DISPLAY"] == ":1"
    assert captured["env"]["CUSTOM"] == "1"


def test_cli_resolves_default_script_from_other_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_scripts(scripts, env=None, parallel=False):
        captured["scripts"] = scripts
        return []

    monkeypatch.setattr(clipped_master, "run_scripts", fake_run_scripts)
    monkeypatch.chdir(tmp_path)
    clipped_master.main([])
    expected = str(resolve_path("menace_master.py"))  # path-ignore
    assert captured["scripts"] == [expected]
