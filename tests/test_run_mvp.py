import json

import run_mvp


def test_cli_success(tmp_path, monkeypatch, capsys):
    payload = {"objective": "demo", "constraints": []}
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    expected_result = {"status": "ok", "task": payload}
    monkeypatch.setattr(run_mvp, "execute_task", lambda data: expected_result)

    exit_code = run_mvp.cli(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.stderr == ""
    output = json.loads(captured.stdout)
    assert output == expected_result


def test_cli_malformed_json(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("{", encoding="utf-8")

    exit_code = run_mvp.cli(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Task file is not valid JSON" in captured.stderr
    assert "Traceback" not in captured.stderr
    assert captured.stdout == ""


def test_cli_missing_file(tmp_path, capsys):
    task_path = tmp_path / "missing.json"

    exit_code = run_mvp.cli(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Unable to read task file" in captured.stderr
    assert "Traceback" not in captured.stderr
    assert captured.stdout == ""


def test_cli_invalid_payload_shape(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    exit_code = run_mvp.cli(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.stdout == ""
    assert captured.stderr.strip() == "Task payload must be a JSON object."
