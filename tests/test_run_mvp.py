import json
import re

import run_mvp


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _assert_json_output_sanitized(raw_output: str) -> dict:
    assert raw_output.strip(), "Expected JSON output"
    assert "Traceback" not in raw_output
    assert not ANSI_ESCAPE_RE.search(raw_output)
    return json.loads(raw_output)


def test_main_success(tmp_path, monkeypatch, capsys):
    payload = {"objective": "demo", "constraints": []}
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    expected_result = {"success": True, "payload": payload}
    monkeypatch.setattr(run_mvp.mvp_workflow, "execute_task", lambda data: expected_result)

    exit_code = run_mvp.main(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    output = _assert_json_output_sanitized(captured.stdout)
    assert output["success"] is True
    assert output == expected_result


def test_main_malformed_json(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("{", encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    output = _assert_json_output_sanitized(captured.stdout)
    assert "Task file is not valid JSON" in output["error"]


def test_main_empty_file(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("", encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    output = _assert_json_output_sanitized(captured.stdout)
    assert "Task file is empty or whitespace-only" in output["error"]


def test_main_missing_file(tmp_path, capsys):
    task_path = tmp_path / "missing.json"

    exit_code = run_mvp.main(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    output = _assert_json_output_sanitized(captured.stdout)
    assert "Unable to read task file" in output["error"]


def test_main_invalid_payload_shape(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    captured = capsys.readouterr()
    assert exit_code != 0
    output = _assert_json_output_sanitized(captured.stdout)
    assert "Task payload must be a JSON object" in output["error"]
