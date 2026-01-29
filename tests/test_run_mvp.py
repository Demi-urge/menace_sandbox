import json

import run_mvp


def _load_json_output(capsys) -> tuple[dict, str]:
    captured = capsys.readouterr()
    output = captured.stdout.strip() or captured.stderr.strip()
    assert output, "Expected JSON output"
    assert "Traceback" not in output
    return json.loads(output), output


def test_main_success(tmp_path, capsys):
    payload = {"objective": "test", "constraints": ["no networking"]}
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    output, _ = _load_json_output(capsys)
    assert exit_code == 0
    assert isinstance(output.get("success"), bool)
    assert output["objective"] == payload["objective"]
    assert output["constraints"] == payload["constraints"]
    assert "roi_score" in output
    assert "success" in output


def test_main_invalid_json(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("{", encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    output, raw_output = _load_json_output(capsys)
    assert exit_code != 0
    assert output["success"] is False
    assert isinstance(output.get("error"), str)
    assert output["error"]
    assert "Traceback" not in raw_output


def test_main_empty_file(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("", encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    output, raw_output = _load_json_output(capsys)
    assert exit_code != 0
    assert output["success"] is False
    assert "empty" in output["error"].lower()
    assert "Traceback" not in raw_output


def test_main_missing_required_fields(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text("{}", encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])

    output, raw_output = _load_json_output(capsys)
    assert exit_code != 0
    assert output["success"] is False
    assert "objective" in output["error"].lower()
    assert "Traceback" not in raw_output
