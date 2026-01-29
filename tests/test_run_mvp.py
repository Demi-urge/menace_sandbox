import json
import re
from pathlib import Path

import pytest

import run_mvp


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _assert_clean_output(output: str) -> None:
    assert "Traceback" not in output
    assert _ANSI_RE.search(output) is None


def _assert_error_result(exit_code: int, out: str, err: str) -> dict:
    assert exit_code != 0
    assert err == ""
    _assert_clean_output(out)
    payload = json.loads(out)
    assert payload.get("success") is False
    error_message = payload.get("error", "")
    assert isinstance(error_message, str)
    assert error_message.strip()
    assert "\n" not in error_message
    assert len(error_message) < 200
    return payload


def test_run_mvp_valid_task_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps({"objective": "Test objective", "constraints": ["no network"]}),
        encoding="utf-8",
    )

    exit_code = run_mvp.main(["--task", str(task_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    _assert_clean_output(captured.out)

    payload = json.loads(captured.out)
    expected_keys = {
        "objective",
        "constraints",
        "generated_code",
        "execution_output",
        "execution_error",
        "evaluation_error",
        "roi_score",
        "started_at",
        "finished_at",
        "duration_ms",
        "success",
    }
    for key in expected_keys:
        assert key in payload
    assert "success" in payload


@pytest.mark.parametrize(
    "content",
    [
        "{malformed json}",
        "",
        json.dumps(["not", "a", "dict"]),
        json.dumps("not a dict"),
    ],
)
def test_run_mvp_invalid_task_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    task_path = tmp_path / "task.json"
    task_path.write_text(content, encoding="utf-8")

    exit_code = run_mvp.main(["--task", str(task_path)])
    captured = capsys.readouterr()

    _assert_error_result(exit_code, captured.out, captured.err)


def test_run_mvp_missing_task_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_mvp.main(["--task", "missing.json"])
    captured = capsys.readouterr()

    _assert_error_result(exit_code, captured.out, captured.err)
