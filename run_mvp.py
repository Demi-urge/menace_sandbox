"""Minimal CLI for running MVP workflow tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mvp_workflow


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.error_message: str | None = None

    def error(self, message: str) -> int:
        self.error_message = f"error: {message}"
        sys.stdout.write(self.error_message + "\n")
        return 2


def _emit_json(payload: dict) -> None:
    message = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write(message + "\n")


def _error_payload(message: str) -> dict:
    return {"error": message, "success": False}


def _read_task_file(path: str) -> tuple[dict | None, str | None]:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, "task file not found or unreadable"

    if not content or not content.strip():
        return None, "task file is empty"

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None, "task file contains invalid JSON"

    if not isinstance(payload, dict):
        return None, "task payload must be a JSON object"

    return payload, None


def _run(argv: list[str] | None) -> int:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to JSON task payload")
    args = parser.parse_args(argv)

    if parser.error_message is not None:
        return 2

    payload, error = _read_task_file(args.task)
    if error:
        _emit_json(_error_payload(error))
        return 1

    result = mvp_workflow.execute_task(payload)
    _emit_json(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except Exception as exc:
        message = str(exc) or "unexpected error"
        _emit_json(_error_payload(message))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
