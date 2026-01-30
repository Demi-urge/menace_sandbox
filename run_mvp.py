"""Minimal CLI for running MVP workflow tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mvp_workflow
from menace_sandbox.stabilization import (
    ValidationError,
    normalize_error_response,
    normalize_mvp_response,
)


class _CliArgumentError(Exception):
    def __init__(self, message: str, status: int = 2, emitted: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.emitted = emitted


class _ArgumentParser(argparse.ArgumentParser):
    def _sanitize_message(self, message: str) -> str:
        cleaned = " ".join(str(message).split())
        return f"error: {cleaned}" if cleaned else "error: invalid arguments"

    def _emit_error(self, message: str, status: int) -> None:
        sanitized = self._sanitize_message(message)
        _emit_json(_error_payload(sanitized))
        raise _CliArgumentError(sanitized, status=status, emitted=True)

    def error(self, message: str) -> None:
        self._emit_error(message, status=2)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            if message:
                sys.stdout.write(message)
            return
        self._emit_error(message or "invalid arguments", status=status)


def _emit_json(payload: dict) -> None:
    message = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write(message + "\n")


def _error_payload(message: str) -> dict:
    return normalize_error_response({"error": message, "success": False})


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _sanitize_exception_message(message: str) -> str:
    cleaned = _strip_ansi(message)
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


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
    try:
        args = parser.parse_args(argv)
    except _CliArgumentError as exc:
        if not exc.emitted:
            _emit_json(_error_payload(str(exc)))
        return exc.status

    payload, error = _read_task_file(args.task)
    if error:
        _emit_json(_error_payload(error))
        return 1

    try:
        result = normalize_mvp_response(mvp_workflow.execute_task(payload))
    except ValidationError:
        _emit_json(_error_payload("response schema validation failed"))
        return 1
    _emit_json(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except Exception as exc:
        raw_message = str(exc) or "unexpected error"
        message = _sanitize_exception_message(raw_message) or "unexpected error"
        _emit_json(_error_payload(message))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
