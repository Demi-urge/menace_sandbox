"""Minimal CLI for running MVP workflow tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mvp_workflow


def _emit_json(payload: dict, *, stream: object) -> None:
    message = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write(message + "\n")


def _error_payload(message: str) -> dict:
    return {"success": False, "error": mvp_workflow.sanitize_error_output(message)}


def _read_task_file(path: str) -> tuple[dict | None, str | None]:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, "task file not found"

    if not content or not content.strip():
        return None, "task file is empty"

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, mvp_workflow.sanitize_error_output(str(exc))

    if not isinstance(payload, dict):
        return None, "task payload must be a JSON object"

    return payload, None


def _validate_task_fields(payload: dict) -> str | None:
    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        return "task payload missing non-empty 'objective'"
    return None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to JSON task payload")
    args = parser.parse_args(argv)

    payload, error = _read_task_file(args.task)
    if error:
        _emit_json(_error_payload(error), stream=sys.stdout)
        return 1

    field_error = _validate_task_fields(payload)
    if field_error:
        _emit_json(_error_payload(field_error), stream=sys.stdout)
        return 1

    try:
        result = mvp_workflow.execute_task(payload)
    except Exception as exc:  # pragma: no cover - defensive
        message = mvp_workflow.sanitize_error_output(str(exc))
        _emit_json(_error_payload(message), stream=sys.stdout)
        return 1

    _emit_json(result, stream=sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
