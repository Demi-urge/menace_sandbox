"""Minimal CLI for running MVP workflow tasks."""

from __future__ import annotations

import argparse
import json
import sys

import mvp_workflow


def _write_error(message: str, details: str | None = None) -> None:
    sanitized_message = mvp_workflow.sanitize_error_output(message)
    payload: dict[str, str] = {"error": sanitized_message}
    if details:
        payload["details"] = mvp_workflow.sanitize_error_output(details)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_task_file(path: str) -> tuple[dict | None, str | None]:
    try:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        return None, mvp_workflow.sanitize_error_output(f"Unable to read task file: {exc}")

    if not content or not content.strip():
        return None, "Task file is empty or whitespace-only."

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, mvp_workflow.sanitize_error_output(f"Task file is not valid JSON: {exc}")

    if not isinstance(payload, dict):
        return None, "Task payload must be a JSON object."

    return payload, None


def _validate_task_fields(payload: dict) -> str | None:
    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        return "Task field 'objective' must be a non-empty string."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to JSON task payload")
    args = parser.parse_args(argv)

    payload, error = _read_task_file(args.task)
    if error:
        _write_error(error)
        return 1

    field_error = _validate_task_fields(payload)
    if field_error:
        _write_error("Missing or invalid task fields.", field_error)
        return 1

    try:
        result = mvp_workflow.execute_task(payload)
    except Exception as exc:  # pragma: no cover - defensive
        _write_error("Task execution failed.", str(exc))
        return 1

    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
