"""Minimal CLI for running MVP workflow tasks."""

from __future__ import annotations

import argparse
import json
import sys

from mvp_workflow import execute_task


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Path to JSON task payload")
    args = parser.parse_args(argv)

    try:
        with open(args.task, encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        sys.stderr.write(f"Unable to read task file: {exc}\n")
        return 1
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Task file is not valid JSON: {exc}\n")
        return 1

    if not isinstance(payload, dict):
        sys.stderr.write("Task payload must be a JSON object.\n")
        return 1

    try:
        result = execute_task(payload)
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Task execution failed: {exc}\n")
        return 1

    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write(output + "\n")
    return 0


def main() -> None:
    sys.exit(cli())


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
