#!/usr/bin/env python3
"""Preflight validator for autonomous preset files.

The checker validates that each supplied JSON document is a preset object or a
list of preset objects and blocks entries that contain forbidden
``user_misuse`` markers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_TOKEN = "user_misuse"


class PresetValidationError(ValueError):
    """Raised when preset JSON does not match the expected structure."""


def _as_entries(data: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise PresetValidationError(
                f"{source}: preset files must contain a JSON object or a list of objects"
            )
        return list(data)

    if isinstance(data, dict):
        presets = data.get("presets")
        if isinstance(presets, list):
            if not all(isinstance(item, dict) for item in presets):
                raise PresetValidationError(
                    f"{source}: 'presets' must be a list of JSON objects"
                )
            return list(presets)
        return [data]

    raise PresetValidationError(
        f"{source}: preset files must contain a JSON object or a list of objects"
    )


def _has_forbidden_failure_mode(value: Any) -> bool:
    if isinstance(value, str):
        return FORBIDDEN_TOKEN in value
    if isinstance(value, list):
        return any(FORBIDDEN_TOKEN in str(item) for item in value)
    return False


def _find_violations(entries: list[dict[str, Any]], source: Path) -> list[str]:
    violations: list[str] = []

    for idx, preset in enumerate(entries):
        scenario_name = str(preset.get("SCENARIO_NAME", ""))
        if FORBIDDEN_TOKEN in scenario_name:
            violations.append(
                f"{source}: preset[{idx}] SCENARIO_NAME contains {FORBIDDEN_TOKEN!r}: {scenario_name!r}"
            )

        failure_modes = preset.get("FAILURE_MODES")
        if _has_forbidden_failure_mode(failure_modes):
            violations.append(
                f"{source}: preset[{idx}] FAILURE_MODES contains {FORBIDDEN_TOKEN!r}: {failure_modes!r}"
            )

    return violations


def check_file(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return [f"{path}: file does not exist"]
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"]

    try:
        entries = _as_entries(data, path)
    except PresetValidationError as exc:
        return [str(exc)]

    return _find_violations(entries, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate preset JSON files and block SCENARIO_NAME/FAILURE_MODES "
            "values that include 'user_misuse'."
        )
    )
    parser.add_argument(
        "preset_files",
        nargs="+",
        help="one or more preset JSON paths to validate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    violations: list[str] = []

    for preset_path in args.preset_files:
        violations.extend(check_file(Path(preset_path)))

    if violations:
        print("Preset preflight failed. Resolve the following issues:", file=sys.stderr)
        for issue in violations:
            print(f" - {issue}", file=sys.stderr)
        print(
            "\nTip: remove 'user_misuse' scenarios or run with explicitly approved preset files only.",
            file=sys.stderr,
        )
        return 1

    print("Preset preflight passed: no forbidden user_misuse scenarios found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
