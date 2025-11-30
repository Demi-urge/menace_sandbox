#!/usr/bin/env python3
"""Summarize recursion-guard offenders from bootstrap logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, TextIO

_REENTRY_KEYWORDS = (
    "prepare_pipeline.bootstrap.reentry_cap_exceeded",
    "prepare-pipeline-bootstrap-reentry-cap-exceeded",
)
_RECURSION_KEYWORDS = (
    "prepare-pipeline-bootstrap-recursion-guard-short-circuit",
    "prepare_pipeline.bootstrap.recursion_guard_promise_short_circuit",
    "prepare_pipeline.bootstrap.recursion_guard_broker_short_circuit",
)
_CALLER_REGEX = re.compile(r"caller_module[\"'=:\s]+(?P<module>[\w\.-/]+)")
_STACK_REGEX = re.compile(r"caller_stack_signature[\"'=:\s]+(?P<stack>[^\s\"]+)")


def _loads_candidate(line: str) -> dict[str, object] | None:
    """Best-effort JSON parse for structured log fragments."""

    for candidate in (line, line[line.find("{") :]):
        if not candidate or "{" not in candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_field(line: str, field: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(line)
    if match:
        return match.group(field)
    return None


def _caller_from_line(line: str) -> tuple[str | None, str | None]:
    parsed = _loads_candidate(line)
    if parsed is not None:
        module = parsed.get("caller_module") or parsed.get("module")
        stack = parsed.get("caller_stack_signature")
        if isinstance(module, str) or isinstance(stack, str):
            return module if isinstance(module, str) else None, stack if isinstance(stack, str) else None

    return _extract_field(line, "module", _CALLER_REGEX), _extract_field(
        line, "stack", _STACK_REGEX
    )


def _interesting_line(line: str) -> bool:
    return any(keyword in line for keyword in (*_REENTRY_KEYWORDS, *_RECURSION_KEYWORDS))


def iter_offenders(streams: Iterable[TextIO]):
    for stream in streams:
        for raw_line in stream:
            if not _interesting_line(raw_line):
                continue
            module, stack = _caller_from_line(raw_line)
            yield module or "<unknown>", stack


def summarize_offenders(streams: Iterable[TextIO], limit: int = 10) -> list[tuple[str, int, str | None]]:
    counts: Counter[str] = Counter()
    stack_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for module, stack in iter_offenders(streams):
        counts[module] += 1
        if stack:
            stack_counts[module][stack] += 1

    summary: list[tuple[str, int, str | None]] = []
    for module, total in counts.most_common(limit):
        top_stack, _ = next(iter(stack_counts[module].most_common(1)), (None, 0))
        summary.append((module, total, top_stack))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Surface top prepare_pipeline recursion-guard offenders from logs."
    )
    parser.add_argument(
        "logfile",
        nargs="*",
        type=argparse.FileType("r"),
        default=[sys.stdin],
        help="Path(s) to log file(s); defaults to stdin.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum offenders to display (default: 10).",
    )
    args = parser.parse_args()

    summary = summarize_offenders(args.logfile, limit=args.limit)
    if not summary:
        print("No recursion guard events detected.")
        return

    print("Top recursion guard offenders:\n")
    for module, count, stack in summary:
        stack_hint = f" | top stack: {stack}" if stack else ""
        print(f"{module}: {count}{stack_hint}")


if __name__ == "__main__":
    main()
