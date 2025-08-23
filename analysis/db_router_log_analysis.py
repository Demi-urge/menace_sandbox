#!/usr/bin/env python3
"""Summarise DBRouter audit logs for cross-instance auditing.

The script reads JSON lines produced by ``DB_ROUTER_AUDIT_LOG`` and prints a
count of operations grouped by menace ID, table name and operation type.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyse(path: Path) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - ignore bad lines
                continue
            key = (
                entry.get("menace_id", ""),
                entry.get("table_name", ""),
                entry.get("operation", ""),
            )
            counter[key] += 1
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse DBRouter audit logs")
    parser.add_argument("logfile", type=Path, help="Path to the audit log file")
    args = parser.parse_args()

    counts = analyse(args.logfile)
    for (menace, table, op), count in sorted(counts.items()):
        print(f"{menace}\t{table}\t{op}\t{count}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
