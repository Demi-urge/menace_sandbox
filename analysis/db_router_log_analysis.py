#!/usr/bin/env python3
"""Summarise DBRouter audit logs for cross-instance auditing.

The script reads JSON lines produced by ``DB_ROUTER_AUDIT_LOG`` and prints
counts of operations.  Per-operation counts are grouped by menace ID, table
name and operation type.  Additional summaries aggregate counts by menace and
table and highlight tables that are written to by multiple menace instances.
These summaries help identify potential shared-table misuse or hotspots.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyse(
    path: Path,
) -> tuple[
    Counter[tuple[str, str, str]],
    Counter[tuple[str, str]],
    dict[str, set[str]],
]:
    """Analyse an audit log file.

    Parameters
    ----------
    path:
        Path to the JSON lines audit log.

    Returns
    -------
    tuple
        ``(op_counter, mt_counter, table_writers)`` where ``op_counter`` groups
        counts by ``(menace_id, table_name, operation)`` and ``mt_counter``
        aggregates counts across operations for ``(menace_id, table_name)``.
        ``table_writers`` maps each table name to the set of menace IDs that
        wrote to it.
    """

    op_counter: Counter[tuple[str, str, str]] = Counter()
    mt_counter: Counter[tuple[str, str]] = Counter()
    table_writers: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - ignore bad lines
                continue
            menace = entry.get("menace_id", "")
            table = entry.get("table_name", "")
            op = entry.get("operation", "")
            op_counter[(menace, table, op)] += 1
            mt_counter[(menace, table)] += 1
            if op == "write":
                table_writers[table].add(menace)
    return op_counter, mt_counter, table_writers


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse DBRouter audit logs")
    parser.add_argument("logfile", type=Path, help="Path to the audit log file")
    args = parser.parse_args()
    op_counts, mt_counts, table_writers = analyse(args.logfile)

    # Detailed counts including operation type
    print("menace\ttable\top\tcount")
    for (menace, table, op), count in sorted(op_counts.items()):
        print(f"{menace}\t{table}\t{op}\t{count}")

    # Aggregated counts across operations
    print("\nmenace\ttable\tcount")
    for (menace, table), count in mt_counts.most_common():
        print(f"{menace}\t{table}\t{count}")

    # Shared table analysis
    shared_tables = {t for t, writers in table_writers.items() if len(writers) > 1}
    if shared_tables:
        writer_counts: Counter[str] = Counter()
        for (menace, table), count in mt_counts.items():
            if table in shared_tables:
                writer_counts[menace] += count
        print("\nTop shared-table writers:")
        for menace, count in writer_counts.most_common(10):
            print(f"{menace}\t{count}")
    else:
        print("\nNo shared table write activity detected")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
