"""CLI to visualise shared vs. local table usage from DBRouter metrics."""

from __future__ import annotations

import argparse
from typing import Dict

from db_router import GLOBAL_ROUTER
import telemetry_backend


def _format_counts(counts: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = []
    for menace, ops in counts.items():
        lines.append(f"{menace}:")
        for kind, tables in ops.items():
            lines.append(f"  {kind}:")
            for table, count in sorted(tables.items()):
                lines.append(f"    {table}: {int(count)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display DBRouter table usage metrics"
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Flush pending counts from GLOBAL_ROUTER before displaying",
    )
    args = parser.parse_args()

    if args.flush and GLOBAL_ROUTER is not None:
        GLOBAL_ROUTER.get_access_counts(flush=True)

    counts = telemetry_backend.get_table_access_counts()
    if not counts:
        print("No metrics recorded")
        return
    print(_format_counts(counts))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
