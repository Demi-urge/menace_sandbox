"""Export training data for a small error classification model.

This script pulls records from :class:`error_bot.ErrorDB`'s telemetry table
and writes ``stack_trace``, ``cause`` and ``error_type`` fields to a JSONL
file.  The resulting dataset can be used to train lightweight models that
predict an error category and fix suggestion from a stack trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - allow execution from package root
    from ..scope_utils import Scope, apply_scope, build_scope_clause
    from ..error_bot import ErrorDB
except Exception:  # pragma: no cover - flat layout fallback
    from scope_utils import Scope, apply_scope, build_scope_clause  # type: ignore
    from error_bot import ErrorDB  # type: ignore


def _iter_rows(
    db: ErrorDB,
    limit: int | None,
    scope: Scope | str,
    source_menace_id: str,
) -> Iterable[dict[str, str]]:
    """Yield telemetry rows containing required fields."""
    clause, scope_params = build_scope_clause("telemetry", Scope(scope), source_menace_id)
    sql = (
        "SELECT stack_trace, COALESCE(cause, inferred_cause) AS cause, "
        "COALESCE(error_type, category) AS error_type "
        "FROM telemetry "
        "WHERE stack_trace IS NOT NULL AND TRIM(stack_trace) != '' "
        "AND COALESCE(cause, inferred_cause) IS NOT NULL "
        "AND TRIM(COALESCE(cause, inferred_cause)) != '' "
        "AND COALESCE(error_type, category) IS NOT NULL "
        "AND TRIM(COALESCE(error_type, category)) != ''"
    )
    sql = apply_scope(sql, clause)
    params: list[object] = list(scope_params)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    cur = db.conn.execute(sql, params)
    for row in cur.fetchall():
        yield {
            "stack_trace": row["stack_trace"],
            "cause": row["cause"],
            "error_type": row["error_type"],
        }


def export_dataset(
    db_path: Path,
    output: Path,
    limit: int | None = None,
    *,
    scope: Scope | str = Scope.ALL,
    source_menace_id: str | None = None,
) -> int:
    """Export telemetry records from ``db_path`` to ``output``.

    Returns the number of rows written.
    """

    db = ErrorDB(db_path)
    menace_id = source_menace_id or db.router.menace_id
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for rec in _iter_rows(db, limit, scope, menace_id):
            json.dump(rec, fh)
            fh.write("\n")
            count += 1
    return count


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Export error dataset")
    parser.add_argument("--db", type=Path, default=ErrorDB().path, help="Path to errors.db")
    parser.add_argument(
        "--out", type=Path, default=Path("error_dataset.jsonl"), help="Output JSONL path"
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on rows")
    parser.add_argument(
        "--scope",
        choices=[s.value for s in Scope],
        default=Scope.ALL.value,
        help="Menace ID scope for telemetry selection",
    )
    args = parser.parse_args()
    count = export_dataset(args.db, args.out, args.limit, scope=args.scope)
    print(f"wrote {count} records to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
