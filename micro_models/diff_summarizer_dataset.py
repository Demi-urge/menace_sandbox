"""Export training data for the diff summariser.

This script reads enhancement records from :class:`chatgpt_enhancement_bot.EnhancementDB`
containing ``before_code``, ``after_code`` and ``summary`` fields and writes them to a
JSONL file suitable for finetuning small language models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - fallback for flat layout
    from ..chatgpt_enhancement_bot import EnhancementDB
except Exception:  # pragma: no cover - allow direct execution
    from chatgpt_enhancement_bot import EnhancementDB  # type: ignore


def _iter_records(db: EnhancementDB, limit: int | None) -> Iterable[dict[str, str]]:
    """Yield raw records from ``db`` with all required fields present."""
    fetch_limit = limit if limit is not None else 10**9
    for enh in db.fetch(fetch_limit):
        if not (enh.before_code and enh.after_code and enh.summary):
            continue
        yield {
            "before_code": enh.before_code,
            "after_code": enh.after_code,
            "summary": enh.summary,
        }


def export_dataset(db_path: Path, output: Path, limit: int | None = None) -> int:
    """Export enhancement records from ``db_path`` to ``output``.

    Returns the number of rows written.
    """

    db = EnhancementDB(db_path)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for rec in _iter_records(db, limit):
            json.dump(rec, fh)
            fh.write("\n")
            count += 1
    return count


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Export diff summariser dataset")
    parser.add_argument("--db", type=Path, default=EnhancementDB().path, help="Path to enhancements.db")
    parser.add_argument("--out", type=Path, default=Path("diff_summarizer_dataset.jsonl"), help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on rows")
    args = parser.parse_args()
    count = export_dataset(args.db, args.out, args.limit)
    print(f"wrote {count} records to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
