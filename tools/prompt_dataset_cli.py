#!/usr/bin/env python3
"""Export PromptDB logs as training datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any

from db_router import DBRouter, LOCAL_TABLES


def _load_rows(db_path: Path) -> Iterable[Dict[str, Any]]:
    LOCAL_TABLES.add("prompts")
    router = DBRouter("prompts", str(db_path), str(db_path))
    conn = router.get_connection("prompts", operation="read")
    cur = conn.cursor()
    cur.execute(
        "SELECT prompt, response_text, outcome_tags, vector_confidence FROM prompts"
    )
    for prompt, response, tags_json, conf in cur.fetchall():
        tags = json.loads(tags_json) if tags_json else []
        yield {
            "prompt": prompt,
            "response": response,
            "tags": tags,
            "confidence": conf,
        }


def _filter_rows(
    rows: Iterable[Dict[str, Any]],
    tags: List[str] | None,
    min_conf: float | None,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for row in rows:
        if tags and not any(t in row["tags"] for t in tags):
            continue
        if min_conf is not None and (
            row["confidence"] is None or row["confidence"] < min_conf
        ):
            continue
        results.append({"prompt": row["prompt"], "completion": row["response"]})
    return results


def _write_jsonl(samples: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _write_csv(samples: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prompt", "completion"])
        writer.writeheader()
        writer.writerows(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PromptDB for fine-tuning")
    parser.add_argument("output", help="Path to export file")
    parser.add_argument("--db", default=os.getenv("PROMPT_DB_PATH", "prompts.db"))
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--tag", action="append", help="Filter by outcome tag")
    parser.add_argument("--min-confidence", type=float, dest="min_conf")
    args = parser.parse_args()

    db_path = Path(args.db)
    rows = _load_rows(db_path)
    samples = _filter_rows(rows, args.tag, args.min_conf)

    out_path = Path(args.output)
    if args.format == "jsonl":
        _write_jsonl(samples, out_path)
    else:
        _write_csv(samples, out_path)
    print(json.dumps({"written": len(samples)}))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
