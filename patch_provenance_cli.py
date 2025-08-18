#!/usr/bin/env python3
"""CLI for querying patch provenance information."""

from __future__ import annotations

import argparse
import json
import sys
import types
from typing import Any

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB
from patch_provenance import get_patch_provenance, search_patches_by_vector


def _rec_to_dict(rec: Any) -> dict:
    return {
        "filename": rec.filename,
        "description": rec.description,
        "roi_before": rec.roi_before,
        "roi_after": rec.roi_after,
        "errors_before": rec.errors_before,
        "errors_after": rec.errors_after,
        "roi_delta": rec.roi_delta,
        "complexity_before": rec.complexity_before,
        "complexity_after": rec.complexity_after,
        "complexity_delta": rec.complexity_delta,
        "entropy_before": rec.entropy_before,
        "entropy_after": rec.entropy_after,
        "entropy_delta": rec.entropy_delta,
        "predicted_roi": rec.predicted_roi,
        "predicted_errors": rec.predicted_errors,
        "reverted": rec.reverted,
        "trending_topic": rec.trending_topic,
        "ts": rec.ts,
        "code_id": rec.code_id,
        "code_hash": rec.code_hash,
        "source_bot": rec.source_bot,
        "version": rec.version,
        "parent_patch_id": rec.parent_patch_id,
        "reason": rec.reason,
        "trigger": rec.trigger,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Query patch provenance")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    show_p = sub.add_parser("show")
    show_p.add_argument("patch_id", type=int)
    search_p = sub.add_parser("search")
    search_p.add_argument("term")
    args = parser.parse_args()

    db = PatchHistoryDB()
    if args.cmd == "list":
        patches = [
            {"id": pid, "filename": rec.filename, "description": rec.description}
            for pid, rec in db.list_patches()
        ]
        print(json.dumps(patches))
    elif args.cmd == "show":
        rec = db.get(args.patch_id)
        if rec is None:
            raise SystemExit(1)
        prov = get_patch_provenance(args.patch_id, patch_db=db)
        out = {
            "id": args.patch_id,
            "record": _rec_to_dict(rec),
            "provenance": prov,
        }
        print(json.dumps(out))
    elif args.cmd == "search":
        res = search_patches_by_vector(args.term, patch_db=db)
        if res:
            patches = [
                {
                    "id": r["patch_id"],
                    "filename": r["filename"],
                    "description": r["description"],
                    "contribution": r["contribution"],
                }
                for r in res
            ]
        else:
            patches = [
                {
                    "id": pid,
                    "filename": rec.filename,
                    "description": rec.description,
                }
                for pid, rec in db.search_with_ids(args.term)
            ]
        print(json.dumps(patches))


if __name__ == "__main__":
    main()
