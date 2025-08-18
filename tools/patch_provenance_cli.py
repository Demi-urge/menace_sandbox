#!/usr/bin/env python3
"""CLI for querying patch provenance information."""

from __future__ import annotations

import argparse
import json
import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB
from patch_provenance import (
    get_patch_provenance,
    build_chain,
    search_patches_by_vector,
    search_patches_by_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch provenance queries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="list recent patches")
    list_p.add_argument("--limit", type=int, default=20)

    show_p = sub.add_parser("show", help="show provenance for a patch")
    show_p.add_argument("patch_id", type=int)

    search_p = sub.add_parser("search", help="search patches by vector or code hash")
    search_p.add_argument("term", help="vector id or code hash")
    search_p.add_argument(
        "--hash",
        action="store_true",
        help="interpret TERM as a code snippet hash",
    )

    args = parser.parse_args()
    db = PatchHistoryDB()

    if args.cmd == "list":
        rows = db.list_patches(args.limit)
        patches = [
            {"id": pid, "filename": rec.filename, "description": rec.description}
            for pid, rec in rows
        ]
        print(json.dumps(patches))
    elif args.cmd == "show":
        rec = db.get(args.patch_id)
        if rec is None:
            print(json.dumps({"error": "not found"}))
            return
        prov = get_patch_provenance(args.patch_id, patch_db=db)
        chain = build_chain(args.patch_id, patch_db=db)
        out = {
            "id": args.patch_id,
            "record": {
                "filename": rec.filename,
                "description": rec.description,
                "parent_patch_id": rec.parent_patch_id,
            },
            "provenance": prov,
            "chain": chain,
        }
        print(json.dumps(out))
    elif args.cmd == "search":
        if args.hash:
            rows = search_patches_by_hash(args.term, patch_db=db)
            patches = [
                {
                    "id": r["patch_id"],
                    "filename": r["filename"],
                    "description": r["description"],
                }
                for r in rows
            ]
        else:
            rows = search_patches_by_vector(args.term, patch_db=db)
            patches = [
                {
                    "id": r["patch_id"],
                    "filename": r["filename"],
                    "description": r["description"],
                    "influence": r["influence"],
                }
                for r in rows
            ]
        print(json.dumps(patches))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
