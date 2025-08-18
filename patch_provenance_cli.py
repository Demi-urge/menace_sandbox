#!/usr/bin/env python3
"""CLI for querying patch ancestry information."""

from __future__ import annotations

import argparse
import json
import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB
from patch_provenance import build_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch provenance queries")
    sub = parser.add_subparsers(dest="cmd", required=True)

    show_p = sub.add_parser("show", help="display ancestry chain for a patch")
    show_p.add_argument("patch_id", type=int)

    search_p = sub.add_parser(
        "search-vector", help="list patches influenced by a vector"
    )
    search_p.add_argument("vector_id")

    args = parser.parse_args()

    db = PatchHistoryDB()

    if args.cmd == "show":
        chain = build_chain(args.patch_id, patch_db=db)
        print(json.dumps(chain))
    elif args.cmd == "search-vector":
        rows = db.find_patches_by_vector(args.vector_id)
        patches = [
            {
                "id": pid,
                "filename": fname,
                "description": desc,
                "influence": infl,
            }
            for pid, infl, fname, desc in rows
        ]
        print(json.dumps(patches))


if __name__ == "__main__":
    main()
