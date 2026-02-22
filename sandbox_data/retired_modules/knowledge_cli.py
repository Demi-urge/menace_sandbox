from __future__ import annotations

"""Command line interface for inspecting local GPT memory and insights."""

import argparse
import sys
from typing import Iterable

import os

from menace_sandbox.gpt_memory import INSIGHT
from local_knowledge_module import init_local_knowledge


# ---------------------------------------------------------------------------
def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Search stored memories by tag")
    p_search.add_argument("tag", help="Tag to filter by")
    p_search.add_argument("query", nargs="?", default="", help="Optional text query")
    p_search.add_argument("--limit", type=int, default=5, help="Maximum results to show")

    p_insight = sub.add_parser("insights", help="List recent stored insights")
    p_insight.add_argument("--limit", type=int, default=5, help="Number of insights to show")

    sub.add_parser("refresh", help="Regenerate stored insights from memory")

    args = parser.parse_args(list(argv) if argv is not None else None)

    module = init_local_knowledge(os.getenv("GPT_MEMORY_DB", "gpt_memory.db"))

    if args.cmd == "search":
        entries = module.memory.search_context(args.query, tags=[args.tag], limit=args.limit)
        for e in entries:
            tag_str = ", ".join(e.tags)
            print(f"[{e.timestamp}] {tag_str}\nQ: {e.prompt}\nA: {e.response}\n")
        return 0

    if args.cmd == "insights":
        cur = module.memory.conn.execute(
            "SELECT response, tags FROM interactions "
            "WHERE tags LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{INSIGHT}%", args.limit),
        )
        for response, tag_str in cur.fetchall():
            tags = [t for t in tag_str.split(",") if t and t != INSIGHT]
            tag = ", ".join(tags)
            print(f"{tag}: {response}")
        return 0

    if args.cmd == "refresh":
        module.refresh()
        return 0

    parser.error("unknown command")
    return 1


# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
