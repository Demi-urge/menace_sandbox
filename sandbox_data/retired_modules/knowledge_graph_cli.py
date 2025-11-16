"""Simple CLI for inspecting knowledge graph insights."""
from __future__ import annotations

import argparse
from typing import Iterable

from knowledge_graph import KnowledgeGraph


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge graph utilities")
    parser.add_argument(
        "--top-insights",
        type=int,
        default=5,
        help="Show top N GPT insights and their links",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    kg = KnowledgeGraph()
    for key, neighbors in kg.top_insights(args.top_insights):
        links = ", ".join(neighbors)
        print(f"{key}: {links}")


if __name__ == "__main__":  # pragma: no cover
    main()
