from __future__ import annotations

"""CLI for comparing two workflow specifications using synergy heuristics."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from workflow_synergy_comparator import WorkflowSynergyComparator


def cli(argv: list[str] | None = None) -> int:
    """Run the workflow synergy comparison command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_a", help="First workflow file or identifier")
    parser.add_argument("workflow_b", help="Second workflow file or identifier")
    parser.add_argument("--out", default="-", help="Output file or '-' for stdout")
    args = parser.parse_args(argv)

    scores = WorkflowSynergyComparator.compare(args.workflow_a, args.workflow_b)
    duplicate = WorkflowSynergyComparator.is_duplicate(scores)
    data = json.dumps({"duplicate": duplicate, **asdict(scores)}, indent=2)

    if args.out == "-":
        sys.stdout.write(data)
    else:
        Path(args.out).write_text(data, encoding="utf-8")

    return 0


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


__all__ = ["cli", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
