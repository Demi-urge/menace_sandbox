from __future__ import annotations

"""CLI for comparing workflow specifications or triggering meta-planning."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from workflow_synergy_comparator import WorkflowSynergyComparator
except Exception:  # pragma: no cover - allow tests to patch comparator
    WorkflowSynergyComparator = None  # type: ignore
from meta_workflow_planner import MetaWorkflowPlanner

_CONTEXT_BUILDER = None  # type: ignore


def cli(argv: list[str] | None = None) -> int:
    """Run the workflow synergy comparison command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_a", help="First workflow file or identifier")
    parser.add_argument(
        "workflow_b",
        nargs="?",
        help="Second workflow file or identifier when comparing",
    )
    parser.add_argument("--out", default="-", help="Output file or '-' for stdout")
    parser.add_argument(
        "--meta-plan",
        action="store_true",
        help="Trigger meta-planning run using workflow_a as seed",
    )
    args = parser.parse_args(argv)

    if args.meta_plan:
        global _CONTEXT_BUILDER
        if _CONTEXT_BUILDER is None:  # pragma: no cover - lazy init
            from context_builder_util import create_context_builder

            _CONTEXT_BUILDER = create_context_builder()
        from workflow_chain_simulator import simulate_suggested_chains
        spec = {}
        path_a = Path(args.workflow_a)
        if path_a.exists():
            try:
                spec = json.loads(path_a.read_text())
            except Exception:
                spec = {}
        planner = MetaWorkflowPlanner(context_builder=_CONTEXT_BUILDER)
        target = planner.encode(args.workflow_a, spec)
        outcomes = simulate_suggested_chains(target)
        data = json.dumps(outcomes, indent=2)
    else:
        if WorkflowSynergyComparator is None:  # pragma: no cover - defensive
            raise RuntimeError("WorkflowSynergyComparator is required")
        scores = WorkflowSynergyComparator.compare(args.workflow_a, args.workflow_b or "")
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
