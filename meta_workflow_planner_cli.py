#!/usr/bin/env python3
"""Command line interface for :class:`MetaWorkflowPlanner` utilities.

The CLI exposes a handful of high level operations:

* ``encode`` – embed a workflow specification and persist the vector.
* ``candidates`` – list high-synergy workflow candidates for a query.
* ``simulate`` – generate a high-synergy pipeline starting from a workflow.
* ``roi-report`` – show ROI impact report for a workflow run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from meta_workflow_planner import (
    MetaWorkflowPlanner,
    find_synergy_candidates,
    find_synergy_chain,
)
from roi_results_db import module_impact_report
try:  # pragma: no cover - optional dependency
    from vector_service.retriever import Retriever  # type: ignore
    from vector_service.context_builder import ContextBuilder  # type: ignore
except Exception:  # pragma: no cover - allow running without retriever
    Retriever = None  # type: ignore
    ContextBuilder = None  # type: ignore


def _cmd_encode(args: argparse.Namespace) -> int:
    """Encode ``args.workflow`` and print the resulting vector."""

    planner = MetaWorkflowPlanner()
    with Path(args.workflow).open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    vec = planner.encode(args.workflow_id, spec)
    print(" ".join(str(x) for x in vec))
    return 0


def _cmd_candidates(args: argparse.Namespace) -> int:
    """List synergy candidates for ``args.workflow_id`` or ``args.embedding``."""

    if args.embedding is not None:
        query: Sequence[float] | str = [float(x) for x in args.embedding.split(",") if x]
    else:
        query = args.workflow_id
    retr: Retriever | None = None
    if Retriever is not None and ContextBuilder is not None:
        try:  # pragma: no cover - best effort
            builder = ContextBuilder()
            retr = Retriever(context_builder=builder)
        except Exception:
            retr = None
    results = (
        find_synergy_candidates(query, top_k=args.top_k, retriever=retr)
        if retr is not None
        else []
    )
    for idx, item in enumerate(results, start=1):
        wf = item["workflow_id"]
        score = item.get("score", 0.0)
        sim = item.get("similarity", 0.0)
        roi = item.get("roi", 0.0)
        print(f"{idx}. {wf} score={score:.4f} sim={sim:.4f} roi={roi:.4f}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Generate a high-synergy pipeline and print its workflow identifiers."""

    chain = find_synergy_chain(args.start, length=args.length)
    if not chain:
        print("no pipeline generated")
        return 1
    print(" -> ".join(chain))
    return 0


def _cmd_roi_report(args: argparse.Namespace) -> int:
    """Output ROI impact report for ``args.workflow_id`` and ``args.run_id``."""

    report = module_impact_report(args.workflow_id, args.run_id, db_path=args.db)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Return argument parser for the meta workflow planner CLI."""

    parser = parser or argparse.ArgumentParser(description="Meta workflow planning utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="Encode workflow specification and persist embedding")
    p_encode.add_argument("workflow_id", help="Identifier for the workflow")
    p_encode.add_argument("workflow", help="Path to workflow specification (JSON)")
    p_encode.set_defaults(func=_cmd_encode)

    p_cand = sub.add_parser("candidates", help="List high-synergy workflow candidates")
    group = p_cand.add_mutually_exclusive_group(required=True)
    group.add_argument("--workflow-id", dest="workflow_id", help="Existing workflow identifier")
    group.add_argument("--embedding", help="Comma separated embedding vector")
    p_cand.add_argument("--top-k", type=int, default=5, help="Number of candidates to display")
    p_cand.set_defaults(func=_cmd_candidates)

    p_sim = sub.add_parser("simulate", help="Generate a high-synergy workflow pipeline")
    p_sim.add_argument("start", help="Starting workflow identifier")
    p_sim.add_argument("--length", type=int, default=5, help="Maximum number of steps in the pipeline")
    p_sim.set_defaults(func=_cmd_simulate)

    p_roi = sub.add_parser("roi-report", help="Show ROI impact report for a workflow run")
    p_roi.add_argument("workflow_id", help="Workflow identifier")
    p_roi.add_argument("run_id", help="Run identifier")
    p_roi.add_argument("--db", default="roi_results.db", help="Path to ROI results database")
    p_roi.set_defaults(func=_cmd_roi_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the meta workflow planner CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
