#!/usr/bin/env python3
"""CLI for generating workflows using :class:`WorkflowSynthesizer`.

Workflows are ranked by combining synergy graph and intent scores using
configurable weights that are normalised by workflow length. Generated
specifications can be written to the local sandbox and existing ones listed
via the ``--list`` flag."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_synthesizer import (
    WorkflowSynthesizer,
    to_workflow_spec,
    evaluate_workflow,
    workflow_to_dict,
    save_workflow,
)


def run(args: argparse.Namespace) -> int:
    """Handle command line arguments to synthesise or list workflows.

    Parameters
    ----------
    args:
        Parsed command line arguments.

    Returns
    -------
    int
        Process exit code, ``0`` for success.
    """

    if getattr(args, "list", False):
        directory = Path("sandbox_data/generated_workflows")
        if directory.exists():
            for path in sorted(directory.glob("*.workflow.json")):
                print(path.name)
        return 0

    synth = WorkflowSynthesizer()
    workflows = synth.generate_workflows(
        start_module=args.start,
        problem=args.problem,
        limit=getattr(args, "limit", 5),
        max_depth=args.max_depth,
        synergy_weight=getattr(args, "synergy_weight", 1.0),
        intent_weight=getattr(args, "intent_weight", 1.0),
    )
    scores = getattr(synth, "workflow_scores", [])
    for idx, (wf, score) in enumerate(zip(workflows, scores), start=1):
        modules = " -> ".join(step["module"] for step in workflow_to_dict(wf)["steps"])
        print(f"{idx}. {score:.4f} {modules}")

    selected = 0
    if len(workflows) > 1 and (
        getattr(args, "out", None) or args.save is not None or getattr(args, "evaluate", False)
    ):
        while True:
            choice = input(f"Select workflow [1-{len(workflows)}] (default 1): ").strip()
            if not choice:
                break
            if choice.isdigit() and 1 <= int(choice) <= len(workflows):
                selected = int(choice) - 1
                break
            print("invalid selection")

    if (
        getattr(args, "out", None)
        or args.save is not None
        or getattr(args, "evaluate", False)
    ) and not workflows:
        print("no workflow generated")
        return 1

    chosen = workflows[selected] if workflows else []
    if args.out:
        save_workflow(chosen, args.out)
    elif args.save is not None:
        name = args.save or args.start
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        directory = Path("sandbox_data/generated_workflows")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe}.workflow.json"
        save_workflow(chosen, path)
    if getattr(args, "evaluate", False):
        spec = to_workflow_spec(chosen)
        ok = evaluate_workflow(spec)
        print("evaluation succeeded" if ok else "evaluation failed")
        return 0 if ok else 1
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Create the argument parser for the workflow synthesiser CLI."""

    parser = parser or argparse.ArgumentParser(
        description=(
            "Generate candidate workflows from a starting module. Scores blend "
            "synergy graph edge weights with intent matches and are normalised "
            "by workflow length."
        ),
    )
    parser.add_argument(
        "start",
        nargs="?",
        help="Starting module name",
    )
    parser.add_argument(
        "--problem",
        help="Optional problem statement to guide workflow generation",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        dest="max_depth",
        help="Maximum traversal depth when exploring module connections",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of workflows to generate",
    )
    parser.add_argument(
        "--out",
        help="File or directory to save generated workflows",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        help=(
            "Save workflow to sandbox_data/generated_workflows/<name>.workflow.json "
            "(default name derived from start)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved workflow specifications and exit",
    )
    parser.add_argument(
        "--synergy-weight",
        type=float,
        default=1.0,
        help="Weight applied to synergy scores when ranking workflows",
    )
    parser.add_argument(
        "--intent-weight",
        type=float,
        default=1.0,
        help="Weight applied to intent scores when ranking workflows",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Execute the generated workflow and report success",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the standalone workflow synthesiser CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list and args.start is None:
        parser.error("start is required unless --list is specified")
    return run(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
