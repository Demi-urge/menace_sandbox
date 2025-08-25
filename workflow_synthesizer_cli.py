#!/usr/bin/env python3
"""CLI for generating workflows using :class:`WorkflowSynthesizer`.

Workflows are ranked by combining synergy graph and intent scores using
configurable weights that are normalised by workflow length."""

from __future__ import annotations

import argparse
import json

from workflow_synthesizer import WorkflowSynthesizer


def run(args: argparse.Namespace) -> int:
    """Handle command line arguments to synthesise workflows.

    Parameters
    ----------
    args:
        Parsed command line arguments.

    Returns
    -------
    int
        Process exit code, ``0`` for success.
    """

    synth = WorkflowSynthesizer()
    workflows = synth.generate_workflows(
        start_module=args.start,
        problem=args.problem,
        max_depth=args.max_depth,
        synergy_weight=args.synergy_weight,
        intent_weight=args.intent_weight,
    )
    data = [
        {"score": score, "steps": wf}
        for wf, score in zip(workflows, getattr(synth, "workflow_scores", []))
    ]
    print(json.dumps(data, indent=2))
    if args.out:
        synth.save(args.out)
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
        "--out",
        help="File or directory to save generated workflows",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the standalone workflow synthesiser CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

