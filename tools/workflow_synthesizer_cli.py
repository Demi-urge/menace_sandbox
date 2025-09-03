#!/usr/bin/env python3
"""CLI for generating workflow specifications."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

from dynamic_path_router import resolve_path

from workflow_synthesizer import WorkflowSynthesizer
from workflow_spec import to_spec, save


def _interactive_edit(modules: List[str]) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    for mod in modules:
        prompt = f"Include step '{mod}'? [Y/n/edit]: "
        choice = input(prompt).strip()
        if not choice or choice.lower() in {"y", "yes"}:
            steps.append({"name": mod})
        elif choice.lower() in {"n", "no"}:
            continue
        else:
            steps.append({"name": choice})
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a workflow and interactively save a specification."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Starting module name or free text problem description",
    )
    parser.add_argument("--out", required=True, help="Output file for workflow spec")
    args = parser.parse_args()

    synth = WorkflowSynthesizer()

    start = args.start
    try:
        resolve_path(start.replace(".", "/") + ".py")
    except FileNotFoundError:
        result = synth.synthesize(start)
        modules = [step.get("module", "") for step in result.get("steps", [])]
    else:
        workflows = synth.generate_workflows(start_module=start)
        if workflows:
            modules = [step.get("module", "") for step in workflows[0] if step.get("module")]
        else:
            modules = [start]

    if not modules:
        print("No modules found")
        return

    steps = _interactive_edit(modules)
    spec = to_spec(steps)
    path = save(spec, Path(args.out))
    print(f"Saved workflow to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
