from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

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
        description="Synthesize a workflow and interactively save a specification.")
    parser.add_argument("--module", dest="module", help="Starting module for synthesis")
    parser.add_argument("--problem", dest="problem", help="Problem description for intent search")
    parser.add_argument("--out", dest="out", required=True, help="Output file for workflow spec")
    args = parser.parse_args()

    synth = WorkflowSynthesizer()

    if args.module:
        workflows = synth.generate_workflows(start_module=args.module, problem=args.problem)
        if workflows:
            modules = [step.get("module", "") for step in workflows[0] if step.get("module")]
        else:
            modules = [args.module]
    else:
        modules = synth.synthesize(start_module=None, problem=args.problem)

    if not modules:
        print("No modules found")
        return

    steps = _interactive_edit(modules)
    spec = to_spec(steps)
    path = save(spec, Path(args.out))
    print(f"Saved workflow to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
