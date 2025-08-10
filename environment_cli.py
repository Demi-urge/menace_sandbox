from __future__ import annotations

"""Command line interface for environment preset utilities."""

import argparse
import json
import sys
from typing import Any, Dict, List

from environment_generator import (
    generate_presets,
    adapt_presets,
    export_preset_policy,
    import_preset_policy,
)
try:
    from menace.roi_tracker import ROITracker as _ROITracker
except Exception:  # pragma: no cover - optional dependency or package stub
    _ROITracker = None


def _serialise(policy: Dict[tuple[int, ...], Dict[int, float]]) -> Dict[str, Dict[str, float]]:
    """Convert policy table to a JSON compatible structure."""
    return {
        ",".join(map(str, state)): {str(a): q for a, q in actions.items()}
        for state, actions in policy.items()
    }


def _deserialise(data: Dict[str, Dict[str, Any]]) -> Dict[tuple[int, ...], Dict[int, float]]:
    """Convert JSON structure back to policy table."""
    return {
        tuple(int(x) for x in key.split(",")): {int(a): float(q) for a, q in val.items()}
        for key, val in data.items()
    }


def cli(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate environment presets")
    p_gen.add_argument("--count", type=int, default=None, help="Number of presets")
    p_gen.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Named profiles to include in generation",
    )
    p_gen.add_argument("--out", default="-", help="Output file or '-' for stdout")

    p_adapt = sub.add_parser("adapt", help="Adapt presets using ROI history")
    p_adapt.add_argument("file", help="Preset JSON file")
    p_adapt.add_argument("--history", help="ROI history file (optional)")
    p_adapt.add_argument("--out", default="-", help="Output file or '-' for stdout")

    p_exp = sub.add_parser("export-policy", help="Export reinforcement policy")
    p_exp.add_argument("--out", default="-", help="Output file or '-' for stdout")

    p_imp = sub.add_parser("import-policy", help="Import reinforcement policy")
    p_imp.add_argument("file", help="JSON file with policy")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        presets = generate_presets(args.count, profiles=args.profiles)
        data = json.dumps(presets, indent=2)
        if args.out == "-":
            sys.stdout.write(data)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(data)
        return 0

    if args.cmd == "adapt":
        with open(args.file, encoding="utf-8") as fh:
            presets = json.load(fh)
        if _ROITracker and args.history:
            tracker = _ROITracker()
            tracker.load_history(args.history)
        else:
            class _Tracker:
                roi_history: list[float] = []
                metrics_history: dict[str, list[float]] = {"security_score": []}

                def diminishing(self) -> float:
                    return 0.01

                def predict_synergy_metric(self, name: str) -> float:
                    return 0.0

            tracker = _Tracker()
        new_presets = adapt_presets(tracker, presets)
        data = json.dumps(new_presets, indent=2)
        if args.out == "-":
            sys.stdout.write(data)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(data)
        return 0

    if args.cmd == "export-policy":
        policy = export_preset_policy()
        data = json.dumps(_serialise(policy), indent=2)
        if args.out == "-":
            sys.stdout.write(data)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(data)
        return 0

    if args.cmd == "import-policy":
        with open(args.file, encoding="utf-8") as fh:
            data = json.load(fh)
        import_preset_policy(_deserialise(data))
        return 0

    parser.error("unknown command")
    return 1


def main(argv: List[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
