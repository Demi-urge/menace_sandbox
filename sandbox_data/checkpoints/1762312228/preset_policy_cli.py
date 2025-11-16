from __future__ import annotations

"""CLI for exporting or importing the preset RL policy."""

import argparse
import json
import sys

from typing import Any, Dict

from environment_generator import (
    export_preset_policy,
    import_preset_policy,
)


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


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export", help="Export current policy to JSON")
    exp.add_argument("--out", default="-", help="Output file or '-' for stdout")

    imp = sub.add_parser("import", help="Import policy from JSON")
    imp.add_argument("file", help="JSON file to read")

    args = parser.parse_args(argv)

    if args.cmd == "export":
        policy = export_preset_policy()
        data = json.dumps(_serialise(policy), indent=2)
        if args.out == "-":
            sys.stdout.write(data)
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(data)
        return 0

    if args.cmd == "import":
        with open(args.file, encoding="utf-8") as fh:
            data = json.load(fh)
        import_preset_policy(_deserialise(data))
        return 0

    parser.error("unknown command")
    return 1


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
