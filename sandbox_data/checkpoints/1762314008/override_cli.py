#!/usr/bin/env python3
"""Utility for generating and applying manual override files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from override_validator import generate_signature, validate_override_file


def _load_json(value: str) -> dict:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual override file helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate signed override file")
    gen.add_argument("data", help="JSON string or path to JSON file")
    gen.add_argument("private_key", help="path to private key for signing")
    gen.add_argument("output", help="where to write the signed override file")

    ap = sub.add_parser("apply", help="validate and apply override file")
    ap.add_argument("override_path", help="path to override file")
    ap.add_argument("public_key", help="path to public key for validation")

    args = parser.parse_args()

    if args.cmd == "generate":
        data = _load_json(args.data)
        sig = generate_signature(data, args.private_key)
        payload = {"data": data, "signature": sig}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(json.dumps({"written": str(args.output)}))
    elif args.cmd == "apply":
        valid, data = validate_override_file(args.override_path, args.public_key)
        print(json.dumps({"valid": valid, "data": data}))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
