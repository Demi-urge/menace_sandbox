#!/usr/bin/env python3
"""Recompute and validate the hash chain for an audit log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _compute_hash(prev_hash: str, record: dict) -> str:
    data = json.dumps(record, sort_keys=True)
    return hashlib.sha256((prev_hash + data).encode()).hexdigest()


def verify(log_path: Path, state_path: Path) -> bool:
    prev_hash = "0" * 64
    try:
        lines = log_path.read_text().splitlines()
    except OSError:
        raise SystemExit(f"unable to read log: {log_path}")

    for idx, line in enumerate(lines, 1):
        rec = json.loads(line)
        rec_hash = rec.get("hash")
        data = {k: v for k, v in rec.items() if k != "hash"}
        expected = _compute_hash(prev_hash, data)
        if rec_hash != expected:
            print(json.dumps({"valid": False, "line": idx}))
            return False
        prev_hash = rec_hash

    if state_path.exists():
        if state_path.read_text().strip() != prev_hash:
            print(json.dumps({"valid": False, "line": len(lines), "reason": "state_mismatch"}))
            return False

    print(json.dumps({"valid": True}))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify audit log chain")
    parser.add_argument("log_path", type=Path, help="path to audit log file")
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="optional path to state file (defaults to <log_path>.state)",
    )
    args = parser.parse_args()
    state_path = args.state_path or Path(f"{args.log_path}.state")
    verify(args.log_path, state_path)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
