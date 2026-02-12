from __future__ import annotations

"""Manual operator utility for objective hash-lock manifests."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation


def _guard(repo_root: Path) -> ObjectiveGuard:
    return ObjectiveGuard(repo_root=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for protected objective paths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "refresh",
        help="Manually regenerate the persisted objective hash-lock manifest (no automatic updates).",
    )
    subparsers.add_parser(
        "update",
        help="Deprecated alias for refresh.",
    )
    subparsers.add_parser(
        "verify",
        help="Verify objective hashes against the persisted manifest.",
    )

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    guard = _guard(repo_root)

    if args.command in {"refresh", "update"}:
        hashes = guard.write_manifest()
        print(
            f"updated manifest: {guard.manifest_path} ({len(hashes)} protected file hashes)"
        )
        return 0

    try:
        report = guard.verify_manifest()
    except ObjectiveGuardViolation as exc:
        print(f"manifest verification failed: {exc.reason} details={exc.details}")
        return 1

    print(
        f"manifest verification ok: {report['manifest_path']} files={len(report['files'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
