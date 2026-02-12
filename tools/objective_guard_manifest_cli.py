from __future__ import annotations

"""Manual operator utility for objective hash-lock manifests."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
from objective_hash_lock import verify_objective_hash_lock

_OBJECTIVE_LOCK_PATH = Path("config/objective_integrity_lock.json")


def _guard(repo_root: Path) -> ObjectiveGuard:
    return ObjectiveGuard(repo_root=repo_root)


def _read_manifest_sha(guard: ObjectiveGuard) -> str | None:
    if not guard.manifest_path.exists():
        return None
    try:
        payload = json.loads(guard.manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("manifest_sha256"), str):
        return str(payload["manifest_sha256"])
    return None


def _lock_path(repo_root: Path) -> Path:
    return (repo_root / _OBJECTIVE_LOCK_PATH).resolve()


def _reset_integrity_lock(
    *,
    repo_root: Path,
    guard: ObjectiveGuard,
    operator: str,
    reason: str,
) -> int:
    lock_path = _lock_path(repo_root)
    if not lock_path.exists():
        print(f"objective integrity lock not present: {lock_path}")
        return 0

    try:
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to read objective integrity lock: {exc}")
        return 1
    if not isinstance(lock_payload, dict):
        print("failed to read objective integrity lock: invalid payload")
        return 1

    manifest_sha_before = lock_payload.get("manifest_sha_at_breach")
    if manifest_sha_before is not None:
        manifest_sha_before = str(manifest_sha_before)

    try:
        verify_objective_hash_lock(guard=guard)
    except ObjectiveGuardViolation as exc:
        print(f"objective lock reset blocked: manifest verification failed {exc.reason} details={exc.details}")
        return 1

    manifest_sha_now = _read_manifest_sha(guard)
    if manifest_sha_before and manifest_sha_now == manifest_sha_before:
        print("objective lock reset blocked: rotate objective hash baseline first")
        return 1

    lock_payload["locked"] = False
    lock_payload["reset"] = {
        "who": operator,
        "why": reason,
        "when": datetime.now(timezone.utc).isoformat(),
        "manifest_sha_before": manifest_sha_before,
        "manifest_sha_after": manifest_sha_now,
    }
    history_path = lock_path.with_suffix(".resets.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.open("a", encoding="utf-8").write(json.dumps(lock_payload, sort_keys=True) + "\n")
    lock_path.unlink()
    print(f"objective integrity lock reset: {history_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for protected objective paths.",
    )
    parser.add_argument("--operator", help="Operator identity (who approved).")
    parser.add_argument("--reason", help="Operator rationale (why/what changed).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "refresh",
        help="Manually regenerate the persisted objective hash-lock manifest (no automatic updates).",
    )
    subparsers.add_parser(
        "rotate",
        help="Rotate objective hash-lock baseline after approved objective surface change.",
    )
    subparsers.add_parser(
        "approve-change",
        help="Operator approval workflow for intentional objective file change (alias for rotate).",
    )
    subparsers.add_parser(
        "update",
        help="Deprecated alias for refresh.",
    )
    subparsers.add_parser(
        "verify",
        help="Verify objective hashes against the persisted manifest.",
    )
    subparsers.add_parser(
        "reset-lock",
        help="Safely clear objective integrity lock after manifest rotation and operator review.",
    )

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    guard = _guard(repo_root)

    if args.command in {"refresh", "update", "rotate", "approve-change"}:
        operator = (args.operator or "").strip()
        reason = (args.reason or "").strip()
        if not operator:
            print("manifest refresh failed: operator is required")
            return 1
        if not reason:
            print("manifest refresh failed: reason is required")
            return 1
        try:
            hashes = guard.write_manifest(
                operator=operator,
                reason=reason,
                rotation=args.command in {"rotate", "approve-change"},
                command_source="tools/objective_guard_manifest_cli.py",
            )
        except ObjectiveGuardViolation as exc:
            print(f"manifest refresh failed: {exc.reason} details={exc.details}")
            return 1
        print(
            f"updated manifest: {guard.manifest_path} ({len(hashes)} protected file hashes)"
        )
        return 0

    if args.command == "reset-lock":
        operator = (args.operator or "").strip()
        reason = (args.reason or "").strip()
        if not operator:
            print("objective lock reset failed: operator is required")
            return 1
        if not reason:
            print("objective lock reset failed: reason is required")
            return 1
        return _reset_integrity_lock(
            repo_root=repo_root,
            guard=guard,
            operator=operator,
            reason=reason,
        )

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
