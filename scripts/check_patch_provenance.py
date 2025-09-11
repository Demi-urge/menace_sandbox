#!/usr/bin/env python3
"""Enforce patch provenance for commits touching coding bot files.

When installed as a ``commit-msg`` hook, the script inspects staged files and
rejects commits that modify self-coding components unless the commit message
contains a ``patch <id>`` tag and provenance metadata provided via the
``PATCH_PROVENANCE_FILE`` environment variable.

The script also supports a ``--ci`` mode for verifying an existing commit in
continuous integration. In this mode the hook checks ``HEAD`` (or a supplied
commit hash) and flags missing tags by recording an ``untracked_commit`` event
via :mod:`mutation_logger` so the :class:`EvolutionOrchestrator` can trigger a
rollback.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Patterns identifying files maintained by the self-coding system.
CODING_BOT_PATTERNS = ["self_coding_", "coding_bot", "quick_fix_engine"]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _changed_files(commit: str | None = None) -> list[str]:
    if commit:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", f"{commit}^!"], text=True
        )
    else:
        diff = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], text=True
        )
    return [f.strip() for f in diff.splitlines() if f.strip()]


def _touches_coding_bot(files: list[str]) -> bool:
    return any(any(pat in f for pat in CODING_BOT_PATTERNS) for f in files)


def _extract_patch_id(message: str) -> str | None:
    m = re.search(r"patch\s+(\d+)", message, re.IGNORECASE)
    return m.group(1) if m else None


def _verify_provenance(patch_id: str) -> bool:
    prov_path = os.environ.get("PATCH_PROVENANCE_FILE")
    if not prov_path:
        print("PATCH_PROVENANCE_FILE not set", file=sys.stderr)
        return False
    path = Path(prov_path)
    try:
        data = json.loads(path.read_text())
    except Exception:
        print("failed to read provenance file", file=sys.stderr)
        return False
    if str(data.get("patch_id")) != str(patch_id):
        print("patch_id mismatch in provenance data", file=sys.stderr)
        return False
    return True


def _ci_check(commit: str) -> int:
    files = _changed_files(commit)
    if not _touches_coding_bot(files):
        return 0
    msg = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%B", commit], text=True
    )
    patch_id = _extract_patch_id(msg)
    if patch_id:
        return 0
    # Flag missing provenance so the orchestrator can rollback.
    try:
        from mutation_logger import flag_untracked_commit

        commit_hash = subprocess.check_output(
            ["git", "rev-parse", commit], text=True
        ).strip()
        flag_untracked_commit(commit_hash)
    except Exception:  # pragma: no cover - best effort
        pass
    print("coding-bot commit lacks patch id", file=sys.stderr)
    return 1


def _hook_check(msg_file: Path) -> int:
    files = _changed_files()
    if not _touches_coding_bot(files):
        return 0
    message = msg_file.read_text()
    patch_id = _extract_patch_id(message)
    if not patch_id:
        print("commit message must include 'patch <id>'", file=sys.stderr)
        return 1
    if not _verify_provenance(patch_id):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("msg_file", nargs="?")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args(argv)

    if args.ci:
        return _ci_check(args.commit)
    if not args.msg_file:
        parser.error("missing commit message file")
    return _hook_check(Path(args.msg_file))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
