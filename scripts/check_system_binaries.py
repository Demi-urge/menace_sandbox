#!/usr/bin/env python3
from __future__ import annotations

"""Verify required system binaries exist before running the sandbox."""

import sys

from dynamic_path_router import resolve_path
from system_binary_check import assert_required_system_binaries, DEFAULT_REQUIREMENTS


def main() -> int:
    # Ensure repository-local imports work when the script is invoked via PATH
    repo_root = resolve_path("")
    sys.path.insert(0, str(repo_root))

    try:
        missing = assert_required_system_binaries(DEFAULT_REQUIREMENTS, exit_on_missing=False)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[!] Unexpected error while checking system binaries: {exc}")
        return 1

    if missing:
        print("[!] Missing required system binaries for start_autonomous_sandbox.py:")
        for name in missing:
            print(f"    - {name}: {DEFAULT_REQUIREMENTS[name]}")
        print("[!] Install the tools above and re-run the sandbox starter.")
        return 1

    print("[✓] All required system binaries found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
