"""Top-level CLI for common Menace workflows."""

import argparse
import json
import subprocess
import sys


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    return subprocess.call(cmd)


from code_database import PatchHistoryDB
from patch_provenance import (
    build_chain,
    search_patches_by_vector,
    search_patches_by_license,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Menace workflow helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Install dependencies and bootstrap the env")

    p_test = sub.add_parser("test", help="Run the test suite")
    p_test.add_argument("pytest_args", nargs=argparse.REMAINDER)

    p_improve = sub.add_parser("improve", help="Run a single self-improvement cycle")
    p_improve.add_argument("extra_args", nargs=argparse.REMAINDER)

    sub.add_parser("benchmark", help="Benchmark registered workflows")

    p_deploy = sub.add_parser("deploy", help="Install Menace as a service")
    p_deploy.add_argument("extra_args", nargs=argparse.REMAINDER)

    p_sandbox = sub.add_parser("sandbox", help="Sandbox utilities")
    sb_sub = p_sandbox.add_subparsers(dest="sandbox_cmd", required=True)
    sb_run = sb_sub.add_parser("run", help="Run the autonomous sandbox")
    sb_run.add_argument("extra_args", nargs=argparse.REMAINDER)

    p_patch = sub.add_parser("patches", help="Patch provenance helpers")
    patch_sub = p_patch.add_subparsers(dest="patches_cmd", required=True)

    p_list = patch_sub.add_parser("list", help="List recent patches")
    p_list.add_argument("--limit", type=int, default=20)

    p_chain = patch_sub.add_parser("ancestry", help="Show patch ancestry chain")
    p_chain.add_argument("patch_id", type=int)

    p_search = patch_sub.add_parser("search", help="Search patches by vector or license")
    grp = p_search.add_mutually_exclusive_group(required=True)
    grp.add_argument("--vector", help="Vector identifier")
    grp.add_argument("--license", help="License filter")

    args = parser.parse_args(argv)

    if args.cmd == "setup":
        return _run(["bash", "scripts/setup_autonomous.sh"])

    if args.cmd == "test":
        return _run(["pytest"] + (args.pytest_args or []))

    if args.cmd == "improve":
        return _run(["python", "run_autonomous.py", "--runs", "1"] + (args.extra_args or []))

    if args.cmd == "benchmark":
        return _run(["python", "workflow_benchmark.py"])

    if args.cmd == "deploy":
        return _run(["python", "service_installer.py"] + (args.extra_args or []))

    if args.cmd == "sandbox":
        if args.sandbox_cmd == "run":
            return _run(["python", "run_autonomous.py"] + (args.extra_args or []))

    if args.cmd == "patches":
        db = PatchHistoryDB()
        if args.patches_cmd == "list":
            rows = db.list_patches(args.limit)
            patches = [
                {"id": pid, "filename": rec.filename, "description": rec.description}
                for pid, rec in rows
            ]
            print(json.dumps(patches))
            return 0
        if args.patches_cmd == "ancestry":
            chain = build_chain(args.patch_id, patch_db=db)
            print(json.dumps(chain))
            return 0
        if args.patches_cmd == "search":
            if args.vector:
                rows = search_patches_by_vector(args.vector, patch_db=db)
                patches = [
                    {
                        "id": r["patch_id"],
                        "filename": r["filename"],
                        "description": r["description"],
                        "influence": r["influence"],
                    }
                    for r in rows
                ]
            else:
                rows = search_patches_by_license(args.license, patch_db=db)
                patches = [
                    {
                        "id": r["patch_id"],
                        "filename": r["filename"],
                        "description": r["description"],
                    }
                    for r in rows
                ]
            print(json.dumps(patches))
            return 0

    parser.error("unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
