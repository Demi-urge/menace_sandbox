"""Top-level CLI for common Menace workflows."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    return subprocess.call(cmd)


from code_database import PatchHistoryDB
from patch_provenance import (
    build_chain,
    search_patches_by_vector,
    search_patches_by_license,
    get_patch_provenance,
)
from vector_service.retriever import (
    Retriever,
    FallbackResult,
    VectorServiceError,
    fts_search,
)


def _normalise_hits(hits, origin=None):
    norm = []
    for h in hits:
        if isinstance(h, dict):
            origin_db = h.get("origin_db", origin or "")
            record_id = h.get("record_id") or h.get("id") or h.get("key")
            snippet = (
                h.get("text")
                or h.get("snippet")
                or h.get("code")
                or h.get("data")
                or h.get("summary")
                or ""
            )
            norm.append(
                {
                    "origin_db": origin_db,
                    "record_id": record_id,
                    "score": h.get("score", 0.0),
                    "snippet": snippet,
                }
            )
        else:
            origin_db = getattr(h, "origin_db", origin or "")
            record_id = (
                getattr(h, "record_id", None)
                or getattr(h, "id", None)
                or getattr(h, "key", None)
            )
            snippet = (
                getattr(h, "text", None)
                or getattr(h, "snippet", None)
                or getattr(h, "code", None)
                or getattr(h, "data", None)
                or getattr(h, "summary", "")
            )
            score = getattr(h, "score", 0.0)
            norm.append(
                {
                    "origin_db": origin_db,
                    "record_id": record_id,
                    "score": score,
                    "snippet": snippet,
                }
            )
    return norm


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

    p_quick = sub.add_parser("patch", help="Apply a patch to a module")
    p_quick.add_argument("module")
    p_quick.add_argument("--desc", required=True, help="Patch description")
    p_quick.add_argument("--context", help="JSON encoded context", default=None)

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

    p_retrieve = sub.add_parser("retrieve", help="Semantic code retrieval")
    p_retrieve.add_argument("query")
    p_retrieve.add_argument("--db", action="append", dest="dbs")
    p_retrieve.add_argument("--top-k", type=int, dest="top_k", default=5)
    p_retrieve.add_argument("--json", action="store_true", help="Output JSON results")
    p_retrieve.add_argument("--no-cache", action="store_true", help="Bypass retrieval cache")

    p_embed = sub.add_parser("embed", help="Backfill vector embeddings")
    p_embed.add_argument("--db", help="Restrict to a specific database class")
    p_embed.add_argument(
        "--batch-size", type=int, dest="batch_size", help="Batch size for backfill"
    )
    p_embed.add_argument(
        "--backend", dest="backend", help="Vector backend to use"
    )

    p_newdb = sub.add_parser("new-db", help="Scaffold a new database module")
    p_newdb.add_argument("name", help="Base name for the new database")

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

    if args.cmd == "new-db":
        return _run([sys.executable, "scripts/scaffold_db.py", args.name])

    if args.cmd == "sandbox":
        if args.sandbox_cmd == "run":
            return _run(["python", "run_autonomous.py"] + (args.extra_args or []))

    if args.cmd == "patch":
        from vector_service import ContextBuilder
        import quick_fix_engine

        ctx = None
        if args.context:
            try:
                ctx = json.loads(args.context)
            except json.JSONDecodeError:
                print("invalid JSON context", file=sys.stderr)
                return 1

        patch_id = quick_fix_engine.generate_patch(
            args.module,
            context_builder=ContextBuilder(),
            engine=None,
            description=args.desc,
            context=ctx,
        )
        if not patch_id:
            return 1
        provenance = get_patch_provenance(patch_id)
        print(json.dumps({"patch_id": patch_id, "provenance": provenance}))
        return 0

    if args.cmd == "retrieve":
        cache_path = Path(".retriever_cache.json")
        retriever = Retriever(
            cache_path=None if args.no_cache else str(cache_path)
        )
        try:
            res = retriever.search(
                args.query,
                session_id=uuid4().hex,
                top_k=args.top_k,
                dbs=args.dbs,
            )
        except VectorServiceError as exc:
            res = FallbackResult(str(exc), [])
        if isinstance(res, FallbackResult):
            results = _normalise_hits(list(res))
            if res.reason == "no results":
                try:
                    extra = fts_search(
                        args.query, dbs=args.dbs, limit=args.top_k
                    )
                except Exception:
                    extra = []
                seen = {(r["origin_db"], r["record_id"]) for r in results}
                for hit in _normalise_hits(extra):
                    key = (hit["origin_db"], hit["record_id"])
                    if key not in seen:
                        results.append(hit)
                        seen.add(key)
        else:
            results = _normalise_hits(res)
        if results and not args.no_cache:
            retriever.save_cache()
        print(json.dumps(results))
        return 0

    if args.cmd == "embed":
        import logging
        from vector_service.embedding_backfill import EmbeddingBackfill
        from vector_service.exceptions import VectorServiceError

        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        try:
            EmbeddingBackfill().run(
                session_id="cli",
                db=args.db,
                batch_size=args.batch_size,
                backend=args.backend,
            )
        except VectorServiceError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:  # pragma: no cover - defensive
            print(str(exc), file=sys.stderr)
            return 1
        return 0

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
