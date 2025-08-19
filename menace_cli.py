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


from code_database import PatchHistoryDB, CodeDB
from patch_provenance import (
    build_chain,
    search_patches_by_vector,
    search_patches_by_license,
)
from vector_service import Retriever, FallbackResult, VectorServiceError
from retrieval_cache import RetrievalCache, get_db_mtimes


def _search_memory(q: str):
    from menace_memory_manager import MenaceMemoryManager

    return MenaceMemoryManager().search(q)


FTS_HELPERS = {
    "code": lambda q: CodeDB().search_fts(q),
    "memory": _search_memory,
}


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

    p_embed = sub.add_parser("embed", help="Backfill vector embeddings")
    p_embed.add_argument("--db", help="Restrict to a specific database class")

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
        return _run([sys.executable, "scripts/new_db.py", args.name])

    if args.cmd == "sandbox":
        if args.sandbox_cmd == "run":
            return _run(["python", "run_autonomous.py"] + (args.extra_args or []))

    if args.cmd == "patch":
        from vector_service import ContextBuilder
        from patch_provenance import get_patch_provenance
        try:
            from self_coding_engine import SelfCodingEngine
            from menace_memory_manager import MenaceMemoryManager
        except Exception as exc:  # pragma: no cover - optional deps
            print(f"self coding engine unavailable: {exc}", file=sys.stderr)
            return 1
        path = Path(args.module)
        if path.suffix == "":
            path = path.with_suffix(".py")
        if not path.exists():
            print(f"module not found: {args.module}", file=sys.stderr)
            return 1
        builder = ContextBuilder()
        import uuid
        cb_session = uuid.uuid4().hex
        ctx_block = ""
        vectors = []
        try:
            ctx_block, _, vectors = builder.build(
                args.desc, session_id=cb_session, include_vectors=True
            )
        except Exception:
            ctx_block = ""
            vectors = []
        desc = args.desc
        if ctx_block:
            desc = f"{desc}\n\n{ctx_block}"
        context_meta = {
            "module": str(path),
            "retrieval_session_id": cb_session,
            "retrieval_vectors": [
                {"origin": o, "vector_id": vid, "score": score}
                for o, vid, score in vectors
            ],
        }
        try:
            engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=builder
            )
            patch_id, _, _ = engine.apply_patch(
                path,
                desc,
                reason="cli_patch",
                trigger="menace_cli_patch",
                context_meta=context_meta,
            )
        except Exception as exc:  # pragma: no cover - runtime issues
            print(f"patch failed: {exc}", file=sys.stderr)
            return 1
        if not patch_id:
            return 1
        print(patch_id)
        prov = get_patch_provenance(patch_id)
        for item in prov:
            print(json.dumps(item))
        return 0

    if args.cmd == "retrieve":
        cache = RetrievalCache()
        db_list = args.dbs or []
        mtimes = get_db_mtimes(db_list)
        cached = cache.get(args.query, db_list, mtimes)
        if cached is not None:
            print(json.dumps(cached))
            return 0
        retriever = Retriever()
        try:
            res = retriever.search(args.query, session_id=uuid4().hex, dbs=args.dbs)
        except VectorServiceError as exc:
            res = FallbackResult(str(exc), [])
        if isinstance(res, FallbackResult):
            results = _normalise_hits(list(res))
            for db_name in (args.dbs or ["code"]):
                helper = FTS_HELPERS.get(db_name)
                if helper:
                    try:
                        results.extend(
                            _normalise_hits(helper(args.query), origin=db_name)
                        )
                    except Exception:
                        pass
        else:
            results = _normalise_hits(res)
        if results:
            cache.set(args.query, db_list, results, get_db_mtimes(db_list))
        print(json.dumps(results))
        return 0

    if args.cmd == "embed":
        import logging
        from vector_service import EmbeddingBackfill, VectorServiceError

        logging.basicConfig(level=logging.INFO)
        try:
            EmbeddingBackfill().run(session_id="cli", db=args.db)
        except VectorServiceError as exc:
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
