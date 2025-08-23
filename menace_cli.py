"""Top-level CLI for common Menace workflows."""

import argparse
import json
import subprocess
import sys
import uuid

from menace.plugins import load_plugins
from db_router import init_db_router


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    return subprocess.call(cmd)


from code_database import PatchHistoryDB
from patch_provenance import (
    PatchLogger,
    build_chain,
    search_patches_by_vector,
    search_patches_by_license,
)
from cache_utils import get_cached_chain, set_cached_chain, _get_cache
from cache_utils import clear_cache, show_cache, cache_stats

# Expose a DBRouter for CLI operations
DB_ROUTER = init_db_router(uuid.uuid4().hex)


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


# ---------------------------------------------------------------------------
# Subcommand handlers


def handle_new_db(args: argparse.Namespace) -> int:
    """Handle ``new-db`` command."""
    return _run([sys.executable, "scripts/new_db_template.py", args.name])


def handle_new_vector(args: argparse.Namespace) -> int:
    """Handle ``new-vector`` command."""
    cmd = [
        sys.executable,
        "scripts/new_vector_module.py",
        args.name,
    ]
    if args.root:
        cmd += ["--root", args.root]
    if args.register_router:
        cmd.append("--register-router")
    if args.create_migration:
        cmd.append("--create-migration")
    return _run(cmd)


def handle_patch(args: argparse.Namespace) -> int:
    """Handle ``patch`` command."""
    from vector_service import ContextBuilder
    from vector_service.retriever import Retriever
    import quick_fix_engine

    retriever = Retriever(cache=_get_cache())

    ctx = None
    if args.context:
        try:
            ctx = json.loads(args.context)
        except json.JSONDecodeError:
            print("invalid JSON context", file=sys.stderr)
            return 1

    db = PatchHistoryDB()
    patch_logger = PatchLogger(patch_db=db)
    patch_id = quick_fix_engine.generate_patch(
        args.module,
        context_builder=ContextBuilder(retriever=retriever),
        engine=None,
        description=args.desc,
        patch_logger=patch_logger,
        context=ctx,
    )
    if not patch_id:
        return 1
    record = db.get(patch_id)
    files = [record.filename] if record else []
    print(json.dumps({"patch_id": patch_id, "files": files}))
    return 0


def handle_retrieve(args: argparse.Namespace) -> int:
    """Handle ``retrieve`` command."""
    try:
        from vector_service.retriever import Retriever, FallbackResult, fts_search
    except Exception:
        print("vector retriever unavailable", file=sys.stderr)
        return 1

    retriever = Retriever(cache=None if args.no_cache else _get_cache())

    if not args.no_cache and not args.rebuild_cache:
        cached = get_cached_chain(args.query, args.dbs)
        if cached is not None:
            print(json.dumps(cached))
            return 0
    try:
        hits = retriever.search(args.query, top_k=args.top_k, dbs=args.dbs)
    except Exception:
        hits = []
    if isinstance(hits, FallbackResult):
        hits = list(hits)
    results = _normalise_hits(hits)
    if not results:
        try:
            extra = fts_search(args.query, dbs=args.dbs, limit=args.top_k)
        except Exception:
            extra = []
        results = _normalise_hits(extra)

    if results and not args.no_cache:
        set_cached_chain(args.query, args.dbs, results)
    if args.json:
        print(json.dumps(results))
    else:
        if not results:
            print("No results")
        else:
            headers = ["origin_db", "record_id", "score", "snippet"]
            origin_w = max(
                [len(headers[0])] + [len(str(r["origin_db"])) for r in results]
            )
            record_w = max(
                [len(headers[1])] + [len(str(r["record_id"])) for r in results]
            )
            score_values = [f"{r['score']:.4f}" for r in results]
            score_w = max([len(headers[2])] + [len(s) for s in score_values])
            header_line = f"{headers[0]:<{origin_w}}  {headers[1]:<{record_w}}  {headers[2]:>{score_w}}  {headers[3]}"
            print(header_line)
            for r in results:
                snippet = str(r["snippet"]).replace("\n", " ")
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                line = f"{r['origin_db']:<{origin_w}}  {str(r['record_id']):<{record_w}}  {r['score']:{score_w}.4f}  {snippet}"
                print(line)
    return 0


def handle_embed(args: argparse.Namespace) -> int:
    """Handle ``embed`` command."""
    import logging
    from vector_service.embedding_backfill import EmbeddingBackfill
    from vector_service.exceptions import VectorServiceError

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        EmbeddingBackfill().run(
            session_id="cli",
            dbs=args.dbs,
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


def handle_cache_show(args: argparse.Namespace) -> int:
    """Handle ``cache show`` command."""
    print(json.dumps(show_cache()))
    return 0


def handle_cache_clear(args: argparse.Namespace) -> int:
    """Handle ``cache clear`` command."""
    clear_cache()
    return 0


def handle_cache_stats(args: argparse.Namespace) -> int:
    """Handle ``cache stats`` command."""
    print(json.dumps(cache_stats()))
    return 0


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
    p_quick.set_defaults(func=handle_patch)

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

    p_cache = sub.add_parser("cache", help="Manage retrieval cache")
    cache_sub = p_cache.add_subparsers(dest="cache_cmd", required=True)
    cache_sub.add_parser("show", help="Show all cached entries").set_defaults(
        func=handle_cache_show
    )
    cache_sub.add_parser("clear", help="Clear the retrieval cache").set_defaults(
        func=handle_cache_clear
    )
    cache_sub.add_parser("stats", help="Display cache statistics").set_defaults(
        func=handle_cache_stats
    )

    p_retrieve = sub.add_parser(
        "retrieve", help="Semantic code retrieval (text table by default)"
    )
    p_retrieve.add_argument("query")
    p_retrieve.add_argument("--db", action="append", dest="dbs")
    p_retrieve.add_argument("--top-k", type=int, dest="top_k", default=5)
    p_retrieve.add_argument(
        "--json", action="store_true", help="Output JSON results instead of text"
    )
    p_retrieve.add_argument("--no-cache", action="store_true", help="Bypass retrieval cache")
    p_retrieve.add_argument(
        "--rebuild-cache", action="store_true", help="Force recomputation of cache"
    )
    p_retrieve.set_defaults(func=handle_retrieve)

    p_embed = sub.add_parser("embed", help="Backfill vector embeddings")
    p_embed.add_argument(
        "--db",
        action="append",
        dest="dbs",
        help="Restrict to a specific database class (can be used multiple times)",
    )
    p_embed.add_argument(
        "--batch-size", type=int, dest="batch_size", help="Batch size for backfill"
    )
    p_embed.add_argument(
        "--backend", dest="backend", help="Vector backend to use"
    )
    p_embed.set_defaults(func=handle_embed)

    p_newdb = sub.add_parser("new-db", help="Scaffold a new database module")
    p_newdb.add_argument("name", help="Base name for the new database")
    p_newdb.set_defaults(func=handle_new_db)

    p_newvec = sub.add_parser(
        "new-vector", help="Scaffold a new vector_service module"
    )
    p_newvec.add_argument("name", help="Base name for the module")
    p_newvec.add_argument("--root", help="Target directory", default=None)
    p_newvec.add_argument(
        "--register-router", action="store_true", help="Update database_router"
    )
    p_newvec.add_argument(
        "--create-migration", action="store_true", help="Create alembic migration"
    )
    p_newvec.set_defaults(func=handle_new_vector)
    # allow plugins to register additional subcommands
    load_plugins(sub)

    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args)

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
