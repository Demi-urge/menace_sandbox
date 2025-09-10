"""Command line utilities for interacting with Menace services.

The CLI initialises :data:`GLOBAL_ROUTER` via :func:`init_db_router` before
performing any database work.
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from dynamic_path_router import resolve_path
from db_router import init_db_router
from context_builder_util import create_context_builder

# Expose a DBRouter for CLI operations early so imported modules can rely on
# ``GLOBAL_ROUTER``.
MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv("MENACE_LOCAL_DB_PATH")
SHARED_DB_PATH = os.getenv("MENACE_SHARED_DB_PATH")
if not LOCAL_DB_PATH:
    LOCAL_DB_PATH = os.path.abspath(f"menace_{MENACE_ID}_local.db")
if not SHARED_DB_PATH:
    SHARED_DB_PATH = os.path.abspath("shared/global.db")
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)


def _run(cmd: list[str]) -> int:
    """Run a subprocess and return its exit code."""
    return subprocess.call(cmd)


from code_database import PatchHistoryDB  # noqa: E402
from cache_utils import get_cached_chain, set_cached_chain, _get_cache  # noqa: E402
from cache_utils import clear_cache, show_cache, cache_stats  # noqa: E402
from workflow_synthesizer_cli import run as handle_workflow  # noqa: E402

PatchLogger = None  # type: ignore
build_chain = None  # type: ignore
search_patches_by_vector = None  # type: ignore
search_patches_by_license = None  # type: ignore


def _ping_vector_service() -> tuple[bool, dict]:
    """Ping the vector service ``/status`` endpoint.

    Returns a tuple ``(ok, details)`` where ``ok`` indicates whether the
    service responded successfully and ``details`` contains diagnostic
    information useful for logging or automated error handling.
    """
    import urllib.request

    base = os.environ.get("VECTOR_SERVICE_URL")
    if not base:
        return True, {"detail": "VECTOR_SERVICE_URL not set"}
    url = f"{base.rstrip('/')}/status"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                payload = json.loads(body)
            except Exception:
                payload = body
            return True, {"status_code": resp.status, "body": payload}
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return False, {"error": str(exc), "url": url}


def _start_vector_service() -> tuple[bool, str]:
    """Attempt to start ``vector_database_service`` in a background process."""
    try:
        subprocess.Popen(
            [sys.executable, "-m", "vector_service.vector_database_service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, ""
    except Exception as exc:  # pragma: no cover - best effort
        return False, str(exc)


def _vector_service_available() -> tuple[bool, dict]:
    """Ensure the vector service is reachable and return diagnostics."""
    ok, info = _ping_vector_service()
    if ok:
        return True, info
    started, err = _start_vector_service()
    info["start_attempted"] = True
    if not started:
        info["start_error"] = err
        return False, info
    # Give the service a moment to start then retry
    import time

    time.sleep(1)
    ok, retry = _ping_vector_service()
    if ok:
        return True, retry
    info.update(retry)
    return False, info


def _require_vector_service() -> bool:
    """Ensure the vector service is available, logging diagnostics on failure."""
    ok, info = _vector_service_available()
    if ok:
        return True
    print(
        json.dumps({"code": "VECTOR_SERVICE_UNAVAILABLE", "diagnostics": info}),
        file=sys.stderr,
    )
    print(
        "Run `python -m vector_service.vector_database_service` to start the service.",
        file=sys.stderr,
    )
    return False


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
    script = Path("scripts/new_db_template.py")
    if not script.exists():
        script = resolve_path("scripts/new_db_template.py")
    return _run([sys.executable, str(script), args.name])


def handle_new_vector(args: argparse.Namespace) -> int:
    """Handle ``new-vector`` command."""
    script_path = str(resolve_path("scripts/new_vector_module.py"))
    cmd = [
        sys.executable,
        script_path,
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
    if not _require_vector_service():
        return 1
    from vector_service.retriever import Retriever
    import quick_fix_engine

    global PatchLogger
    if PatchLogger is None:  # pragma: no cover - lazy import
        from patch_provenance import PatchLogger as _PatchLogger

        PatchLogger = _PatchLogger

    builder = args.builder
    retriever = Retriever(context_builder=builder, cache=_get_cache())
    builder.retriever = retriever
    builder.refresh_db_weights()

    ctx = None
    if args.context:
        try:
            ctx = json.loads(args.context)
        except json.JSONDecodeError:
            print("invalid JSON context", file=sys.stderr)
            return 1

    db_kwargs = {}
    if args.db_path:
        db_kwargs["path"] = args.db_path
    try:
        db = PatchHistoryDB(**db_kwargs)
    except TypeError:  # pragma: no cover - fallback for stub implementations
        db = PatchHistoryDB()
    patch_logger = PatchLogger(patch_db=db)
    patch_id = quick_fix_engine.generate_patch(
        args.module,
        context_builder=builder,
        engine=None,
        description=args.desc,
        patch_logger=patch_logger,
        context=ctx,
        effort_estimate=args.effort_estimate,
    )
    if not patch_id:
        return 1
    record = db.get(patch_id)
    files = [record.filename] if record else []
    print(json.dumps({"patch_id": patch_id, "files": files}))
    return 0


def handle_retrieve(args: argparse.Namespace) -> int:
    """Handle ``retrieve`` command."""
    if not _require_vector_service():
        return 1
    try:
        from vector_service.retriever import Retriever, FallbackResult, fts_search
    except Exception:
        print("vector retriever unavailable", file=sys.stderr)
        return 1

    builder = args.builder
    retriever = Retriever(
        context_builder=builder, cache=None if args.no_cache else _get_cache()
    )

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
            header_line = (
                f"{headers[0]:<{origin_w}}  {headers[1]:<{record_w}}  "
                f"{headers[2]:>{score_w}}  {headers[3]}"
            )
            print(header_line)
            for r in results:
                snippet = str(r["snippet"]).replace("\n", " ")
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                line = (
                    f"{r['origin_db']:<{origin_w}}  {str(r['record_id']):<{record_w}}  "
                    f"{r['score']:{score_w}.4f}  {snippet}"
                )
                print(line)
    return 0


def handle_embed(args: argparse.Namespace) -> int:
    """Handle ``embed`` command."""
    if not _require_vector_service():
        return 1
    import logging
    from typing import IO

    from tqdm import tqdm

    from compliance.license_fingerprint import (
        check as license_check,
        fingerprint as license_fingerprint,
    )
    from vector_service.embedding_backfill import (
        EmbeddingBackfill,
        _RUN_SKIPPED,
        _log_violation,
    )
    from vector_service.exceptions import VectorServiceError

    stream: IO[str] = (
        open(args.log_file, "a", encoding="utf-8")
        if getattr(args, "log_file", None)
        else sys.stdout
    )

    class _CLIBackfill(EmbeddingBackfill):
        def __init__(self, *a, log_stream: IO[str], **kw):
            super().__init__(*a, **kw)
            self._log_stream = log_stream

        def _process_db(self, db, *, batch_size, session_id=""):
            processed = 0
            skipped: list[tuple[str, str]] = []
            records = list(db.iter_records())
            for record_id, record, kind in tqdm(
                records,
                desc=db.__class__.__name__,
                disable=not sys.stderr.isatty(),
            ):
                if processed >= batch_size:
                    break
                if not db.needs_refresh(record_id, record):
                    continue
                text = record if isinstance(record, str) else str(record)
                lic = license_check(text)
                if lic:
                    _log_violation(str(record_id), lic, license_fingerprint(text))
                    _RUN_SKIPPED.labels(db.__class__.__name__, lic).inc()
                    print(
                        f"{db.__class__.__name__}:{record_id}:license {lic}",
                        file=self._log_stream,
                    )
                    skipped.append((str(record_id), lic))
                    continue
                try:
                    db.add_embedding(record_id, record, kind)
                except Exception as exc:  # pragma: no cover - best effort
                    print(
                        f"{db.__class__.__name__}:{record_id}:embedding error {exc}",
                        file=self._log_stream,
                    )
                    continue
                processed += 1
            return skipped

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    backfill = _CLIBackfill(
        batch_size=args.batch_size if args.batch_size is not None else 100,
        backend=args.backend or "annoy",
        log_stream=stream,
    )

    try:
        out_of_sync = backfill.check_out_of_sync(dbs=args.dbs)
        if not out_of_sync:
            logging.info("no databases require re-embedding")
            return 0

        backfill.run(
            session_id="cli",
            dbs=out_of_sync,
            batch_size=args.batch_size,
            backend=args.backend,
        )
        if getattr(args, "verify", False):
            be = args.backend or backfill.backend
            subclasses = backfill._load_known_dbs(names=out_of_sync)
            for cls in subclasses:
                try:
                    db = cls(vector_backend=be)  # type: ignore[call-arg]
                except Exception:
                    try:
                        db = cls()  # type: ignore[call-arg]
                    except Exception:
                        continue
                try:
                    record_total = sum(1 for _ in db.iter_records())
                except Exception:
                    continue
                vector_total = len(getattr(db, "_id_map", []))
                if record_total != vector_total:
                    print(
                        (
                            f"WARNING: {cls.__name__} has {vector_total} "
                            f"vectors for {record_total} records"
                        ),
                        file=sys.stderr,
                    )
    except VectorServiceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if stream is not sys.stdout:
            stream.close()
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


def handle_branch_log(args: argparse.Namespace) -> int:
    """Display patch branch actions from the audit trail."""
    path = os.getenv("AUDIT_LOG_PATH", "audit.log")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    _sig, msg = line.split(" ", 1)
                    data = json.loads(msg)
                except ValueError:
                    continue
                if data.get("action") == "patch_branch":
                    print(json.dumps(data))
    except FileNotFoundError:
        print("audit trail not found", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Menace workflow helper")
    parser.add_argument("--bots-db", default="bots.db", help="Path to bots DB")
    parser.add_argument("--code-db", default="code.db", help="Path to code DB")
    parser.add_argument("--errors-db", default="errors.db", help="Path to errors DB")
    parser.add_argument(
        "--workflows-db", default="workflows.db", help="Path to workflows DB"
    )
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
    p_quick.add_argument(
        "--effort-estimate", type=float, default=None, help="Estimated effort for patch"
    )
    p_quick.add_argument("--db-path", help="Patch history database path")
    p_quick.set_defaults(func=handle_patch)

    p_patch = sub.add_parser("patches", help="Patch provenance helpers")
    patch_sub = p_patch.add_subparsers(dest="patches_cmd", required=True)

    p_list = patch_sub.add_parser("list", help="List recent patches")
    p_list.add_argument("--limit", type=int, default=20)

    p_chain = patch_sub.add_parser("ancestry", help="Show patch ancestry chain")
    p_chain.add_argument("patch_id", type=int)

    p_search = patch_sub.add_parser(
        "search", help="Search patches by vector or license"
    )
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
    p_retrieve.add_argument(
        "--no-cache", action="store_true", help="Bypass retrieval cache"
    )
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
    p_embed.add_argument("--backend", dest="backend", help="Vector backend to use")
    p_embed.add_argument(
        "--log-file", dest="log_file", help="Path to log skipped records"
    )
    p_embed.add_argument(
        "--verify",
        action="store_true",
        help="Verify vector counts match record totals after backfill",
    )
    p_embed.set_defaults(func=handle_embed)

    p_newdb = sub.add_parser("new-db", help="Scaffold a new database module")
    p_newdb.add_argument("name", help="Base name for the new database")
    p_newdb.set_defaults(func=handle_new_db)

    p_newvec = sub.add_parser("new-vector", help="Scaffold a new vector_service module")
    p_newvec.add_argument("name", help="Base name for the module")
    p_newvec.add_argument("--root", help="Target directory", default=None)
    p_newvec.add_argument(
        "--register-router", action="store_true", help="Update db_router"
    )
    p_newvec.add_argument(
        "--create-migration", action="store_true", help="Create alembic migration"
    )
    p_newvec.set_defaults(func=handle_new_vector)

    p_workflow = sub.add_parser(
        "workflow", help="Generate workflows using WorkflowSynthesizer"
    )
    p_workflow.add_argument("start", help="Starting module name")
    p_workflow.add_argument("--problem", help="Optional problem statement")
    p_workflow.add_argument(
        "--max-depth", type=int, dest="max_depth", help="Maximum traversal depth"
    )
    p_workflow.add_argument(
        "--limit",
        type=int,
        dest="limit",
        help="Maximum workflows to generate",
        default=5,
    )
    p_workflow.add_argument(
        "--out", help="File or directory to save generated workflows"
    )
    p_workflow.set_defaults(func=handle_workflow)

    # allow plugins to register additional subcommands
    from menace.plugins import load_plugins  # type: ignore

    load_plugins(sub)

    sub.add_parser("branch-log", help="Show patch branch audit trail").set_defaults(
        func=handle_branch_log
    )

    args = parser.parse_args(argv)
    builder = create_context_builder()
    setattr(args, "builder", builder)

    if hasattr(args, "func"):
        return args.func(args)

    if args.cmd == "setup":
        return _run(["bash", "scripts/setup_autonomous.sh"])

    if args.cmd == "test":
        return _run(["pytest"] + (args.pytest_args or []))

    if args.cmd == "improve":
        return _run(
            ["python", str(resolve_path("run_autonomous.py")), "--runs", "1"]
            + (args.extra_args or [])
        )

    if args.cmd == "benchmark":
        return _run(["python", str(resolve_path("workflow_benchmark.py"))])

    if args.cmd == "deploy":
        return _run(
            ["python", str(resolve_path("service_installer.py"))]
            + (args.extra_args or [])
        )

    if args.cmd == "sandbox":
        if args.sandbox_cmd == "run":
            return _run(
                ["python", str(resolve_path("run_autonomous.py"))]
                + (args.extra_args or [])
            )

    if args.cmd == "patches":
        global build_chain, search_patches_by_vector, search_patches_by_license
        if (
            build_chain is None
            or search_patches_by_vector is None
            or search_patches_by_license is None
        ):  # pragma: no cover - lazy import
            from patch_provenance import (
                build_chain as _build_chain,
                search_patches_by_vector as _search_patches_by_vector,
                search_patches_by_license as _search_patches_by_license,
            )
            build_chain = _build_chain
            search_patches_by_vector = _search_patches_by_vector
            search_patches_by_license = _search_patches_by_license

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
                if not _require_vector_service():
                    return 1
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
