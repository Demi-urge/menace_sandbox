"""Launch, test and deploy Menace bots.

This entry point initialises :data:`GLOBAL_ROUTER` via :func:`init_db_router`
before importing modules that touch the database.  Doing so ensures all
database access uses the configured router rather than creating implicit
connections.  A :class:`ContextBuilder` is created with local database paths
inside :func:`debug_and_deploy` unless supplied by the caller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ast
import logging
import os
import uuid

from db_router import init_db_router
from scope_utils import Scope, build_scope_clause, apply_scope
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

# Placeholder assigned during runtime import within ``debug_and_deploy`` so
# tests can monkeypatch :class:`SelfDebuggerSandbox`.
SelfDebuggerSandbox = None  # type: ignore

from menace.code_database import CodeDB  # noqa: E402
from menace.menace_memory_manager import MenaceMemoryManager  # noqa: E402
from menace.self_coding_engine import SelfCodingEngine  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
from menace.error_bot import ErrorDB  # noqa: E402
from menace.error_logger import ErrorLogger  # noqa: E402
from menace.knowledge_graph import KnowledgeGraph  # noqa: E402
from menace.task_handoff_bot import TaskInfo  # noqa: E402
from menace.bot_testing_bot import BotTestingBot  # noqa: E402
from menace.deployment_bot import DeploymentBot, DeploymentSpec  # noqa: E402


def _extract_functions(code: str) -> list[str]:
    """Return a list of function names defined at module level."""

    try:
        tree = ast.parse(code)
    except Exception:
        return []
    return [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _extract_docstrings(code: str) -> tuple[str, dict[str, str]]:
    """Return the module docstring and a mapping of function docstrings."""

    try:
        tree = ast.parse(code)
    except Exception:
        return "", {}

    module_doc = ast.get_docstring(tree) or ""
    func_docs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                func_docs[node.name] = doc
    return module_doc, func_docs


def _estimate_resources(code: str) -> dict[str, float]:
    """Very rough CPU and memory estimation based on code length."""
    lines = code.count("\n") + 1
    cpu = max(0.1, round(lines / 1000, 2))
    memory = max(128.0, float(lines * 2))
    return {"cpu": cpu, "memory": memory}


def _infer_schedule(doc: str) -> str:
    """Derive a schedule hint from the module docstring."""
    low = doc.lower()
    if "hourly" in low:
        return "hourly"
    if "daily" in low:
        return "daily"
    return "once"


def debug_and_deploy(
    repo: Path,
    *,
    context_builder: ContextBuilder | None = None,
    jobs: int = 1,
    override_veto: bool = False,
) -> None:
    """Run tests, apply fixes and deploy existing bots in *repo*.

    Parameters
    ----------
    repo:
        Repository root containing bots to validate and deploy.
    context_builder:
        Optional :class:`~vector_service.context_builder.ContextBuilder` used
        for semantic context generation.  When ``None`` a new builder is
        instantiated using the local database paths.  Its weights are refreshed
        before building the coding engine.
    jobs:
        Number of parallel jobs for the test runner.
    override_veto:
        Whether to bypass governance vetoes when allowed.
    """

    try:
        code_db = CodeDB(router=GLOBAL_ROUTER)
    except TypeError:
        code_db = CodeDB()
    memory_mgr = MenaceMemoryManager()
    if context_builder is None:
        context_builder = ContextBuilder()
    context_builder.refresh_db_weights()
    engine = SelfCodingEngine(
        code_db, memory_mgr, context_builder=context_builder
    )
    error_db = ErrorDB(router=GLOBAL_ROUTER)
    tester = BotTestingBot()
    # instantiate telemetry logger for completeness
    try:
        _ = ErrorLogger(
            error_db, knowledge_graph=KnowledgeGraph(), context_builder=context_builder
        )
    except TypeError:
        _ = ErrorLogger(error_db, context_builder=context_builder)

    class _TelemProxy:
        def __init__(self, db: ErrorDB) -> None:
            self.db = db

        def recent_errors(
            self,
            limit: int = 5,
            *,
            scope: Scope | str = "local",
            source_menace_id: str | None = None,
        ) -> list[str]:
            menace_id = self.db._menace_id(source_menace_id)
            clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
            query = apply_scope(
                "SELECT stack_trace FROM telemetry",
                clause,
            ) + " ORDER BY id DESC LIMIT ?"
            cur = self.db.conn.execute(query, [*params, limit])
            return [str(r[0]) for r in cur.fetchall()]

        def patterns(
            self,
            *,
            scope: Scope | str = "local",
            source_menace_id: str | None = None,
        ) -> dict[str, int]:
            menace_id = self.db._menace_id(source_menace_id)
            clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
            query = apply_scope(
                "SELECT root_module, COUNT(*) FROM telemetry",
                clause,
            ) + " GROUP BY root_module"
            cur = self.db.conn.execute(query, params)
            return {str(r[0]): int(r[1]) for r in cur.fetchall()}

    global SelfDebuggerSandbox
    if SelfDebuggerSandbox is None:
        try:
            from menace.self_debugger_sandbox import (
                SelfDebuggerSandbox as _SelfDebuggerSandbox,
            )
        except Exception:
            class _SelfDebuggerSandbox:  # pragma: no cover - test fallback
                def __init__(self, *a, **k) -> None:  # noqa: D401
                    """Fallback stub when SelfDebuggerSandbox is unavailable."""

                def analyse_and_fix(self) -> None:
                    pass

        SelfDebuggerSandbox = _SelfDebuggerSandbox
    sandbox = SelfDebuggerSandbox(
        _TelemProxy(error_db), engine, context_builder=context_builder
    )
    try:
        deployer = DeploymentBot(
            code_db=code_db,
            error_db=error_db,
            db_router=GLOBAL_ROUTER,
            memory_mgr=memory_mgr,
        )
    except TypeError:
        deployer = DeploymentBot()

    # Collect modules from the entire repository tree so subpackages are
    # included. ``module_paths`` holds the file paths while ``module_names``
    # contains dotted import paths for the testing framework.
    # Skip files under .git and other auxiliary directories so only valid
    # source modules are processed by the pipeline
    _non_source = {
        ".git",
        "logs",
        "docs",
        "scripts",
        "tests",
        "sql_templates",
        "migrations",
        "systemd",
    }
    module_paths = [
        p
        for p in repo.rglob("*.py")
        if p.is_file() and not any(part in _non_source for part in p.parts)
    ]
    if not module_paths:
        print("No Python modules found to test")
        return

    module_names = [".".join(p.with_suffix("").relative_to(repo).parts) for p in module_paths]

    # 1) planning and optimisation -> development
    tasks = []
    for p in module_paths:
        code = p.read_text(encoding="utf-8")
        funcs = _extract_functions(code) or ["run"]
        description, func_docs = _extract_docstrings(code)
        res = _estimate_resources(code)
        schedule = _infer_schedule(description)
        context = " ".join((doc.splitlines()[0] for doc in func_docs.values()))
        tasks.append(
            TaskInfo(
                name=p.stem,
                dependencies=[],
                resources=res,
                schedule=schedule,
                code=code,
                metadata={
                    "purpose": p.stem,
                    "functions": funcs,
                    "description": description,
                    "function_docs": func_docs,
                    "context": context,
                },
            )
        )

    # Run the unit tests first using the importable module names
    tester.run_unit_tests(module_names)

    # Apply telemetry-driven patches after the initial test run
    sandbox.analyse_and_fix()

    # Store aggregated telemetry patterns for future reference
    for module, count in _TelemProxy(error_db).patterns().items():
        try:
            engine.memory_mgr.store(
                f"telemetry:{module}",
                {"errors": count},
                tags="telemetry",
            )
        except Exception:
            logging.getLogger(__name__).exception("memory store failed")

    # Deployment specification built from static resource estimates
    resources = {t.name: t.resources for t in tasks}
    telem_counts = _TelemProxy(error_db).patterns()
    spec = DeploymentSpec(name="menace", resources=resources, env={
        f"ERR_{k.upper()}": str(v) for k, v in telem_counts.items()
    })
    deployer.deploy("menace", [p.stem for p in module_paths], spec, override_veto=override_veto)


def main() -> None:
    """CLI entry point for self-debugging and deployment."""

    parser = argparse.ArgumentParser(description="Self-debug and deploy Menace")
    parser.add_argument("repo", nargs="?", default=".", help="Path to repo")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Parallel jobs")
    parser.add_argument(
        "--override-veto",
        action="store_true",
        default=False,
        help="Bypass governance vetoes when override is allowed",
    )
    args = parser.parse_args()
    debug_and_deploy(
        Path(args.repo),
        jobs=args.jobs,
        override_veto=args.override_veto,
    )


if __name__ == "__main__":
    main()
