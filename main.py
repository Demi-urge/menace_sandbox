"""Command line interface for running Menace modules.

The entry point initialises a database router via :func:`init_db_router` before
importing modules so that all database access goes through :data:`GLOBAL_ROUTER`.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import pkgutil
import sys
from typing import Iterable, List
import uuid

from db_router import init_db_router
from dynamic_path_router import get_project_root, resolve_path

# Ensure repository root on sys.path for direct execution
sys.path.insert(0, str(get_project_root()))

logger = logging.getLogger(__name__)

# Initialise the global router early so modules can access ``GLOBAL_ROUTER``
# without explicit dependency injection.  A unique ``menace_id`` keeps local
# tables isolated for this process. Explicit paths avoid relying on defaults.
MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)


def discover_modules() -> List[str]:
    """Return a list of modules within ``menace`` exposing a ``main`` function."""

    import menace

    modules: List[str] = []
    for _, modname, _ in pkgutil.iter_modules(menace.__path__, prefix="menace."):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # pragma: no cover - optional modules may fail
            logger.debug("Failed to import %s: %s", modname, exc)
            continue
        if hasattr(mod, "main"):
            modules.append(modname)
    return sorted(modules)


def run_module(module_name: str, argv: List[str] | None = None) -> None:
    """Import ``module_name`` and execute its ``main`` function."""

    argv = argv or []
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "main"):
        raise SystemExit(f"Module {module_name} has no main()")
    main_func = getattr(mod, "main")
    try:
        sig = inspect.signature(main_func)
    except (TypeError, ValueError):  # pragma: no cover - builtins etc
        sig = None

    if sig is not None and len(sig.parameters) > 0:
        main_func(argv)
    else:
        main_func()


def cli(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Menace module")
    parser.add_argument("module", nargs="?", default="menace.clipped.clipped_master",
                        help="Module to run")
    parser.add_argument("--list", dest="list_modules", action="store_true",
                        help="List available modules with a main() function")
    parser.add_argument("--", dest="mod_args", nargs=argparse.REMAINDER,
                        help="Arguments passed to the module")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_modules:
        for modname in discover_modules():
            print(modname)
        return

    mod_args = args.mod_args or []
    run_module(args.module, mod_args)


if __name__ == "__main__":
    cli()
