from __future__ import annotations

"""Startup validations for dependencies and configuration."""

import importlib
import os
import sys
import subprocess
import base64
from pathlib import Path
from typing import Iterable, Dict, Sequence
import logging

from .audit_trail import AuditTrail

import tomllib

from .dependency_verifier import verify_dependencies, verify_modules
from .dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

OPTIONAL_LIBS = [
    "pandas",
    "sklearn",
    "stripe",  # router will fail at runtime if this is absent
    "httpx",
]

# critical dependencies with expected versions
CRITICAL_LIBS: Dict[str, str] = {
    "pandas": "",
    "torch": "",
    "networkx": "",
}

# Default path to the repository's ``pyproject.toml``
PYPROJECT_PATH = resolve_path("pyproject.toml")

REQUIRED_VARS = [
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "MENACE_EMAIL",
    "MENACE_PASSWORD",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]
# Stripe keys are handled by stripe_billing_router and intentionally omitted.


def validate_dependencies(modules: Iterable[str] = OPTIONAL_LIBS) -> list[str]:
    """Return list of optional modules that are missing."""
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        logger.warning(
            "Missing optional dependencies: %s", ", ".join(missing)
        )
    return missing


def verify_optional_dependencies(modules: Iterable[str] | None = None) -> list[str]:
    """Return list of optional modules specific to the sandbox that are missing."""

    modules = list(modules) if modules is not None else ["quick_fix_engine"]
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            logger.warning("optional dependency %s is missing", mod)
            missing.append(mod)
    return missing


def _parse_requirement(req: str) -> str:
    """Return the package name portion of a dependency string."""
    name = req.split(";")[0].strip()
    name = name.split("[")[0]
    for sep in ("==", ">=", "<=", "~=", ">", "<"):
        if sep in name:
            name = name.split(sep)[0]
            break
    return name.strip()


def _coerce_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute :class:`Path` within the repo."""

    if isinstance(path, Path):
        return path if path.is_absolute() else resolve_path(path.as_posix())
    return resolve_path(str(path))


def dependencies_from_pyproject(path: str | Path = PYPROJECT_PATH) -> Sequence[str]:
    """Return list of dependency package names from pyproject."""

    target = _coerce_path(path)
    try:
        with open(target, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return []
    deps = data.get("project", {}).get("dependencies", [])
    return [_parse_requirement(d) for d in deps]


def verify_project_dependencies(path: str | Path = PYPROJECT_PATH) -> list[str]:
    """Return missing modules declared in pyproject."""

    modules = dependencies_from_pyproject(path)
    return verify_modules(modules)


def verify_stripe_router(mandatory_bot_ids: Iterable[str] | None = None) -> None:
    """Import ``stripe_billing_router`` and ensure keys and routes exist.

    ``mandatory_bot_ids`` allows callers to require that specific bots have
    valid billing routes.  Each identifier in the iterable is passed to
    ``stripe_billing_router._resolve_route`` and a ``RuntimeError`` is raised if
    any lookup fails.
    """
    repo_root = resolve_path(".")
    try:
        files = subprocess.check_output(
            ["git", "ls-files", "*.py"],
            cwd=repo_root,
            text=True,
        ).splitlines()
    except Exception as exc:  # pragma: no cover - git failure
        raise RuntimeError(f"git ls-files failed: {exc}") from exc

    checks = [
        [
            sys.executable,
            str(resolve_path("scripts/check_stripe_imports.py")),
            *files,
        ],
        [
            sys.executable,
            str(resolve_path("scripts/check_raw_stripe_usage.py")),
        ],
    ]
    for cmd in checks:
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)

    import importlib

    module_name = f"{__package__}.stripe_billing_router" if __package__ else "stripe_billing_router"
    try:
        sbr = sys.modules.get(module_name) or importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure
        raise RuntimeError(f"stripe_billing_router import failed: {exc}") from exc
    if not getattr(sbr, "BILLING_RULES", None):
        raise RuntimeError("stripe_billing_router has no billing rules configured")
    if not getattr(sbr, "STRIPE_SECRET_KEY", "") or not getattr(
        sbr, "STRIPE_PUBLIC_KEY", ""
    ):
        raise RuntimeError("stripe_billing_router is missing Stripe API keys")

    # Ensure every bot in the registry has a corresponding billing route
    reg_module_name = f"{__package__}.bot_registry" if __package__ else "bot_registry"
    try:
        reg_mod = sys.modules.get(reg_module_name) or importlib.import_module(
            reg_module_name
        )
    except Exception as exc:  # pragma: no cover - import failure
        raise RuntimeError(f"bot_registry import failed: {exc}") from exc

    bots: set[str] = set()

    # Attempt to extract bot names from a registry mapping first
    registry = getattr(reg_mod, "REGISTRY", None)
    if registry is not None:
        try:
            bots = set(registry.keys() if isinstance(registry, dict) else registry)
        except Exception:
            bots = set()
    elif hasattr(reg_mod, "BotRegistry"):
        persist = os.getenv("BOT_DB_PATH", "bots.db")
        try:
            reg_obj = reg_mod.BotRegistry(persist=persist)
        except Exception as exc:  # pragma: no cover - registry load failure
            raise RuntimeError(f"bot registry load failed: {exc}") from exc
        graph = getattr(reg_obj, "graph", None)
        if graph is not None and hasattr(graph, "nodes"):
            try:
                bots = {str(n) for n in graph.nodes}
            except Exception:
                bots = set()
        elif hasattr(reg_obj, "bots"):
            bots = {str(n) for n in getattr(reg_obj, "bots")}

    routing_table = getattr(sbr, "ROUTING_TABLE", getattr(sbr, "BILLING_RULES", {}))
    missing = [b for b in bots if not any(key[-1] == b for key in routing_table)]
    if missing:
        raise RuntimeError(
            f"Missing billing routes for bots: {', '.join(sorted(missing))}"
        )

    required = list(mandatory_bot_ids or [])
    if required:
        resolver = getattr(sbr, "_resolve_route", None)
        if resolver is None:
            raise RuntimeError("stripe_billing_router is missing _resolve_route")
        failures: list[str] = []
        for bot_id in required:
            try:
                resolver(bot_id)
            except Exception:
                failures.append(bot_id)
        if failures:
            raise RuntimeError(
                "Missing billing routes for required bots: "
                + ", ".join(sorted(failures))
            )


def validate_config(vars: Iterable[str] = REQUIRED_VARS) -> list[str]:
    """Return missing vars and raise in production mode."""
    missing = [v for v in vars if not os.getenv(v)]
    mode = os.getenv("MENACE_MODE", "test").lower()
    if mode == "production" and missing:
        raise RuntimeError(
            f"Missing required configuration variables: {', '.join(missing)}"
        )
    elif missing:
        logger.warning(
            "Missing configuration variables: %s", ", ".join(missing)
        )
    return missing


def verify_critical_libs(libs: Dict[str, str] = CRITICAL_LIBS) -> Dict[str, str]:
    """Check for critical dependencies and versions."""
    failures = verify_dependencies(libs)
    if failures:
        for name, status in failures.items():
            logger.error(
                "Critical dependency issue for %s: %s", name, status
            )
    return failures


def _install_packages(packages: Iterable[str]) -> None:
    for pkg in packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed installing %s: %s", pkg, exc)


def _prompt_for_vars(names: Iterable[str]) -> None:
    if os.getenv("MENACE_NON_INTERACTIVE") == "1" or not sys.stdin.isatty():
        return
    for name in names:
        try:
            value = input(f"{name}: ").strip()
        except EOFError:
            value = ""
        if value:
            os.environ[name] = value


def run_startup_checks(pyproject_path: str | Path | None = None) -> None:
    """Run dependency and configuration checks."""
    missing_optional = validate_dependencies()
    if missing_optional:
        _install_packages(missing_optional)
    verify_optional_dependencies()
    missing = verify_project_dependencies(pyproject_path or PYPROJECT_PATH)
    verify_stripe_router()
    mode = os.getenv("MENACE_MODE", "test").lower()
    if missing:
        msg = f"Missing required dependencies: {', '.join(missing)}"
        if mode == "production":
            raise RuntimeError(msg)
        else:
            logger.warning(msg)
    vars_missing = validate_config()
    if vars_missing:
        _prompt_for_vars(vars_missing)
        vars_missing = [v for v in vars_missing if not os.getenv(v)]
        if vars_missing and mode == "production":
            raise RuntimeError(
                f"Missing required configuration variables: {', '.join(vars_missing)}"
            )
    verify_critical_libs()

    audit_path = os.getenv("AUDIT_LOG_PATH", "audit.log")
    pubkey_env = os.getenv("AUDIT_PUBKEY")
    path_obj = Path(audit_path)
    if path_obj.exists() and pubkey_env:
        try:
            pubkey = base64.b64decode(pubkey_env)
        except Exception:
            raise RuntimeError("Invalid AUDIT_PUBKEY")
        trail = AuditTrail(audit_path)
        if not trail.verify(pubkey):
            raise RuntimeError("Audit log verification failed")


__all__ = [
    "run_startup_checks",
    "validate_dependencies",
    "validate_config",
    "verify_critical_libs",
    "verify_project_dependencies",
    "dependencies_from_pyproject",
    "verify_optional_dependencies",
    "verify_stripe_router",
]
