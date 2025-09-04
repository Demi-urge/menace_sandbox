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

logger = logging.getLogger(__name__)

OPTIONAL_LIBS = [
    "pandas",
    "sklearn",
    "stripe",
    "httpx",
]

# critical dependencies with expected versions
CRITICAL_LIBS: Dict[str, str] = {
    "pandas": "",
    "torch": "",
    "networkx": "",
}

# Default path to the repository's ``pyproject.toml``
PYPROJECT_PATH = Path(__file__).resolve().with_name("pyproject.toml")

REQUIRED_VARS = [
    "STRIPE_SECRET_KEY",
    "STRIPE_PUBLIC_KEY",
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "MENACE_EMAIL",
    "MENACE_PASSWORD",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]


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


def dependencies_from_pyproject(path: Path = PYPROJECT_PATH) -> Sequence[str]:
    """Return list of dependency package names from pyproject."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return []
    deps = data.get("project", {}).get("dependencies", [])
    return [_parse_requirement(d) for d in deps]


def verify_project_dependencies(path: Path = PYPROJECT_PATH) -> list[str]:
    """Return missing modules declared in pyproject."""
    modules = dependencies_from_pyproject(path)
    return verify_modules(modules)


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


def run_startup_checks(pyproject_path: Path | None = None) -> None:
    """Run dependency and configuration checks."""
    missing_optional = validate_dependencies()
    if missing_optional:
        _install_packages(missing_optional)
    verify_optional_dependencies()
    missing = verify_project_dependencies(pyproject_path or PYPROJECT_PATH)
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
]
