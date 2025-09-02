"""Verify that critical dependencies are installed and match expected versions."""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
from typing import Dict, Iterable
import logging

logger = logging.getLogger(__name__)


def verify_dependencies(packages: Dict[str, str]) -> Dict[str, str]:
    """Return packages with missing or mismatched versions.

    Parameters
    ----------
    packages:
        Mapping of package name to required version string. ``""`` means any
        version is accepted but the package must be importable.
    """
    failures: Dict[str, str] = {}
    for name, version in packages.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures[name] = "missing"
            logger.warning(
                "package '%s' is not installed; install it with 'pip install %s' or upgrade with 'pip install --upgrade %s'",
                name,
                name,
                name,
            )
            continue
        except Exception as exc:
            failures[name] = "missing"
            logger.warning(
                "failed to determine version for package '%s': %s. install or upgrade with 'pip install %s' or 'pip install --upgrade %s'",
                name,
                exc,
                name,
                name,
            )
            continue
        if version and installed != version:
            failures[name] = installed
    return failures


def verify_modules(modules: Iterable[str]) -> list[str]:
    """Return list of modules that failed to import."""
    missing: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return missing


__all__ = ["verify_dependencies", "verify_modules"]
