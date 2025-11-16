from __future__ import annotations
"""Verify required dependencies and attempt installation if missing."""

import importlib
import json
import subprocess
from pathlib import Path
from typing import Iterable
import logging
import sys

from .dependency_verifier import verify_dependencies


logger = logging.getLogger(__name__)

INSTALL_CACHE = Path.home() / ".menace" / "install_cache.json"


def _load_cache() -> dict[str, str]:
    try:
        with open(INSTALL_CACHE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_cache(cache: dict[str, str]) -> None:
    try:
        INSTALL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(INSTALL_CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("failed to write install cache: %s", exc)


def _parse_pyproject(path: str = "pyproject.toml") -> dict[str, str]:
    deps: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return deps
    try:
        import tomllib  # Python >=3.11
    except Exception:  # pragma: no cover - missing
        return deps
    data = tomllib.loads(p.read_text())
    for item in data.get("project", {}).get("dependencies", []):
        pkg = item.split(";")[0].strip()
        if pkg:
            name = pkg.split("==")[0].strip()
            ver = ""
            if "==" in pkg:
                ver = pkg.split("==", 1)[1].strip()
            deps[name] = ver
    return deps


def install_missing(packages: dict[str, str]) -> None:
    cache = _load_cache()
    updated = False
    for pkg, ver in packages.items():
        if pkg in cache and (cache[pkg] == ver or not ver):
            logger.debug("skipping cached install for %s", pkg)
            continue
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            target = f"{pkg}=={ver}" if ver else pkg
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", target])
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("failed to install %s: %s", target, exc)
                continue
            cache[pkg] = ver
            updated = True
    if updated:
        _write_cache(cache)


def self_check(requirements: Iterable[str] | None = None) -> None:
    deps = _parse_pyproject()
    if requirements:
        for r in requirements:
            deps.setdefault(r, "")
    missing = verify_dependencies(deps)
    if missing:
        install_missing(missing)


__all__ = ["self_check"]
