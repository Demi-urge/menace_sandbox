from __future__ import annotations

"""Helpers for installing Python packages with optional offline mode."""

import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Dict

from .startup_checks import _parse_requirement

logger = logging.getLogger(__name__)


def install_packages(
    packages: Iterable[str], *, offline: bool = False, wheel_dir: str | Path | None = None
) -> dict[str, str]:
    """Install ``packages`` via pip and return a mapping of failures to error messages.

    Parameters
    ----------
    packages:
        Iterable of requirement strings to install.
    offline:
        If ``True`` network operations are skipped and each package is marked as
        skipped.
    """
    pkgs = [p for p in packages if p]
    errors: Dict[str, str] = {}

    if not pkgs:
        return errors

    if offline:
        if wheel_dir:
            wheel_path = Path(wheel_dir)
            for pkg in pkgs:
                mod = _parse_requirement(pkg)
                try:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-index",
                            "--find-links",
                            str(wheel_path),
                            pkg,
                        ]
                    )
                    importlib.import_module(mod)
                except Exception as exc:  # pragma: no cover - best effort
                    errors[pkg] = str(exc)
            return errors
        msg = "offline mode; installation skipped"
        for pkg in pkgs:
            errors[pkg] = msg
        return errors

    lock = None
    for name in ("uv.lock", "requirements.txt"):
        path = Path(name)
        if path.exists():
            lock = path
            break

    if lock:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(lock)])
        except Exception as exc:  # pragma: no cover - best effort
            errors[str(lock)] = str(exc)

    for pkg in pkgs:
        mod = _parse_requirement(pkg)
        success = False
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                importlib.import_module(mod)
                success = True
                break
            except Exception as exc:  # pragma: no cover - retry
                last_exc = exc
                time.sleep(1)
        if not success:
            errors[pkg] = str(last_exc) if last_exc else "unknown error"

    return errors


__all__ = ["install_packages"]
