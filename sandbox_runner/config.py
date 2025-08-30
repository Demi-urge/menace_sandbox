"""Configuration helpers for sandbox runners.

This module determines where the sandbox repository lives.  Values are read
from the :envvar:`SANDBOX_REPO_URL` and :envvar:`SANDBOX_REPO_PATH`
environment variables or from a ``SandboxSettings`` instance when supplied.
If neither source provides a value sensible defaults are used that point to
the current checkout of the Menace sandbox repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_REPO_URL = "https://github.com/Demi-urge/menace_sandbox"
DEFAULT_REPO_PATH = Path(__file__).resolve().parents[1]


def get_sandbox_repo_url(settings: Any | None = None) -> str:
    """Return the sandbox repository URL.

    Parameters
    ----------
    settings:
        Optional ``SandboxSettings`` providing a ``sandbox_repo_url`` attribute.

    The function first checks ``settings`` then the ``SANDBOX_REPO_URL``
    environment variable before falling back to ``DEFAULT_REPO_URL``.
    """

    if settings and getattr(settings, "sandbox_repo_url", None):
        return str(getattr(settings, "sandbox_repo_url"))
    return os.getenv("SANDBOX_REPO_URL", DEFAULT_REPO_URL)


def get_sandbox_repo_path(settings: Any | None = None) -> Path:
    """Return the local sandbox repository path.

    Parameters
    ----------
    settings:
        Optional ``SandboxSettings`` providing a ``sandbox_repo_path`` attribute.

    ``settings`` takes precedence over the ``SANDBOX_REPO_PATH`` environment
    variable.  When neither is specified, ``DEFAULT_REPO_PATH`` is used.
    """

    if settings and getattr(settings, "sandbox_repo_path", None):
        return Path(getattr(settings, "sandbox_repo_path")).resolve()
    return Path(os.getenv("SANDBOX_REPO_PATH", DEFAULT_REPO_PATH)).resolve()


# Backwards compatible constants evaluated at import time
SANDBOX_REPO_URL = get_sandbox_repo_url()
SANDBOX_REPO_PATH = get_sandbox_repo_path()


__all__ = [
    "DEFAULT_REPO_URL",
    "DEFAULT_REPO_PATH",
    "get_sandbox_repo_url",
    "get_sandbox_repo_path",
    "SANDBOX_REPO_URL",
    "SANDBOX_REPO_PATH",
]


