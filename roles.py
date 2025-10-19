from __future__ import annotations

"""Dynamic role and permission management."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Set

from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

_DEFAULT_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "read": {"read"},
    "write": {"read", "write"},
    "admin": {"read", "write", "admin"},
}

default_path = resolve_path("config/role_permissions.json")
ROLE_PERMISSIONS_FILE = Path(os.getenv("ROLE_PERMISSIONS_FILE", str(default_path)))


def load_role_permissions(path: str | Path = ROLE_PERMISSIONS_FILE) -> Dict[str, Set[str]]:
    """Load role permissions from ``path``.

    Returns defaults if the file is missing or invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise TypeError("role permissions config must be a mapping")
        perms: Dict[str, Set[str]] = {}
        for role, actions in data.items():
            if not isinstance(actions, list | set):
                raise TypeError(f"Role {role} must map to a list of actions")
            perms[role] = set(actions)
        return perms
    except FileNotFoundError:
        logger.warning("Role permissions file not found at %s; using defaults", path)
    except Exception as exc:  # pragma: no cover - fallback
        logger.error("Failed to load role permissions: %s", exc)
    return {k: set(v) for k, v in _DEFAULT_ROLE_PERMISSIONS.items()}


ROLE_PERMISSIONS: Dict[str, Set[str]] = load_role_permissions()

READ = "read"
WRITE = "write"
ADMIN = "admin"

_default_role = os.getenv("DEFAULT_BOT_ROLE", READ)
if _default_role not in ROLE_PERMISSIONS:
    logger.warning(
        "Invalid DEFAULT_BOT_ROLE '%s'; falling back to %s",
        _default_role,
        READ,
    )
    _default_role = READ
DEFAULT_ROLE = _default_role

__all__ = [
    "ROLE_PERMISSIONS",
    "READ",
    "WRITE",
    "ADMIN",
    "DEFAULT_ROLE",
    "ROLE_PERMISSIONS_FILE",
    "load_role_permissions",
]
