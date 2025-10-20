from __future__ import annotations

"""Dynamic role and permission management."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Set

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

_DEFAULT_BOT_ROLES: Dict[str, str] = {}

_bot_roles_path = resolve_path("config/bot_roles.json")
BOT_ROLES_FILE = Path(os.getenv("BOT_ROLES_FILE", str(_bot_roles_path)))


def load_bot_roles(path: str | Path = BOT_ROLES_FILE) -> Dict[str, str]:
    """Load the bot to role mapping from ``path``.

    Entries are validated against :data:`ROLE_PERMISSIONS`.  Returns the default
    (empty) mapping when the file is missing or invalid.
    """

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, Mapping):
            raise TypeError("bot roles config must be a mapping of bot to role")
        mapping: Dict[str, str] = {}
        for bot, role in data.items():
            if not isinstance(bot, str):
                raise TypeError("bot identifiers must be strings")
            if not isinstance(role, str):
                raise TypeError(f"Role for bot '{bot}' must be a string")
            if role not in ROLE_PERMISSIONS:
                raise ValueError(f"Unknown role '{role}' for bot '{bot}'")
            mapping[bot] = role
        return mapping
    except FileNotFoundError:
        logger.warning("Bot roles file not found at %s; using defaults", path)
    except Exception as exc:  # pragma: no cover - fallback
        logger.error("Failed to load bot roles: %s", exc)
    return dict(_DEFAULT_BOT_ROLES)


BOT_ROLES: Dict[str, str] = load_bot_roles()

__all__ = [
    "READ",
    "WRITE",
    "ADMIN",
    "DEFAULT_ROLE",
    "ROLE_PERMISSIONS_FILE",
    "BOT_ROLES_FILE",
    "ROLE_PERMISSIONS",
    "load_role_permissions",
    "BOT_ROLES",
    "load_bot_roles",
]
