from __future__ import annotations

"""Access control helpers for Menace."""

import logging
from fnmatch import fnmatch
from typing import Mapping

try:  # pragma: no cover - support execution without package context
    from .roles import (
        ROLE_PERMISSIONS,
        READ,
        WRITE,
        ADMIN,
        DEFAULT_ROLE,
        BOT_ROLES,
    )
    from .models import BotRole
except ImportError:  # pragma: no cover - fallback when imported as script
    from roles import (  # type: ignore
        ROLE_PERMISSIONS,
        READ,
        WRITE,
        ADMIN,
        DEFAULT_ROLE,
        BOT_ROLES,
    )
    from models import BotRole  # type: ignore

logger = logging.getLogger(__name__)


def check_permission(role: str, action: str) -> None:
    """Raise ``PermissionError`` if *role* lacks permission for *action*."""
    if role not in ROLE_PERMISSIONS:
        logger.warning("Unknown role %s for action %s", role, action)
        raise PermissionError(f"Unknown role: {role}")
    if action not in ROLE_PERMISSIONS[role]:
        logger.warning("Unauthorized access attempt: role=%s action=%s", role, action)
        raise PermissionError(f"Role {role} not permitted for {action}")


def resolve_bot_role(
    bot: str, roles_map: Mapping[str, str], default: str = DEFAULT_ROLE
) -> str:
    """Return the role for ``bot`` using wildcard-aware lookup."""

    if not bot:
        return default

    role = roles_map.get(bot)
    if role:
        return role

    for pattern, candidate in roles_map.items():
        if pattern == bot:
            continue
        if any(ch in pattern for ch in "*?[]") and fnmatch(bot, pattern):
            return candidate

    return roles_map.get("__default__", default)


__all__ = [
    "BotRole",
    "ROLE_PERMISSIONS",
    "check_permission",
    "READ",
    "WRITE",
    "ADMIN",
    "DEFAULT_ROLE",
    "BOT_ROLES",
    "resolve_bot_role",
]
