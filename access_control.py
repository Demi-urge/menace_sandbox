from __future__ import annotations

"""Access control helpers for Menace."""

import logging

from .roles import ROLE_PERMISSIONS, READ, WRITE, ADMIN
from .models import BotRole

logger = logging.getLogger(__name__)


def check_permission(role: str, action: str) -> None:
    """Raise ``PermissionError`` if *role* lacks permission for *action*."""
    if role not in ROLE_PERMISSIONS:
        logger.warning("Unknown role %s for action %s", role, action)
        raise PermissionError(f"Unknown role: {role}")
    if action not in ROLE_PERMISSIONS[role]:
        logger.warning("Unauthorized access attempt: role=%s action=%s", role, action)
        raise PermissionError(f"Role {role} not permitted for {action}")


__all__ = [
    "BotRole",
    "ROLE_PERMISSIONS",
    "check_permission",
    "READ",
    "WRITE",
    "ADMIN",
]
