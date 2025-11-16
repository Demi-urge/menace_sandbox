from __future__ import annotations

"""Shared ROI tag definitions."""

from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RoiTag(str, Enum):
    SUCCESS = "success"
    HIGH_ROI = "high-ROI"
    LOW_ROI = "low-ROI"
    BUG_INTRODUCED = "bug-introduced"
    NEEDS_REVIEW = "needs-review"
    BLOCKED = "blocked"

    @classmethod
    def validate(cls, value: "RoiTag | str | None") -> "RoiTag":
        """Return a valid :class:`RoiTag`, defaulting to ``SUCCESS``.

        Unknown tags are logged and coerced to ``SUCCESS``.
        """

        if value is None:
            return cls.SUCCESS
        try:
            return cls(value)
        except ValueError:
            logger.warning("Invalid ROI tag %r, defaulting to 'success'", value)
            return cls.SUCCESS


__all__ = ["RoiTag"]
