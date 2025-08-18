"""Minimal fingerprint based license detection helpers."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Dict

class LicenseType(str, Enum):
    """Types of licenses recognised by the fingerprint detector."""

    UNKNOWN = "unknown"
    GPL = "GPL"
    NON_COMMERCIAL = "NonCommercial"

# Fingerprints for tiny representative snippets.  These are intentionally
# small and are not a replacement for full legal checks.
_GPL_SNIPPETS = [
    "gnu general public license",
    "gnu lesser general public license",
    "gnu affero general public license",
]

_CC_NC_SNIPPETS = [
    "creative commons attribution-noncommercial",
    "creativecommons.org/licenses/by-nc",
    "non-commercial",
]

# Pre-compute SHA-256 fingerprints for the snippets above for quick lookup.
_FINGERPRINTS: Dict[str, LicenseType] = {
    hashlib.sha256(s.encode("utf-8")).hexdigest(): LicenseType.GPL
    for s in _GPL_SNIPPETS
}
_FINGERPRINTS.update(
    {
        hashlib.sha256(s.encode("utf-8")).hexdigest(): LicenseType.NON_COMMERCIAL
        for s in _CC_NC_SNIPPETS
    }
)

def fingerprint(text: str) -> str:
    """Return SHA-256 fingerprint for ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def detect_license(text: str) -> LicenseType:
    """Detect disallowed license types from ``text``.

    This performs a simple substring search against known markers and, as a
    fallback, compares SHA-256 fingerprints of exact snippets.  It returns
    :class:`LicenseType.UNKNOWN` when no problematic license is detected.
    """

    lower = text.lower()
    for marker in _GPL_SNIPPETS:
        if marker in lower:
            return LicenseType.GPL
    for marker in _CC_NC_SNIPPETS:
        if marker in lower:
            return LicenseType.NON_COMMERCIAL
    fp = fingerprint(lower)
    return _FINGERPRINTS.get(fp, LicenseType.UNKNOWN)

__all__ = ["LicenseType", "detect_license", "fingerprint"]
