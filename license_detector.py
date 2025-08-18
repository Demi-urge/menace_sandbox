"""Lightweight license header detector for embeddings.

This module combines simple fingerprint checks with heuristic
pattern searches to identify disallowed licenses such as GPL
variants or non‑commercial clauses.  It is not a substitute for
full legal review but serves as a guard rail to avoid embedding
obviously problematic text.
"""

from __future__ import annotations

import re
from typing import Optional

from compliance.license_fingerprint import check as _fingerprint_check, fingerprint

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------
# Regex patterns for common GPL headers and non‑commercial clauses.
_GPL_PATTERNS = [
    re.compile(r"gnu (?:affero |lesser )?general public license", re.I),
    re.compile(r"gpl\s*(?:v?\d)?", re.I),
]

_NONCOMM_PATTERNS = [
    re.compile(r"non[- ]commercial", re.I),
    re.compile(r"not\s+for\s+commercial", re.I),
]


def detect(text: str) -> Optional[str]:
    """Return the detected disallowed license name if present."""

    lic = _fingerprint_check(text)
    if lic:
        return lic
    lower = text.lower()
    for pat in _GPL_PATTERNS:
        if pat.search(lower):
            return "GPL"
    for pat in _NONCOMM_PATTERNS:
        if pat.search(lower):
            return "NonCommercial"
    return None


__all__ = ["detect", "fingerprint"]
