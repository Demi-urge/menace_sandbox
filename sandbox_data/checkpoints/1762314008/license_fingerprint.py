"""SPDX-style fingerprinting helpers for license compliance.

This module computes normalized fingerprints for code snippets and compares
them against a denylist of identifiers for GPL and non‑commercial licenses.
It is a lightweight safeguard and does not replace full legal review.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Dict

# ---------------------------------------------------------------------------
# Denylist
# ---------------------------------------------------------------------------
# Known phrases that indicate disallowed licenses.  The keys are example
# substrings; their normalized fingerprints are stored in ``DENYLIST`` for
# quick comparisons.
_SNIPPETS: Dict[str, str] = {
    "gnu general public license version 2": "GPL-2.0",
    "gnu general public license": "GPL-3.0",
    "gnu lesser general public license": "LGPL-3.0",
    "gnu affero general public license": "AGPL-3.0",
    "creative commons attribution-noncommercial": "CC-BY-NC-4.0",
    "non-commercial": "CC-BY-NC-4.0",
    "for non-commercial use only": "CC-BY-NC-4.0",
    "not for commercial use": "CC-BY-NC-4.0",
}


def _normalise(text: str) -> str:
    """Return a canonical form of ``text`` suitable for hashing."""

    return re.sub(r"\s+", "", text.lower())


DENYLIST: Dict[str, str] = {
    hashlib.sha256(_normalise(s).encode("utf-8")).hexdigest(): lic
    for s, lic in _SNIPPETS.items()
}
"""Mapping of fingerprint -> disallowed license identifier."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fingerprint(text: str) -> str:
    """Compute an SPDX‑like fingerprint for ``text``.

    The input is lower‑cased and all whitespace is stripped before hashing
    with SHA‑256, mirroring the normalisation used by SPDX snippet matching.
    """

    return hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest()


def check(text: str) -> Optional[str]:
    """Return the license identifier if ``text`` matches a denylisted snippet."""

    fp = fingerprint(text)
    lic = DENYLIST.get(fp)
    if lic:
        return lic
    # Fallback substring search for partial matches.
    norm = _normalise(text)
    for snippet, identifier in _SNIPPETS.items():
        if _normalise(snippet) in norm:
            return identifier
    return None


__all__ = ["fingerprint", "check", "DENYLIST"]
