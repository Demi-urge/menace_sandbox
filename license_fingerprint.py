"""Simple fingerprint-based license detector.

This module stores pre-computed SHA-256 fingerprints for common
open‑source licenses that are disallowed by project policy.  It exposes a
:func:`check` helper that returns the matching license name when the
input text matches one of the fingerprints.

The fingerprints are intentionally short and only cover a minimal set of
licenses – they are **not** a full legal solution but act as a quick
heuristic to prevent obviously problematic content from being embedded
into vector stores.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
# The values below are SHA-256 digests of small representative snippets
# from the respective licenses.  They are intentionally truncated
# examples; real implementations would use more robust datasets.
DISALLOWED_HASHES = {
    # "GNU GENERAL PUBLIC LICENSE"
    "77e79ead261b00cd1de03ebd3876540a69b15424bd68fcee642e6c2d93d36093": "GPL-3.0",
    # "GNU LESSER GENERAL PUBLIC LICENSE"
    "3d4f3c52e5b14404ba093397522d933b911752c78f9fe0f2c6764a447216b55d": "LGPL-3.0",
    # "AFFERO GENERAL PUBLIC LICENSE" (AGPL)
    "03a7779e57dfc0c063d64693c8c1bd248bf1db884ebf28752221e9cdb2595eab": "AGPL-3.0",
}


def fingerprint(text: str) -> str:
    """Return the SHA-256 hex digest for ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(text: str) -> Optional[str]:
    """Return the license name if ``text`` matches a disallowed fingerprint.

    Parameters
    ----------
    text:
        The text snippet to fingerprint.

    Returns
    -------
    Optional[str]
        The detected license name or ``None`` if the snippet appears
        clean.
    """

    return DISALLOWED_HASHES.get(fingerprint(text))


__all__ = ["check", "fingerprint", "DISALLOWED_HASHES"]
