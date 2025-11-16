"""Tools for removing secrets from text using regular expressions."""

from __future__ import annotations

import re
from typing import Iterable

# Regular expressions capturing common secret formats
_SECRET_PATTERNS: Iterable[re.Pattern[str]] = [
    # Generic API or access tokens
    re.compile(r"(?i)(?:api|secret|access|token)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=_-]{16,}['\"]?"),
    # AWS access key ids
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # JWT tokens
    re.compile(r"eyJ[0-9a-zA-Z_-]+\.[0-9a-zA-Z_-]+\.[0-9a-zA-Z_-]+"),
    # SSH or other private key blocks
    re.compile(
        r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----[\s\S]*?-----END(?: [A-Z]+)? PRIVATE KEY-----",
        re.MULTILINE,
    ),
]

_PLACEHOLDER = "[REDACTED]"


def redact_secrets(text: str, placeholder: str = _PLACEHOLDER) -> str:
    """Replace sensitive tokens in ``text`` with ``placeholder``.

    Parameters
    ----------
    text:
        Arbitrary text that may contain secrets.
    placeholder:
        Replacement string for detected secrets.
    """
    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


__all__ = ["redact_secrets"]
