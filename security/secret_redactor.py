"""Secret redaction utilities."""

from __future__ import annotations

import re
from collections import Counter
from math import log2
from typing import Any

# Regular expressions capturing common secret formats such as API keys,
# passwords or tokens.
_SECRET_PATTERNS = [
    # Generic password assignments like password=..., "password: foo" etc.
    re.compile(r"(?i)password\s*[:=]\s*['\"]?[\w!@#\$%\^&\*\-/\\+=]{6,}['\"]?"),
    # Generic API or access tokens
    re.compile(r"(?i)(?:api|secret|access|token)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=_-]{16,}['\"]?"),
    # AWS Access Key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # AWS secret access keys
    re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?"),
    # Bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9\-._]+"),
    # Private key blocks
    re.compile(
        r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----[\s\S]*?-----END(?: [A-Z]+)? PRIVATE KEY-----",
        re.MULTILINE,
    ),
]

# Fallback regex used in combination with an entropy heuristic to catch
# high-entropy tokens that may represent secrets.
_TOKEN_RE = re.compile(r"[A-Za-z0-9/+=_-]{20,}")
_PLACEHOLDER = "[REDACTED]"


def _entropy(s: str) -> float:
    counts = Counter(s)
    length = len(s)
    probs = [c / length for c in counts.values()]
    return -sum(p * log2(p) for p in probs)


def redact(text: str, placeholder: str = _PLACEHOLDER) -> str:
    """Redact likely secrets from ``text``."""

    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(placeholder, text)

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if _entropy(token) >= 4.0:
            return placeholder
        return token

    return _TOKEN_RE.sub(_replace, text)


def redact_dict(data: Any, placeholder: str = _PLACEHOLDER) -> Any:
    """Recursively apply :func:`redact` to all strings within ``data``."""

    if isinstance(data, dict):
        return {k: redact_dict(v, placeholder) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_dict(v, placeholder) for v in data]
    if isinstance(data, str):
        return redact(data, placeholder)
    return data


__all__ = ["redact", "redact_dict"]
