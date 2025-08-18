import re
from collections import Counter
from math import log2
from typing import Any

_SECRET_PATTERNS = [
    # Generic password assignments
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

_TOKEN_RE = re.compile(r"[A-Za-z0-9/+=_-]{20,}")
_PLACEHOLDER = "[REDACTED]"


def _entropy(s: str) -> float:
    counts = Counter(s)
    length = len(s)
    probs = [c / length for c in counts.values()]
    return -sum(p * log2(p) for p in probs)


def redact_secrets(text: str, placeholder: str = _PLACEHOLDER) -> str:
    """Redact likely secrets from ``text``.

    Combines regular expressions for known secret formats with a
    high-entropy heuristic to catch generic tokens.
    """
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


def redact_secrets_dict(data: Any, placeholder: str = _PLACEHOLDER) -> Any:
    """Recursively apply :func:`redact_secrets` to all strings within ``data``."""
    if isinstance(data, dict):
        return {k: redact_secrets_dict(v, placeholder) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_secrets_dict(v, placeholder) for v in data]
    if isinstance(data, str):
        return redact_secrets(data, placeholder)
    return data


__all__ = ["redact_secrets", "redact_secrets_dict"]
