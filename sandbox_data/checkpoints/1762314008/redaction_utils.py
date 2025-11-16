import re
from typing import Any

# Patterns detecting common secret formats such as API tokens or private keys
_SECRET_PATTERNS = [
    # Generic API or access tokens: api_key=..., token: ..., etc.
    re.compile(r"(?i)(?:api|secret|access|token)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=_-]{16,}['\"]?"),
    # AWS Access Key IDs
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9\-._]+"),
    # Private key blocks (handles different key types or none)
    re.compile(
        r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----[\s\S]*?-----END(?: [A-Z]+)? PRIVATE KEY-----",
        re.MULTILINE,
    ),
]

_PLACEHOLDER = "[REDACTED]"

def redact_text(text: str, placeholder: str = _PLACEHOLDER) -> str:
    """Replace sensitive tokens in ``text`` with ``placeholder``.

    Args:
        text: Arbitrary text that may contain secrets.
        placeholder: Replacement string for detected secrets.
    """
    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text

def redact_dict(data: Any, placeholder: str = _PLACEHOLDER) -> Any:
    """Recursively apply :func:`redact_text` to all strings within ``data``."""
    if isinstance(data, dict):
        return {k: redact_dict(v, placeholder) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_dict(v, placeholder) for v in data]
    if isinstance(data, str):
        return redact_text(data, placeholder)
    return data

__all__ = ["redact_text", "redact_dict"]
