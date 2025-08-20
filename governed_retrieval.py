from __future__ import annotations

"""Helper utilities for safe retrieval handling.

This module centralises light‑weight governance steps applied to text
returned from vector databases or other retrieval systems.  It exposes a
single :func:`govern_retrieval` helper which combines licence detection,
secret redaction and semantic risk analysis.  The :mod:`security` package
provides the redaction primitives which are re‑exported here for
convenience so callers only need to import this module.
"""

from typing import Any, Dict, Optional, Tuple

from license_detector import detect as license_detect
from security.secret_redactor import redact, redact_dict
from analysis.semantic_diff_filter import find_semantic_risks
from compliance.license_fingerprint import fingerprint as license_fingerprint


def govern_retrieval(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    max_alert_severity: float = 1.0,
) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
    """Return governed ``(metadata, reason)`` for *text* or ``None`` if blocked.

    The function performs the following steps:

    * Detect disallowed licences using :func:`license_detect`.  When a
      licence is found ``None`` is returned signalling the caller to skip the
      item entirely.
    * Identify semantic risks with :func:`find_semantic_risks` and attach the
      resulting alert list under ``"semantic_alerts"`` in the returned
      metadata.  ``max_alert_severity`` controls the maximum allowed alert
      score; when exceeded ``None`` is returned to signal that the item should
      be skipped.  The highest alert score is also stored under
      ``"alignment_severity"`` for downstream tracking.
    * Redact secrets from ``metadata`` and ``reason`` using the
      :mod:`security.secret_redactor` helpers.  The ``"redacted"`` flag is set
      on the metadata to signal downstream consumers that secrets have been
      removed.
    """

    lic = license_detect(text)
    if lic:
        return None

    alerts = find_semantic_risks(text.splitlines())
    meta: Dict[str, Any] = metadata.copy() if isinstance(metadata, dict) else {}
    if alerts:
        meta.setdefault("semantic_alerts", alerts)
        max_sev = max(score for _, _, score in alerts)
        meta.setdefault("alignment_severity", max_sev)
        if max_sev > max_alert_severity:
            return None
    meta = redact_dict(meta)
    meta.setdefault("redacted", True)
    meta.setdefault("license_fingerprint", license_fingerprint(text))
    cleaned_reason = redact(reason) if isinstance(reason, str) else reason
    return meta, cleaned_reason


__all__ = ["govern_retrieval", "license_detect", "redact", "redact_dict", "find_semantic_risks"]
