from __future__ import annotations

"""Manual override handler for Security AI violations.

This module enables a verified human operator to override logged violations.
It preserves all records for auditability and requires cryptographic
validation via :mod:`override_validator`.
"""

import json
import os
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict

from dynamic_path_router import resolve_dir, resolve_path

from .override_validator import verify_signature

# ---------------------------------------------------------------------------
# Paths for violation logs and override records
LOG_DIR = resolve_dir("logs")
VIOLATION_LOG = resolve_path("logs/violation_log.jsonl")
OVERRIDE_LOG = resolve_path("logs/violation_overrides.jsonl")


def _ensure_log_dir() -> None:
    """Create the log directory if missing."""
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Return all records from ``path`` if it exists."""
    if not os.path.exists(path):
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Public API


def load_violation_by_id(violation_id: str) -> dict[str, Any]:
    """Return the first violation entry matching ``violation_id``."""
    for record in _load_jsonl(VIOLATION_LOG):
        if record.get("entry_id") == violation_id:
            return record
    return {}


def _load_override_record(violation_id: str) -> dict[str, Any] | None:
    """Return the most recent override record for ``violation_id`` if any."""
    records = [r for r in _load_jsonl(OVERRIDE_LOG) if r.get("violation_id") == violation_id]
    return records[-1] if records else None


def is_override_validated(violation_id: str) -> bool:
    """Return ``True`` if ``violation_id`` already has an override record."""
    return _load_override_record(violation_id) is not None


def apply_override(violation_id: str, override_path: str, public_key_path: str) -> dict[str, Any]:
    """Validate and apply an override file.

    Parameters
    ----------
    violation_id:
        The identifier of the violation to override.
    override_path:
        Path to the JSON override instruction file.
    public_key_path:
        Path to the public key used for signature verification.

    Returns
    -------
    dict
        The override record that was written, or an empty dict if validation
        failed.
    """
    try:
        with open(override_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}

    data = payload.get("data") if isinstance(payload, dict) else None
    signature = payload.get("signature") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(signature, str):
        return {}

    # Ensure the override targets this violation
    payload_vid = data.get("violation_id") or data.get("action_id")
    if payload_vid != violation_id:
        return {}

    if not verify_signature(data, signature, public_key_path):
        return {}

    existing = _load_override_record(violation_id)
    if existing and existing.get("signature") == signature:
        return existing

    _ensure_log_dir()
    record = {
        "override_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "violation_id": violation_id,
        "data": data,
        "signature": signature,
    }
    with open(OVERRIDE_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def generate_override_summary(violation_id: str) -> str:
    """Return a human readable summary of the violation and its override."""
    violation = load_violation_by_id(violation_id)
    override = _load_override_record(violation_id)

    if not violation:
        return f"Violation {violation_id} not found."

    ts_violation = violation.get("timestamp")
    if isinstance(ts_violation, (int, float)):
        vio_time = datetime.utcfromtimestamp(float(ts_violation)).isoformat()
    else:
        try:
            vio_time = str(datetime.fromisoformat(str(ts_violation)))
        except Exception:
            vio_time = str(ts_violation)

    lines = [f"Violation {violation_id}:", f"  time: {vio_time}"]
    if "violation_type" in violation:
        lines.append(f"  type: {violation.get('violation_type')}")
    if "severity" in violation:
        lines.append(f"  severity: {violation.get('severity')}")
    if "evidence" in violation:
        lines.append(f"  evidence: {json.dumps(violation.get('evidence'))}")

    if override:
        lines.append("Override applied:")
        lines.append(f"  time: {override.get('timestamp')}")
        ov_data = override.get("data", {})
        new_type = ov_data.get("new_classification")
        if new_type is not None:
            lines.append(f"  new_classification: {new_type}")
        new_sev = ov_data.get("new_severity")
        if new_sev is not None:
            lines.append(f"  new_severity: {new_sev}")
        reason = ov_data.get("reason")
        if reason:
            lines.append(f"  reason: {reason}")
    else:
        lines.append("No override recorded.")

    return "\n".join(lines)


__all__ = [
    "load_violation_by_id",
    "apply_override",
    "is_override_validated",
    "generate_override_summary",
]
