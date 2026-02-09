from __future__ import annotations

"""Aggregate registry for recurring failures with cooldown handling."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("maintenance-logs/failure_registry_summary.json")


@dataclass(frozen=True)
class FailureFingerprint:
    exception_type: str
    message: str
    stage: str
    context: Mapping[str, Any]

    def digest(self) -> str:
        payload = {
            "exception_type": self.exception_type,
            "message": self.message,
            "stage": self.stage,
            "context": dict(self.context),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.debug("failed to read failure registry", exc_info=True)
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    if not isinstance(data, dict):
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    data.setdefault("version", 1)
    data.setdefault("failures", {})
    return data


def _persist_registry(path: Path, registry: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        LOGGER.exception("failed to persist failure registry")


def _coerce_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(context or {})


def _is_stripe_related(
    *,
    exception_type: str,
    message: str,
    stage: str,
    context: Mapping[str, Any],
) -> bool:
    needle = "stripe"
    candidates = [exception_type, message, stage]
    candidates.extend(str(key) for key in context.keys())
    candidates.extend(str(value) for value in context.values())
    return any(needle in str(candidate).lower() for candidate in candidates)


def _tag_stripe_context(
    *,
    exception_type: str,
    message: str,
    stage: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    tagged = dict(context)
    if not _is_stripe_related(
        exception_type=exception_type,
        message=message,
        stage=stage,
        context=tagged,
    ):
        return tagged
    tagged.setdefault("domain", "stripe")
    tagged.setdefault("severity", "non_fatal")
    return tagged


def compute_fingerprint(
    *,
    exception_type: str,
    message: str,
    stage: str,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable fingerprint for the failure payload."""

    fingerprint = FailureFingerprint(
        exception_type=exception_type,
        message=message,
        stage=stage,
        context=_coerce_context(context),
    )
    return fingerprint.digest()


def is_new_failure(record: Mapping[str, Any]) -> bool:
    """Return True when the registry record represents a first occurrence."""

    return int(record.get("occurrences", 0)) <= 1


def is_cooling_down(record: Mapping[str, Any], *, now: float | None = None) -> bool:
    """Return True if the record is still within its cooldown window."""

    timestamp = float(record.get("backoff_until", 0.0))
    now_value = time.time() if now is None else now
    return now_value < timestamp


def record_failure(
    *,
    exception_type: str,
    message: str,
    stage: str,
    context: Mapping[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    base_backoff: float = 60.0,
    max_backoff: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Record a failure in the aggregate registry and update cooldown state."""

    now_value = time.time() if now is None else now
    tagged_context = _tag_stripe_context(
        exception_type=exception_type,
        message=message,
        stage=stage,
        context=_coerce_context(context),
    )
    fingerprint = compute_fingerprint(
        exception_type=exception_type,
        message=message,
        stage=stage,
        context=tagged_context,
    )
    registry = _load_registry(registry_path)
    failures = registry.get("failures")
    if not isinstance(failures, dict):
        failures = {}

    record = failures.get(fingerprint)
    if not isinstance(record, dict):
        record = {
            "fingerprint": fingerprint,
            "exception_type": exception_type,
            "message": message,
            "stage": stage,
            "context": tagged_context,
            "occurrences": 0,
            "first_seen": now_value,
            "last_seen": now_value,
            "backoff_seconds": base_backoff,
            "backoff_until": now_value + base_backoff,
        }

    occurrences = int(record.get("occurrences", 0)) + 1
    record["occurrences"] = occurrences
    record["last_seen"] = now_value

    backoff_seconds = min(max_backoff, base_backoff * (2 ** max(0, occurrences - 1)))
    record["backoff_seconds"] = backoff_seconds
    record["backoff_until"] = now_value + backoff_seconds

    failures[fingerprint] = record
    registry["failures"] = failures
    registry["last_updated"] = now_value
    _persist_registry(registry_path, registry)
    return record


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "compute_fingerprint",
    "is_new_failure",
    "is_cooling_down",
    "record_failure",
    "FailureFingerprint",
]
