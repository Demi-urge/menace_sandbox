from __future__ import annotations

"""Manual override validation utilities using HMAC signatures."""

from typing import Any, Tuple
import hashlib
import hmac
import json
import os


def _load_key(path: str) -> bytes:
    """Return the key bytes from ``path``."""
    with open(path, "rb") as fh:
        return fh.read().strip()


def _normalize(data: dict[str, Any]) -> bytes:
    """Return a canonical JSON representation for signing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_signature(data: dict[str, Any], private_key_path: str) -> str:
    """Sign *data* with the key at ``private_key_path``.

    Returns the hex encoded SHA-256 HMAC digest.
    """
    key = _load_key(private_key_path)
    msg = _normalize(data)
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_signature(data: dict[str, Any], signature: str, public_key_path: str) -> bool:
    """Return ``True`` if ``signature`` matches ``data`` using the key at ``public_key_path``."""
    key = _load_key(public_key_path)
    msg = _normalize(data)
    expected = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def validate_override_file(override_path: str, public_key_path: str) -> Tuple[bool, dict[str, Any]]:
    """Validate and mark an override instruction file.

    The override file must be a JSON object with ``data`` and ``signature`` fields.
    If verification succeeds, the file will be marked as used to prevent replay.
    Returns ``(True, data)`` on success, ``(False, {})`` otherwise.
    """
    try:
        with open(override_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return False, {}

    signature = payload.get("signature")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(signature, str):
        return False, {}

    if payload.get("used"):
        return False, {}

    valid = verify_signature(data, signature, public_key_path)
    if valid:
        payload["used"] = True
        try:
            with open(override_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass
    return valid, data if valid else {}


def execute_valid_override(data: dict[str, Any]) -> None:
    """Execute a validated override command.

    Supported commands include unlocking evolution, clearing lockdowns and
    modifying configuration flags. This function assumes *data* has already been
    verified via :func:`validate_override_file`.
    """
    override_type = data.get("override_type")
    if override_type == "unlock_evolution":
        os.environ.pop("EVOLUTION_LOCK", None)
    elif override_type == "clear_lockdown":
        lock_path = "lockdown.flag"
        if os.path.exists(lock_path):
            os.remove(lock_path)
    elif override_type == "set_config":
        name = data.get("name")
        value = data.get("value")
        if isinstance(name, str):
            os.environ[name] = str(value)
    else:
        raise ValueError(f"Unknown override type: {override_type}")


__all__ = [
    "generate_signature",
    "verify_signature",
    "validate_override_file",
    "execute_valid_override",
]
