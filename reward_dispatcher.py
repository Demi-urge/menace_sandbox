from __future__ import annotations

"""Secure reward signal dispatcher for Menace.

This module provides functions to compute rewards using the immutable KPI
core and securely deliver them to the Menace system via a shared file
system. The produced JSON files include an HMAC-SHA256 signature to allow
integrity verification and prevent tampering.
"""

from typing import Any
import json
import os
import time
import hmac
import hashlib

from dynamic_path_router import resolve_path

from .kpi_reward_core import compute_reward

REWARD_DIR = "/mnt/shared/security_ai"
REWARD_FILE = os.path.join(REWARD_DIR, "reward.json")
SECRET_KEY_PATH = resolve_path("secret.key")


def _load_secret_key() -> bytes:
    """Load the secret key used for HMAC signatures."""
    with open(SECRET_KEY_PATH, "rb") as fh:
        return fh.read().strip()


def _generate_signature(data: dict[str, Any], key: bytes) -> str:
    """Return a hex digest for *data* using *key*."""
    msg = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def dispatch_reward(log_entry: dict[str, Any]) -> str:
    """Compute reward for *log_entry* and write it securely.

    Returns the path to the written reward file.
    """
    reward_value = compute_reward(log_entry)
    timestamp = int(time.time())
    payload = {"reward": reward_value, "timestamp": timestamp, "used": False}

    key = _load_secret_key()
    payload["signature"] = _generate_signature(payload, key)

    primary_path = REWARD_FILE
    if os.path.exists(primary_path):
        try:
            with open(primary_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to read existing reward file: {exc}") from exc
        if not existing.get("used"):
            queued = os.path.join(REWARD_DIR, f"reward_{timestamp}.json")
            _write_json(queued, payload)
            return queued
    _write_json(primary_path, payload)
    return primary_path


def verify_reward_file(path: str = REWARD_FILE) -> bool:
    """Validate integrity of reward file at *path*."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to read reward file: {exc}") from exc

    signature = data.pop("signature", None)
    if not signature:
        return False

    key = _load_secret_key()
    expected = _generate_signature(data, key)
    return hmac.compare_digest(signature, expected)


__all__ = ["dispatch_reward", "verify_reward_file", "REWARD_FILE", "REWARD_DIR"]
