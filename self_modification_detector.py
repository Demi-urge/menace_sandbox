"""Self-modification detector for Security AI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List

import requests


_LOGGER = logging.getLogger(__name__)
_REFERENCE_HASHES: Dict[str, str] = {}
_MONITOR_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def generate_code_hashes(directory_path: str) -> Dict[str, str]:
    """Return SHA-256 hashes for all Python files under *directory_path*.

    Directories named ``log`` or ``config`` (and their plurals) are skipped.
    """
    hashes: Dict[str, str] = {}
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if d not in {"log", "logs", "config", "configs"}]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                rel = os.path.relpath(path, directory_path)
                hashes[rel] = digest
            except Exception:
                continue
    return hashes


def save_reference_hashes(hash_dict: Dict[str, str], output_path: str) -> None:
    """Persist ``hash_dict`` to ``output_path`` in JSON format."""
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(hash_dict, fh, indent=2, sort_keys=True)
    except Exception:
        _LOGGER.exception("failed writing reference hashes %s", output_path)


def load_reference_hashes(path: str) -> Dict[str, str]:
    """Load previously saved reference hashes from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fetch_reference_hashes(url: str) -> Dict[str, str] | None:
    """Return reference hashes from ``url`` if possible."""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:  # pragma: no cover - network errors not deterministic
        _LOGGER.error("failed to fetch reference hashes: %s", exc)
    return None


def detect_self_modification(
    reference_hashes: Dict[str, str],
    current_hashes: Dict[str, str],
) -> List[str]:
    """Return list of files whose hashes differ from ``reference_hashes``."""
    changed: List[str] = []
    for filename, ref_hash in reference_hashes.items():
        if current_hashes.get(filename) != ref_hash:
            changed.append(filename)
    for filename in current_hashes:
        if filename not in reference_hashes:
            changed.append(filename)
    return sorted(set(changed))


def trigger_lockdown(file_list: List[str]) -> None:
    """Log fatal tampering message and halt the process."""
    _LOGGER.critical("SELF-MODIFICATION DETECTED: %s", ", ".join(file_list))
    try:
        with open("lockdown.flag", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"timestamp": time.time(), "files": file_list}))
    except Exception as exc:
        _LOGGER.error("failed to write lockdown flag: %s", exc)
    raise SystemExit("lockdown triggered due to self modification")


def monitor_self_integrity(interval_seconds: int = 10, reference_url: str | None = None) -> None:
    """Start a background watchdog verifying code integrity.

    If ``reference_url`` is provided, reference hashes are periodically fetched
    from this location and compared against local hashes.
    """

    def _monitor() -> None:
        directory = os.path.dirname(os.path.abspath(__file__))
        reference = _REFERENCE_HASHES
        if reference_url:
            fetched = _fetch_reference_hashes(reference_url)
            if fetched:
                reference.clear()
                reference.update(fetched)
        if not reference:
            ref_path = os.path.join(directory, "immutable_reference.json")
            try:
                reference.update(load_reference_hashes(ref_path))
            except Exception:
                reference.update(generate_code_hashes(directory))
                try:
                    save_reference_hashes(reference, ref_path)
                except Exception:
                    pass
        while not _STOP_EVENT.wait(interval_seconds):
            if reference_url:
                fetched = _fetch_reference_hashes(reference_url)
                if fetched:
                    reference.clear()
                    reference.update(fetched)
            current = generate_code_hashes(directory)
            modified = detect_self_modification(reference, current)
            if modified:
                trigger_lockdown(modified)
                break

    global _MONITOR_THREAD
    if _MONITOR_THREAD is None or not _MONITOR_THREAD.is_alive():
        _STOP_EVENT.clear()
        _MONITOR_THREAD = threading.Thread(target=_monitor, daemon=True)
        _MONITOR_THREAD.start()


def stop_monitoring() -> None:
    """Stop the integrity monitoring thread if running."""
    global _MONITOR_THREAD
    _STOP_EVENT.set()
    if _MONITOR_THREAD is not None:
        try:
            _MONITOR_THREAD.join()
        finally:
            _MONITOR_THREAD = None


__all__ = [
    "generate_code_hashes",
    "save_reference_hashes",
    "load_reference_hashes",
    "detect_self_modification",
    "monitor_self_integrity",
    "stop_monitoring",
    "trigger_lockdown",
]
