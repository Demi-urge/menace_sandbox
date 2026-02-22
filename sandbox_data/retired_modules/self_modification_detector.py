from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List

import requests

from .dynamic_path_router import resolve_path
from .sandbox_settings import SandboxSettings


_LOGGER = logging.getLogger(__name__)


class SelfModificationDetector:
    """Monitor the codebase for unexpected modifications."""

    def __init__(self, settings: SandboxSettings, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else resolve_path(".")
        self.interval_seconds = settings.self_mod_interval_seconds
        self.reference_path = self.base_dir / settings.self_mod_reference_path
        self.reference_url = settings.self_mod_reference_url
        self.lockdown_flag_path = Path(settings.self_mod_lockdown_flag_path)
        self.reference_hashes: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def generate_code_hashes(directory_path: str) -> Dict[str, str]:
        """Return SHA-256 hashes for all Python files under *directory_path*."""
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
                except OSError:
                    _LOGGER.exception("failed to hash %s", path)
                    raise
                rel = os.path.relpath(path, directory_path)
                hashes[rel] = digest
        return hashes

    # ------------------------------------------------------------------
    @staticmethod
    def save_reference_hashes(hash_dict: Dict[str, str], output_path: str) -> None:
        """Persist ``hash_dict`` to ``output_path`` in JSON format."""
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(hash_dict, fh, indent=2, sort_keys=True)
        except OSError:
            _LOGGER.exception("failed writing reference hashes %s", output_path)
            raise

    # ------------------------------------------------------------------
    @staticmethod
    def load_reference_hashes(path: str) -> Dict[str, str]:
        """Load previously saved reference hashes from ``path``."""
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_reference_hashes(url: str) -> Dict[str, str] | None:
        """Return reference hashes from ``url`` if possible."""
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network errors
            _LOGGER.error("failed to fetch reference hashes: %s", exc)
            raise
        data = resp.json()
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def detect_self_modification(
        reference_hashes: Dict[str, str], current_hashes: Dict[str, str]
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

    # ------------------------------------------------------------------
    def trigger_lockdown(self, file_list: List[str]) -> None:
        """Log fatal tampering message and halt the process."""
        _LOGGER.critical("SELF-MODIFICATION DETECTED: %s", ", ".join(file_list))
        try:
            with open(self.lockdown_flag_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp": time.time(), "files": file_list}))
        except OSError:
            _LOGGER.exception("failed to write lockdown flag %s", self.lockdown_flag_path)
            raise
        raise SystemExit("lockdown triggered due to self modification")

    # ------------------------------------------------------------------
    def _load_reference(self) -> None:
        if self.reference_url:
            fetched = self._fetch_reference_hashes(self.reference_url)
            if fetched:
                self.reference_hashes = fetched
                return
        if not self.reference_hashes:
            try:
                self.reference_hashes = self.load_reference_hashes(str(self.reference_path))
            except (OSError, json.JSONDecodeError):
                _LOGGER.exception(
                    "failed to load reference hashes from %s", self.reference_path
                )
                self.reference_hashes = self.generate_code_hashes(str(self.base_dir))
                self.save_reference_hashes(self.reference_hashes, str(self.reference_path))

    # ------------------------------------------------------------------
    def _monitor(self) -> None:
        self._load_reference()
        while not self._stop_event.wait(self.interval_seconds):
            if self.reference_url:
                try:
                    fetched = self._fetch_reference_hashes(self.reference_url)
                except requests.RequestException:
                    continue
                if fetched:
                    self.reference_hashes = fetched
            current = self.generate_code_hashes(str(self.base_dir))
            modified = self.detect_self_modification(self.reference_hashes, current)
            if modified:
                self.trigger_lockdown(modified)
                break

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # ------------------------------------------------------------------
    def reconfigure(self, settings: SandboxSettings) -> None:
        """Update configuration parameters from ``settings``."""
        self.interval_seconds = settings.self_mod_interval_seconds
        self.reference_path = self.base_dir / settings.self_mod_reference_path
        self.reference_url = settings.self_mod_reference_url
        self.lockdown_flag_path = Path(settings.self_mod_lockdown_flag_path)


# Convenience wrappers preserving the previous functional API -----------------

_DETECTOR: SelfModificationDetector | None = None


def generate_code_hashes(directory_path: str) -> Dict[str, str]:
    return SelfModificationDetector.generate_code_hashes(directory_path)


def save_reference_hashes(hash_dict: Dict[str, str], output_path: str) -> None:
    SelfModificationDetector.save_reference_hashes(hash_dict, output_path)


def load_reference_hashes(path: str) -> Dict[str, str]:
    return SelfModificationDetector.load_reference_hashes(path)


def detect_self_modification(
    reference_hashes: Dict[str, str], current_hashes: Dict[str, str]
) -> List[str]:
    return SelfModificationDetector.detect_self_modification(reference_hashes, current_hashes)


def monitor_self_integrity(settings: SandboxSettings, base_dir: str | Path | None = None) -> None:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = SelfModificationDetector(settings, base_dir)
    else:
        _DETECTOR.reconfigure(settings)
    _DETECTOR.start()


def stop_monitoring() -> None:
    global _DETECTOR
    if _DETECTOR is not None:
        _DETECTOR.stop()
        _DETECTOR = None


__all__ = [
    "SelfModificationDetector",
    "generate_code_hashes",
    "save_reference_hashes",
    "load_reference_hashes",
    "detect_self_modification",
    "monitor_self_integrity",
    "stop_monitoring",
]
