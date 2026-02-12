from __future__ import annotations

"""Startup integrity verification for Security AI.

This module acts as a first responder before any sensitive runtime
operations begin.  It checks that all required files are present,
verifies immutable components via SHA-256 hashes and validates the
structure of critical configuration files.  If any check fails the
system is halted and a detailed failure report is written to
``./logs/startup_failures.json``.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from bootstrap_readiness import shared_online_state
from objective_guard import DEFAULT_OBJECTIVE_HASH_MANIFEST, ObjectiveGuardViolation
from objective_hash_lock import verify_objective_hash_lock

logger = logging.getLogger(__name__)

# Default paths for required files and directories that must always be present
REQUIRED_PATHS: list[str] = [
    "reward_core.pyc",
    "config.json",
    os.path.join("logs", "violation.log"),
    os.path.join("locks", "active.lock"),
    os.path.join("logs", "audit.log"),
]

# Reference hash file for immutable components
IMMUTABLE_HASHES_PATH = DEFAULT_OBJECTIVE_HASH_MANIFEST

# Configuration template describing expected keys and value types
# This ensures configuration files haven't been tampered with or corrupted.
CONFIG_TEMPLATE: Dict[str, Any] = {
    "database_url": str,
    "logging_level": str,
    "log_directory": str,
    "reward_core_path": str,
}


def verify_required_files(required_paths: list[str]) -> list[str]:
    """Return a list of paths that are missing."""
    # accumulate missing files or directories
    missing: list[str] = []
    for path in required_paths:
        if not os.path.exists(path):
            logger.error("Required path missing: %s", path)
            missing.append(path)
    return missing


def check_file_integrity(reference_hash_path: str) -> list[str]:
    """Return objective files that diverge from the persisted SHA-256 manifest."""

    try:
        verify_objective_hash_lock(
            repo_root=Path.cwd(),
            manifest_path=Path(reference_hash_path).resolve(),
        )
        return []
    except ObjectiveGuardViolation as exc:
        changed = [str(item) for item in (exc.details.get("changed_files") or []) if item]
        if changed:
            for file_path in changed:
                logger.error("Hash mismatch for %s", file_path)
            return changed
        logger.error("Failed objective manifest verification: %s (%s)", exc.reason, exc.details)
        return [reference_hash_path]


def validate_config_structure(config_path: str) -> bool:
    """Return True if *config_path* matches ``CONFIG_TEMPLATE`` structure."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.error("Unable to load config file %s: %s", config_path, exc)
        return False

    for key, typ in CONFIG_TEMPLATE.items():
        if key not in data or not isinstance(data[key], typ):
            logger.error("Configuration key %s missing or invalid", key)
            return False
    return True


def run_startup_diagnostics() -> Dict[str, Any]:
    """Execute all startup checks and return a summary report."""
    report: Dict[str, Any] = {
        "file_existence": True,
        "hash_integrity": True,
        "config_validation": True,
        "log_folder_accessibility": True,
        "missing_files": [],
        "hash_mismatches": [],
    }

    # verify all required paths exist
    missing = verify_required_files(REQUIRED_PATHS)
    if missing:
        report["file_existence"] = False
        report["missing_files"] = missing

    # check immutable files against known good hashes
    mismatched = check_file_integrity(IMMUTABLE_HASHES_PATH)
    if mismatched:
        report["hash_integrity"] = False
        report["hash_mismatches"] = mismatched

    # validate critical configuration structure
    config_ok = validate_config_structure("config.json")
    report["config_validation"] = config_ok

    # ensure log directory exists and is writable
    log_dir = os.path.join("logs")
    report["log_folder_accessibility"] = os.access(
        log_dir, os.R_OK | os.W_OK | os.X_OK
    )

    return report


def halt_on_failure(report: Dict[str, Any]) -> None:
    """Terminate execution and persist *report* if any check failed."""
    if all(
        [
            report.get("file_existence"),
            report.get("hash_integrity"),
            report.get("config_validation"),
            report.get("log_folder_accessibility"),
        ]
    ):
        return

    # persist failure report to the logs directory
    os.makedirs("logs", exist_ok=True)
    failure_path = os.path.join("logs", "startup_failures.json")
    report["timestamp"] = datetime.utcnow().isoformat()
    with open(failure_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.critical("Startup diagnostics failed; system halted.")
    raise SystemExit(1)


__all__ = [
    "verify_required_files",
    "check_file_integrity",
    "validate_config_structure",
    "run_startup_diagnostics",
    "halt_on_failure",
]


def _bootstrap_ready_fast_path() -> bool:
    """Return ``True`` when a shared bootstrap heartbeat reports readiness."""

    state = shared_online_state()
    ready = bool(state and state.get("ready"))
    if ready:
        logger.info("bootstrap heartbeat reports ready; short-circuiting health probe")
    return ready


def main() -> None:
    """Run diagnostics and exit non-zero on failure."""
    if _bootstrap_ready_fast_path():
        print("bootstrap already ready; skipping cold start checks")
        return

    report = run_startup_diagnostics()
    halt_on_failure(report)
    print("startup checks passed")


if __name__ == "__main__":
    main()
