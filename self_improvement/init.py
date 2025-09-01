"""Helpers for the self-improvement subsystem.

The routines expect auxiliary packages to be installed:

* ``quick_fix_engine`` – generates corrective patches.
* ``sandbox_runner.orphan_integration`` – reintroduces orphaned modules.

Missing dependencies raise :class:`RuntimeError` with guidance on how to
install them.

Configuration is provided via :class:`sandbox_settings.SandboxSettings` with
notable options:

* ``orphan_retry_attempts`` – retry attempts for orphan integration hooks.
* ``orphan_retry_delay`` – delay between retries for orphan integration hooks.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from filelock import FileLock

from sandbox_settings import SandboxSettings, load_sandbox_settings
from sandbox_runner.bootstrap import initialize_autonomous_sandbox

try:
    from ..logging_utils import get_logger, setup_logging, log_record
except Exception:  # pragma: no cover - simplified environments
    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

    def setup_logging() -> None:  # type: ignore
        return None

    def log_record(**fields: Any) -> dict[str, Any]:  # type: ignore
        return fields


logger = get_logger(__name__)
settings = SandboxSettings()


def verify_dependencies() -> None:
    """Validate required helper modules and their versions.

    When ``settings.auto_install_dependencies`` is ``True`` missing or
    mismatched dependencies trigger a ``pip install`` attempt.  Offline mode is
    respected via ``settings.menace_offline_install`` and results in actionable
    error messages instead of install attempts.
    """

    import importlib
    from importlib import metadata
    from packaging.specifiers import SpecifierSet

    checks: dict[str, dict[str, Any]] = {
        "quick_fix_engine": {
            "modules": ("quick_fix_engine",),
            "install": "pip install quick_fix_engine",
        },
        "sandbox_runner.orphan_integration": {
            "modules": ("sandbox_runner.orphan_integration",),
            "install": "Install the sandbox_runner package or ensure it is on PYTHONPATH.",
        },
        "relevancy_radar": {
            "modules": ("relevancy_radar",),
            "install": "pip install relevancy_radar",
        },
        "error_logger": {
            "modules": ("error_logger",),
            "install": "Ensure the error_logger module is available.",
        },
        "telemetry_feedback": {
            "modules": ("telemetry_feedback",),
            "install": "Ensure telemetry helpers are available.",
        },
        "telemetry_backend": {
            "modules": ("telemetry_backend",),
            "install": "Ensure telemetry helpers are available.",
        },
        "torch": {
            "modules": ("torch", "pytorch"),
            "install": "pip install torch",
            "version": ">=2.0",  # Reinforcement learning components require modern torch
        },
    }

    missing: list[str] = []
    mismatched: list[str] = []

    for pkg, cfg in checks.items():
        modules = cfg["modules"]
        requirement = cfg.get("version")
        found = None
        for mod in modules:
            try:
                importlib.import_module(mod)
                found = mod
                break
            except Exception:
                logger.debug(
                    "module %s not found",
                    mod,
                    extra=log_record(dependency=mod),
                )
        if not found:
            missing.append(f"{pkg} – {cfg['install']}")
            continue
        if requirement:
            try:
                installed = metadata.version(pkg.split(".")[0])
            except Exception:
                installed = "unknown"
            if installed == "unknown" or installed not in SpecifierSet(requirement):
                mismatched.append(
                    f"{pkg} (installed {installed}, required {requirement})"
                )

    if missing or mismatched:
        lines: list[str] = []
        if missing:
            lines.append("Missing dependencies for self-improvement:\n  - " + "\n  - ".join(missing))
        if mismatched:
            lines.append("Version mismatches:\n  - " + "\n  - ".join(mismatched))
        message = "\n".join(lines)

        if missing and settings.auto_install_dependencies:
            if settings.menace_offline_install:
                message += "\nOffline mode is enabled; install the dependencies manually."
                raise RuntimeError(message)
            for pkg in missing:
                name = pkg.split(" – ", 1)[0]
                requirement = checks[name].get("version") or ""
                req = f"{name}{requirement}" if requirement else name
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(
                        "auto install failed for %s: %s",
                        name,
                        exc,
                        extra=log_record(dependency=name, error=str(exc)),
                    )
                    raise RuntimeError(message) from exc
            # Re-run to verify after installation
            original = settings.auto_install_dependencies
            settings.auto_install_dependencies = False
            try:
                verify_dependencies()
            finally:
                settings.auto_install_dependencies = original
            return

        raise RuntimeError(message)


def _lock_for(path: Path) -> FileLock:
    """Return a :class:`FileLock` for ``path`` using a ``.lock`` suffix."""

    return FileLock(str(path) + ".lock")


def _rotate_backups(path: Path, *, lock: FileLock | None = None) -> None:
    """Rotate ``path`` backups using ``.bak<N>`` suffixes under ``lock``."""

    def _do_rotate() -> None:
        count = getattr(settings, "backup_rotation_count", 3)
        backups = [path.with_suffix(path.suffix + f".bak{i}") for i in range(1, count + 1)]
        for i in range(count - 1, 0, -1):
            if backups[i - 1].exists():
                if backups[i].exists():
                    backups[i].unlink()
                os.replace(backups[i - 1], backups[i])
        if path.exists():
            os.replace(path, backups[0])

    lock = lock or _lock_for(path)
    if lock.is_locked:
        _do_rotate()
    else:
        with lock:
            _do_rotate()


def _atomic_write(
    path: Path,
    data: bytes | str,
    *,
    binary: bool = False,
    lock: FileLock | None = None,
) -> None:
    """Write ``data`` to ``path`` atomically with backup rotation under ``lock``."""

    lock = lock or _lock_for(path)

    def _do_write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if binary else "w"
        encoding = None if binary else "utf-8"
        with tempfile.NamedTemporaryFile(
            mode, encoding=encoding, dir=path.parent, delete=False
        ) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
            tmp = Path(fh.name)
        _rotate_backups(path, lock=lock)
        os.replace(tmp, path)

    if lock.is_locked:
        _do_write()
    else:
        with lock:
            _do_write()


def get_default_synergy_weights() -> dict[str, float]:
    """Return baseline synergy weights.

    The function first tries to read a ``synergy_weights`` mapping from the
    metrics snapshot referenced by ``alignment_baseline_metrics_path``.  If no
    such data is found it falls back to values defined in a
    ``SandboxSettings`` configuration file.  Hard-coded unit weights are used
    only when neither source provides a mapping.
    """

    cfg = SandboxSettings()
    weights: dict[str, float] | None = None

    metrics_path = Path(getattr(cfg, "alignment_baseline_metrics_path", ""))
    if metrics_path.is_file():
        try:
            with open(metrics_path, "r", encoding="utf-8") as fh:
                if metrics_path.suffix in {".yml", ".yaml"}:
                    data = yaml.safe_load(fh) or {}
                else:
                    data = json.load(fh)
            candidate = data.get("synergy_weights")
            if isinstance(candidate, dict):
                weights = {k: float(v) for k, v in candidate.items()}
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(
                "failed to load synergy weights from metrics snapshot %s",
                metrics_path,
                extra=log_record(path=str(metrics_path), error=str(exc)),
                exc_info=exc,
            )

    if weights is None:
        settings_path = os.getenv("SANDBOX_SETTINGS_PATH") or os.getenv(
            "SANDBOX_SETTINGS_YAML"
        )
        if settings_path and Path(settings_path).is_file():
            try:
                cfg_file = load_sandbox_settings(settings_path)
                candidate = getattr(cfg_file, "default_synergy_weights", None)
                if isinstance(candidate, dict):
                    weights = {k: float(v) for k, v in candidate.items()}
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    "failed to load synergy weights from sandbox settings %s",
                    settings_path,
                    extra=log_record(path=settings_path, error=str(exc)),
                    exc_info=exc,
                )

    if weights is None:
        candidate = getattr(cfg, "default_synergy_weights", None)
        if isinstance(candidate, dict):
            weights = {k: float(v) for k, v in candidate.items()}

    if weights is None:
        weights = {
            "roi": 1.0,
            "efficiency": 1.0,
            "resilience": 1.0,
            "antifragility": 1.0,
            "reliability": 1.0,
            "maintainability": 1.0,
            "throughput": 1.0,
        }

    return dict(weights)


def _repo_path() -> Path:
    """Return repository root from :class:`SandboxSettings`."""

    return Path(SandboxSettings().sandbox_repo_path)


def _data_dir() -> Path:
    """Return sandbox data directory from :class:`SandboxSettings`."""

    return Path(SandboxSettings().sandbox_data_dir)


def _load_initial_synergy_weights() -> None:
    """Populate ``settings`` with persisted synergy weights.

    Each weight is clamped to the inclusive range ``0``–``10``. Any
    normalisation is logged and the sanitized result written back to disk.
    """

    default_path = Path(getattr(settings, "sandbox_data_dir", ".")) / "synergy_weights.json"
    path = Path(
        getattr(
            settings,
            "synergy_weight_file",
            getattr(settings, "synergy_weights_path", default_path),
        )
    )
    weights = get_default_synergy_weights()
    changed = False
    doc = "Default synergy weights. Adjust values between 0.0 and 10.0."
    lock = _lock_for(path)
    with lock:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                doc = str(data.get("_doc", doc))
                for key, default in weights.items():
                    raw = data.get(key)
                    if isinstance(raw, (int, float)):
                        value = float(raw)
                        if not 0.0 <= value <= 10.0:
                            clipped = min(max(value, 0.0), 10.0)
                            logger.info(
                                "normalised synergy weight %s from %s to %s",
                                key,
                                value,
                                clipped,
                                extra=log_record(weight=key, original=value, normalised=clipped),
                            )
                            value = clipped
                            changed = True
                        weights[key] = value
                    else:
                        logger.warning(
                            "invalid synergy weight for %s: %r; using default %s",
                            key,
                            raw,
                            default,
                            extra=log_record(weight=key, original=repr(raw)),
                        )
                        changed = True
            else:
                logger.warning(
                    "invalid synergy weights %s",
                    path,
                    extra=log_record(path=str(path)),
                )
                changed = True
        except FileNotFoundError:
            logger.info(
                "synergy weights file %s missing; creating defaults",
                path,
                extra=log_record(path=str(path)),
            )
            changed = True
        except OSError as exc:
            logger.warning(
                "I/O error loading synergy weights %s",
                path,
                extra=log_record(path=str(path), error=str(exc)),
                exc_info=exc,
            )
            changed = True
        except ValueError as exc:
            logger.warning(
                "invalid synergy weights %s",
                path,
                extra=log_record(path=str(path), error=str(exc)),
                exc_info=exc,
            )
            changed = True

        if changed:
            payload = {"_doc": doc, **weights}
            try:
                _atomic_write(path, json.dumps(payload, indent=2), lock=lock)
                logger.info(
                    "persisted sanitized synergy weights %s",
                    path,
                    extra=log_record(path=str(path)),
                )
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning(
                    "failed to write sanitized synergy weights %s",
                    path,
                    extra=log_record(path=str(path), error=str(exc)),
                    exc_info=exc,
                )

    settings.synergy_weight_roi = weights["roi"]
    settings.synergy_weight_efficiency = weights["efficiency"]
    settings.synergy_weight_resilience = weights["resilience"]
    settings.synergy_weight_antifragility = weights["antifragility"]
    settings.synergy_weight_reliability = weights["reliability"]
    settings.synergy_weight_maintainability = weights["maintainability"]
    settings.synergy_weight_throughput = weights["throughput"]
    settings.synergy_weight_file = str(path)


def reload_synergy_weights() -> None:
    """Reload synergy weights from disk for dynamic tuning."""

    _load_initial_synergy_weights()


def init_self_improvement(new_settings: SandboxSettings | None = None) -> SandboxSettings:
    """Initialise the self-improvement subsystem explicitly."""

    global settings
    settings = new_settings or load_sandbox_settings()
    verify_dependencies()
    initialize_autonomous_sandbox(settings)
    if getattr(settings, "sandbox_central_logging", False):
        setup_logging()
    reload_synergy_weights()
    try:
        from . import meta_planning

        meta_planning.reload_settings(settings)
    except Exception as exc:  # pragma: no cover - best effort
        message = "failed to reload meta_planning settings"
        logging.getLogger(__name__).exception(  # ensure real logging backend
            message, extra=log_record(error=str(exc))
        )
        raise RuntimeError(message) from exc
    return settings


__all__ = [
    "init_self_improvement",
    "settings",
    "_repo_path",
    "_data_dir",
    "_atomic_write",
    "get_default_synergy_weights",
    "reload_synergy_weights",
    "verify_dependencies",
]
