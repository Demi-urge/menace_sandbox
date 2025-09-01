"""Helpers for the self-improvement subsystem.

The routines expect auxiliary packages to be installed:

* ``quick_fix_engine`` – generates corrective patches.
* ``sandbox_runner.orphan_integration`` – reintroduces orphaned modules.

Missing dependencies raise :class:`RuntimeError` with guidance on how to
install them.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

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
    """Ensure optional helper packages for self-improvement are installed."""

    try:
        import quick_fix_engine  # type: ignore # noqa: F401
    except Exception as exc:  # pragma: no cover - import guidance
        raise RuntimeError(
            "quick_fix_engine is required for self-improvement. "
            "Install it with 'pip install quick_fix_engine'."
        ) from exc

    try:
        import sandbox_runner.orphan_integration  # type: ignore # noqa: F401
    except Exception as exc:  # pragma: no cover - import guidance
        raise RuntimeError(
            "sandbox_runner.orphan_integration is required for self-improvement. "
            "Install the sandbox_runner package or ensure it is on PYTHONPATH."
        ) from exc


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
    """Return default synergy weights from :class:`SandboxSettings`.

    A fresh :class:`SandboxSettings` instance is created on each call so that
    changes to configuration or environment variables are reflected in the
    returned defaults.
    """

    cfg = SandboxSettings()
    return dict(
        getattr(
            cfg,
            "default_synergy_weights",
            {
                "roi": 1.0,
                "efficiency": 1.0,
                "resilience": 1.0,
                "antifragility": 1.0,
                "reliability": 1.0,
                "maintainability": 1.0,
                "throughput": 1.0,
            },
        )
    )


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
