from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

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


def _rotate_backups(path: Path) -> None:
    """Rotate ``path`` backups using ``.bak<N>`` suffixes."""

    count = getattr(settings, "backup_rotation_count", 3)
    backups = [path.with_suffix(path.suffix + f".bak{i}") for i in range(1, count + 1)]
    for i in range(count - 1, 0, -1):
        if backups[i - 1].exists():
            if backups[i].exists():
                backups[i].unlink()
            os.replace(backups[i - 1], backups[i])
    if path.exists():
        os.replace(path, backups[0])


def _atomic_write(path: Path, data: bytes | str, *, binary: bool = False) -> None:
    """Write ``data`` to ``path`` atomically with backup rotation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    with tempfile.NamedTemporaryFile(mode, encoding=encoding, dir=path.parent, delete=False) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
        tmp = Path(fh.name)
    _rotate_backups(path)
    os.replace(tmp, path)


DEFAULT_SYNERGY_WEIGHTS: dict[str, float] = dict(
    getattr(
        settings,
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
    """Populate ``settings`` with persisted synergy weights."""

    default_path = Path(getattr(settings, "sandbox_data_dir", ".")) / "synergy_weights.json"
    path = Path(
        getattr(
            settings,
            "synergy_weight_file",
            getattr(settings, "synergy_weights_path", default_path),
        )
    )
    weights = DEFAULT_SYNERGY_WEIGHTS.copy()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        valid = isinstance(data, dict) and all(
            k in data and isinstance(data.get(k), (int, float)) for k in weights
        )
        if valid:
            for k in weights:
                weights[k] = float(data[k])
    except FileNotFoundError:
        logger.info(
            "synergy weights file %s missing; creating defaults",
            path,
            extra=log_record(path=str(path)),
        )
        payload = {
            "_doc": "Default synergy weights. Adjust values between 0.0 and 10.0.",
            **weights,
        }
        try:
            _atomic_write(path, json.dumps(payload, indent=2))
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning(
                "failed to write default synergy weights %s",
                path,
                extra=log_record(path=str(path), error=str(exc)),
                exc_info=exc,
            )
    except OSError as exc:
        logger.warning(
            "I/O error loading synergy weights %s",
            path,
            extra=log_record(path=str(path), error=str(exc)),
            exc_info=exc,
        )
    except ValueError as exc:
        logger.warning(
            "invalid synergy weights %s",
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


def init_self_improvement(new_settings: SandboxSettings | None = None) -> SandboxSettings:
    """Initialise the self-improvement subsystem explicitly."""

    global settings
    settings = new_settings or load_sandbox_settings()
    initialize_autonomous_sandbox(settings)
    if getattr(settings, "sandbox_central_logging", False):
        setup_logging()
    _load_initial_synergy_weights()
    return settings


__all__ = [
    "init_self_improvement",
    "settings",
    "_repo_path",
    "_data_dir",
    "_atomic_write",
    "DEFAULT_SYNERGY_WEIGHTS",
]

