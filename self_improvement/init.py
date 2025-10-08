"""Helpers for the self-improvement subsystem.

The routines expect auxiliary packages to be installed:

* ``quick_fix_engine`` >=1.0 – generates corrective patches.
* ``sandbox_runner`` >=1.0 – provides sandbox orchestration helpers.
* ``neurosales`` – supplies predictive sales models.

Missing dependencies raise :class:`RuntimeError` with guidance on how to
install or upgrade them.  When the ``auto_install`` flag is enabled the
verifier attempts to install missing or mismatched packages automatically and
only raises an error if those installations fail.  The
``init_self_improvement`` routine enables this behaviour automatically when the
process is running without an interactive terminal, allowing unattended
deployments to bootstrap required packages on first run.

Configuration is provided via :class:`sandbox_settings.SandboxSettings` with
notable options:

* ``orphan_retry_attempts`` – retry attempts for orphan integration hooks.
* ``orphan_retry_delay`` – delay between retries for orphan integration hooks.
"""

import json
import logging
import os
import tempfile
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from filelock import FileLock

from sandbox_settings import SandboxSettings, load_sandbox_settings
from sandbox_runner.bootstrap import initialize_autonomous_sandbox
try:  # pragma: no cover - prefer absolute imports when running from repo root
    from metrics_exporter import self_improvement_failure_total
except ImportError:  # pragma: no cover - fallback to package-relative import
    from menace_sandbox.metrics_exporter import self_improvement_failure_total  # type: ignore

try:  # pragma: no cover - prefer absolute imports when running from repo root
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback to package-relative import
    from menace_sandbox.dynamic_path_router import resolve_path

try:
    from logging_utils import get_logger, setup_logging, log_record  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - fallback to package layout
    try:
        from menace_sandbox.logging_utils import get_logger, setup_logging, log_record
    except (ImportError, AttributeError) as exc:  # pragma: no cover - simplified environments
        logging.getLogger(__name__).warning(
            "logging utils unavailable",
            exc_info=exc,
            extra={"component": __name__},
        )
        self_improvement_failure_total.labels(reason="logging_utils_import").inc()

        def get_logger(name: str) -> logging.Logger:  # type: ignore
            return logging.getLogger(name)

        def setup_logging() -> None:  # type: ignore
            return None

        _RESERVED_LOG_ATTRS = set(
            logging.LogRecord(
                name="", level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None
            ).__dict__
        )
        _RESERVED_LOG_ATTRS.update({"message", "asctime"})

        def _safe_key(key: str, existing: dict[str, Any]) -> str:
            if key not in _RESERVED_LOG_ATTRS and key not in existing:
                return key

            base = f"extra_{key}"
            if base not in _RESERVED_LOG_ATTRS and base not in existing:
                return base

            suffix = 1
            candidate = f"{base}_{suffix}"
            while candidate in _RESERVED_LOG_ATTRS or candidate in existing:
                suffix += 1
                candidate = f"{base}_{suffix}"
            return candidate

        def log_record(**fields: Any) -> dict[str, Any]:  # type: ignore
            safe: dict[str, Any] = {}
            for key, value in fields.items():
                if value is None:
                    continue
                safe[_safe_key(key, safe)] = value
            return safe


logger = get_logger(__name__)
settings = SandboxSettings()


def verify_dependencies(*, auto_install: bool = False) -> None:
    """Validate required helper modules and their versions.

    Parameters
    ----------
    auto_install:
        When ``True``, missing or mismatched packages are installed via ``pip``
        before failing.  On installation failure the original error message is
        raised.

    Missing or mismatched packages raise a :class:`RuntimeError` listing the
    required installation commands when ``auto_install`` is disabled or when an
    installation attempt fails.
    """

    import importlib
    from importlib import metadata
    from packaging.specifiers import SpecifierSet
    import shlex
    import subprocess

    neurosales_cfg: dict[str, Any] = {
        "modules": ("neurosales",),
        "install": "pip install neurosales",
    }
    try:  # capture version if package metadata is published
        neurosales_cfg["version"] = f">={metadata.version('neurosales')}"
    except Exception as exc:
        logger.debug(
            "failed to read neurosales metadata",
            extra=log_record(error=str(exc)),
        )
        neurosales_cfg["version"] = None

    checks: dict[str, dict[str, Any]] = {
        "quick_fix_engine": {
            "modules": ("quick_fix_engine",),
            "install": "pip install quick_fix_engine",
            "version": ">=1.0",
        },
        "sandbox_runner": {
            "modules": ("sandbox_runner",),
            "install": "pip install sandbox_runner",
            "version": ">=1.0",
        },
        "sandbox_runner.orphan_integration": {
            "modules": ("sandbox_runner.orphan_integration",),
            "install": "pip install sandbox_runner",
        },
        "sandbox_runner.environment": {
            "modules": ("sandbox_runner.environment",),
            "install": "pip install sandbox_runner",
        },
        "neurosales": neurosales_cfg,
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

    missing: list[tuple[str, str]] = []  # (message, install cmd)
    mismatched: list[tuple[str, str]] = []

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
            missing.append((f"{pkg} – {cfg['install']}", cfg["install"]))
            continue
        if requirement:
            try:
                installed = metadata.version(pkg.split(".")[0])
            except Exception:
                installed = "unknown"
            if installed == "unknown" or installed not in SpecifierSet(requirement):
                cmd = cfg["install"]
                if cmd.startswith("pip install") and "--upgrade" not in cmd:
                    cmd += " --upgrade"
                mismatched.append(
                    (
                        f"{pkg} (installed {installed}, required {requirement}) – {cmd}",
                        cmd,
                    )
                )
    if missing or mismatched:
        if auto_install:
            for msg, cmd in missing + mismatched:
                logger.info("auto-installing dependency", extra=log_record(command=cmd))
                try:
                    subprocess.run(shlex.split(cmd), check=True)
                except Exception as exc:
                    logger.error(
                        "dependency installation failed",
                        extra=log_record(command=cmd, error=str(exc)),
                    )
                    _raise_dep_error(missing, mismatched)
            # verify again without auto-install to surface any remaining issues
            verify_dependencies(auto_install=False)
            return

        _raise_dep_error(missing, mismatched)


def _raise_dep_error(
    missing: list[tuple[str, str]], mismatched: list[tuple[str, str]]
) -> None:
    lines: list[str] = []
    if missing:
        lines.append(
            "Missing dependencies for self-improvement:\n  - "
            + "\n  - ".join(m for m, _ in missing)
            + "\nInstall the missing packages before launching."
        )
    if mismatched:
        lines.append(
            "Version mismatches:\n  - " + "\n  - ".join(m for m, _ in mismatched)
        )
    message = "\n".join(lines)
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


@dataclass
class SynergyWeights:
    """Schema for synergy weight configuration."""

    roi: float
    efficiency: float
    resilience: float
    antifragility: float
    reliability: float
    maintainability: float
    throughput: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynergyWeights":
        """Validate ``data`` and construct :class:`SynergyWeights`."""

        required = {f.name for f in fields(cls)}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(
                "missing synergy weight(s): " + ", ".join(missing)
            )
        values: dict[str, float] = {}
        for f in fields(cls):
            raw = data[f.name]
            if not isinstance(raw, (int, float)):
                raise ValueError(
                    f"synergy weight '{f.name}' must be a number, got {raw!r}"
                )
            values[f.name] = float(raw)
        return cls(**values)


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

    baseline_path = getattr(cfg, "alignment_baseline_metrics_path", "")
    try:
        metrics_path = resolve_path(str(baseline_path))
    except FileNotFoundError:
        metrics_path = Path(str(baseline_path))
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
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            logger.debug(
                "failed to load synergy weights from metrics snapshot %s",
                metrics_path,
                extra=log_record(path=str(metrics_path), error=str(exc)),
                exc_info=exc,
            )
            self_improvement_failure_total.labels(reason="synergy_weights_snapshot_load").inc()

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
            except (OSError, yaml.YAMLError, AttributeError, RuntimeError) as exc:
                logger.debug(
                    "failed to load synergy weights from sandbox settings %s",
                    settings_path,
                    extra=log_record(path=settings_path, error=str(exc)),
                    exc_info=exc,
                )
                self_improvement_failure_total.labels(reason="synergy_settings_load").inc()

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

    default_path = resolve_path(getattr(settings, "sandbox_data_dir", ".")) / "synergy_weights.json"
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
            if not isinstance(data, dict):
                raise ValueError("synergy weights file must contain an object")
            doc = str(data.get("_doc", doc))
            loaded = SynergyWeights.from_dict(data)
            for key, value in loaded.__dict__.items():
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
            raise ValueError(f"invalid synergy weights {path}: {exc}") from exc

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
    """Initialise the self-improvement subsystem explicitly.

    The function enables automatic dependency installation when running
    without an interactive terminal unless explicitly disabled via
    ``SandboxSettings.auto_install_dependencies``.
    """

    global settings
    settings = new_settings or load_sandbox_settings()
    auto_install = settings.auto_install_dependencies or not sys.stdin.isatty()
    verify_dependencies(auto_install=auto_install)
    initialize_autonomous_sandbox(settings)
    if getattr(settings, "sandbox_central_logging", False):
        setup_logging()
    reload_synergy_weights()
    try:
        from . import meta_planning

        meta_planning.reload_settings(settings)
    except Exception as exc:  # pragma: no cover - best effort
        logging.getLogger(__name__).exception(
            "failed to reload meta_planning settings",
            extra=log_record(error=str(exc), component="meta_planning"),
        )
        self_improvement_failure_total.labels(reason="meta_planning_reload").inc()
        raise
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
