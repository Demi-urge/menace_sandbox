#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies.

Run ``python scripts/bootstrap_env.py`` to install required tooling and
configuration.  Pass ``--skip-stripe-router`` to bypass the Stripe router
startup verification when working offline or without Stripe credentials.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """Raised when the environment bootstrap process cannot proceed."""


def _coerce_log_level(value: str | int | None) -> int:
    """Translate user provided log level into the numeric representation."""

    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip().upper()
        if not candidate:
            return logging.INFO
        if candidate.isdigit():
            return int(candidate)
        level = logging.getLevelName(candidate)
        if isinstance(level, int):
            return level
    raise BootstrapError(f"Unsupported log level: {value!r}")


@dataclass(frozen=True)
class BootstrapConfig:
    """Normalized configuration derived from command-line flags."""

    skip_stripe_router: bool = False
    env_file: Path | None = None
    log_level: int = logging.INFO

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "BootstrapConfig":
        env_path = namespace.env_file
        if isinstance(env_path, Path):
            env_path = env_path.expanduser()
        log_level = _coerce_log_level(namespace.log_level)
        return cls(
            skip_stripe_router=namespace.skip_stripe_router,
            env_file=env_path,
            log_level=log_level,
        )

    def resolved_env_file(self) -> Path | None:
        if self.env_file is None:
            return None
        try:
            return self.env_file.resolve()
        except OSError as exc:  # pragma: no cover - environment specific
            raise BootstrapError(
                f"Unable to resolve environment file '{self.env_file}'"
            ) from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stripe-router",
        action="store_true",
        help=(
            "Bypass the Stripe router startup verification. Useful when Stripe "
            "credentials are unavailable during local bootstraps."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional path to the environment file that should receive generated "
            "defaults.  When omitted the bootstrap process falls back to the "
            "standard discovery rules in bootstrap_defaults."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help=(
            "Logging level for bootstrap diagnostics. Accepts either a standard "
            "logging name (DEBUG, INFO, WARNING, ERROR, CRITICAL) or the "
            "corresponding numeric value."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _apply_environment(overrides: Mapping[str, str]) -> None:
    for key, value in overrides.items():
        os.environ[key] = value


def _ensure_windows_compatibility() -> None:
    """Augment environment defaults with Windows specific safeguards."""

    if os.name != "nt":  # pragma: no cover - exercised via integration tests
        return

    scripts_dirs: list[Path] = []
    executable = Path(sys.executable)
    candidates = [
        executable.with_name("Scripts"),
        executable.parent / "Scripts",
    ]
    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        candidates.append(Path(venv_root) / "Scripts")

    existing_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in existing_path.split(os.pathsep) if entry]
    seen: dict[str, str] = {}
    ordered_entries: list[str] = []
    for entry in path_entries:
        key = entry.lower()
        if key in seen:
            continue
        seen[key] = entry
        ordered_entries.append(entry)

    updated = False
    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        key = str(candidate)
        lookup = key.lower()
        if lookup not in seen:
            seen[lookup] = key
            ordered_entries.insert(0, key)
            scripts_dirs.append(candidate)
            updated = True

    if updated:
        new_path = os.pathsep.join(ordered_entries)
        os.environ["PATH"] = new_path
        LOGGER.info(
            "Ensured Windows PATH contains Scripts directories: %s",
            ", ".join(str(path) for path in scripts_dirs),
        )

    os.environ.setdefault("PYTHONUTF8", "1")
    pathext = os.environ.get("PATHEXT")
    if pathext:
        required_exts = {".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW"}
        current = [ext.strip() for ext in pathext.split(os.pathsep) if ext]
        normalized = {ext.upper() for ext in current}
        if not required_exts.issubset(normalized):
            updated_exts = current[:]
            for ext in sorted(required_exts):
                if ext.upper() not in normalized:
                    updated_exts.append(ext)
            os.environ["PATHEXT"] = os.pathsep.join(updated_exts)
            LOGGER.info("Augmented PATHEXT with Python aware extensions: %s", ". ".join(sorted(required_exts)))
    else:
        os.environ["PATHEXT"] = os.pathsep.join([".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW"])
        LOGGER.info("Initialized PATHEXT to include Python executables")


def _prepare_environment(config: BootstrapConfig) -> Path | None:
    resolved_env_file = config.resolved_env_file()
    defaults = {
        "MENACE_ALLOW_MISSING_HF_TOKEN": "1",
        "MENACE_NON_INTERACTIVE": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    overrides: dict[str, str] = {
        "MENACE_SAFE": "0",
        "MENACE_SUPPRESS_PROMETHEUS_FALLBACK_NOTICE": "1",
    }
    if config.skip_stripe_router:
        overrides["MENACE_SKIP_STRIPE_ROUTER"] = "1"
    if resolved_env_file is not None:
        overrides["MENACE_ENV_FILE"] = str(resolved_env_file)
    _apply_environment(overrides)
    _ensure_windows_compatibility()
    return resolved_env_file


def _run_bootstrap(config: BootstrapConfig) -> None:
    resolved_env_file = _prepare_environment(config)

    from menace.bootstrap_policy import PolicyLoader
    from menace.environment_bootstrap import EnvironmentBootstrapper
    import startup_checks
    from startup_checks import run_startup_checks
    from menace.bootstrap_defaults import ensure_bootstrap_defaults

    created, env_file = ensure_bootstrap_defaults(
        startup_checks.REQUIRED_VARS,
        repo_root=_REPO_ROOT,
        env_file=resolved_env_file,
    )
    if created:
        LOGGER.info("Persisted generated defaults to %s", env_file)

    loader = PolicyLoader()
    auto_install = startup_checks.auto_install_enabled()
    env_requested = os.getenv("MENACE_BOOTSTRAP_PROFILE")
    requested = env_requested or ("minimal" if not auto_install else None)
    policy = loader.resolve(
        requested=requested,
        auto_install_enabled=auto_install,
    )
    run_startup_checks(skip_stripe_router=config.skip_stripe_router, policy=policy)
    EnvironmentBootstrapper(policy=policy).bootstrap()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = BootstrapConfig.from_namespace(args)
    _configure_logging(config.log_level)
    try:
        _run_bootstrap(config)
    except BootstrapError as exc:
        LOGGER.error("bootstrap aborted: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        LOGGER.warning("Bootstrap interrupted by user")
        raise SystemExit(130)
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.exception("Unexpected error during bootstrap")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
