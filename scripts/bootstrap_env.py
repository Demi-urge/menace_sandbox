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
import re
import sys
import sysconfig
from functools import lru_cache
import ntpath
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalize_sys_path_entry(entry: object) -> str | None:
    """Return a normalized representation of *entry* suitable for comparison."""

    if isinstance(entry, os.PathLike):
        entry = os.fspath(entry)
    if isinstance(entry, str):
        try:
            return os.path.normcase(os.path.abspath(entry))
        except OSError:
            return os.path.normcase(entry)
    return None


def _ensure_repo_root_on_path(repo_root: Path) -> None:
    """Inject *repo_root* into ``sys.path`` while avoiding duplicates."""

    target = os.path.normcase(str(repo_root.resolve()))
    normalized_entries: list[str | None] = [
        _normalize_sys_path_entry(entry) for entry in sys.path
    ]

    canonical = str(repo_root)

    try:
        existing_index = normalized_entries.index(target)  # type: ignore[arg-type]
    except ValueError:
        sys.path.insert(0, canonical)
        return

    if existing_index == 0:
        sys.path[0] = canonical
        return

    sys.path.pop(existing_index)
    sys.path.insert(0, canonical)


_ensure_repo_root_on_path(_REPO_ROOT)


LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """Raised when the environment bootstrap process cannot proceed."""


_WINDOWS_ENV_VAR_PATTERN = re.compile(r"%(?P<name>[A-Za-z0-9_]+)%")
_POSIX_ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<simple>[A-Za-z_][A-Za-z0-9_]*))"
)


def _resolve_windows_env_fallback(name: str) -> str | None:
    """Provide cross-platform fallbacks for common Windows placeholders."""

    normalized = name.upper()
    try:
        home = Path.home()
    except OSError:
        home = None

    if home is None:
        return None

    home_str = os.fspath(home)

    if normalized == "USERPROFILE":
        return home_str

    if normalized == "LOCALAPPDATA":
        return os.fspath(home / "AppData" / "Local")

    if normalized == "APPDATA":
        return os.fspath(home / "AppData" / "Roaming")

    if normalized in {"TEMP", "TMP"}:
        return os.fspath(home / "AppData" / "Local" / "Temp")

    if normalized == "HOMEPATH":
        drive = home.drive
        if drive:
            suffix = home_str[len(drive) :]
            return suffix or os.sep
        return home_str

    if normalized == "HOMEDRIVE":
        drive = home.drive
        return drive or None

    return None


def _collect_unresolved_env_tokens(value: str) -> set[str]:
    """Return unresolved environment variable placeholders found in *value*."""

    unresolved: set[str] = set()
    for match in _WINDOWS_ENV_VAR_PATTERN.finditer(value):
        unresolved.add(match.group(0))
    for match in _POSIX_ENV_VAR_PATTERN.finditer(value):
        token = match.group(0)
        if token == "$$":
            continue
        unresolved.add(token)
    return unresolved


def _expand_environment_path(value: str) -> str:
    """Expand environment variables in *value* across platforms."""

    expanded = os.path.expandvars(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        candidates = [name]
        if not _is_windows():
            for alias in (name.upper(), name.lower()):
                if alias not in candidates:
                    candidates.append(alias)
        for candidate in candidates:
            if candidate in os.environ:
                return os.environ[candidate]
        fallback = _resolve_windows_env_fallback(name)
        if fallback:
            return fallback
        return match.group(0)

    expanded = _WINDOWS_ENV_VAR_PATTERN.sub(replace, expanded)

    unresolved_tokens = _collect_unresolved_env_tokens(expanded)
    if unresolved_tokens:
        tokens = ", ".join(sorted(unresolved_tokens))
        raise BootstrapError(
            "Unable to expand environment variables in path "
            f"{value!r}; unresolved placeholder(s): {tokens}. "
            "Define the missing variables or escape literal percent/dollar "
            "symbols by doubling them."
        )

    return expanded


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
        env_path_raw = namespace.env_file
        env_path: Path | None
        if env_path_raw is None:
            env_path = None
        else:
            path_string = os.fspath(env_path_raw)
            expanded = _expand_environment_path(path_string)
            # ``expandvars`` leaves values untouched when an environment
            # variable cannot be resolved; avoid introducing empty segments.
            sanitized = expanded.strip()
            if sanitized:
                env_path = Path(sanitized).expanduser()
            else:
                env_path = None
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
            resolved = self.env_file.resolve(strict=False)
        except OSError as exc:  # pragma: no cover - environment specific
            raise BootstrapError(
                f"Unable to resolve environment file '{self.env_file}'"
            ) from exc

        if resolved.exists() and resolved.is_dir():
            raise BootstrapError(
                f"Environment file path '{resolved}' refers to a directory"
            )

        return resolved


def _ensure_parent_directory(path: Path | None) -> None:
    """Create the parent directory for *path* when it does not yet exist."""

    if path is None:
        return

    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - environment specific
        raise BootstrapError(
            f"Unable to create parent directory '{parent}' for '{path}'"
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


def _iter_windows_script_candidates(executable: Path) -> Iterable[Path]:
    """Yield plausible ``Scripts`` directories for the active interpreter."""

    scripts_dir = executable.with_name("Scripts")
    yield scripts_dir
    yield executable.parent / "Scripts"

    # ``sysconfig`` provides a reliable view into the interpreter layout even
    # when ``sys.executable`` points at a shim such as ``py.exe``.
    try:
        scripts_path = Path(sysconfig.get_path("scripts"))
    except (KeyError, TypeError, ValueError):
        scripts_path = None
    if scripts_path:
        yield scripts_path

    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        if prefix:
            yield Path(prefix) / "Scripts"

    venv_root = os.environ.get("VIRTUAL_ENV")
    if venv_root:
        yield Path(venv_root) / "Scripts"


def _is_windows() -> bool:
    return os.name == "nt"


@lru_cache(maxsize=None)
def _windows_path_normalizer() -> Callable[[str], str]:
    """Return a callable that normalizes Windows paths for comparison."""

    def _normalize(value: str) -> str:
        collapsed = ntpath.normcase(ntpath.normpath(value))
        collapsed = collapsed.rstrip("\\/")
        return collapsed

    return _normalize


def _strip_windows_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _is_quoted_windows_value(value: str) -> bool:
    value = value.strip()
    return len(value) >= 2 and value[0] == value[-1] == '"'


def _needs_windows_path_quotes(value: str) -> bool:
    return any(symbol in value for symbol in (" ", ";", "(", ")", "&"))


def _format_windows_path_entry(value: str) -> str:
    trimmed = _strip_windows_quotes(value)
    if not trimmed:
        return trimmed
    if _needs_windows_path_quotes(trimmed):
        return f'"{trimmed}"'
    return trimmed


def _score_windows_entry(entry: str) -> tuple[int, int, int, int]:
    stripped = _strip_windows_quotes(entry)
    try:
        exists = Path(stripped).exists()
    except OSError:
        exists = False
    quotes_mismatch = int(_needs_windows_path_quotes(stripped) != _is_quoted_windows_value(entry))
    trailing_sep = int(stripped.endswith(("\\", "/")))
    return (
        0 if exists else 1,
        quotes_mismatch,
        trailing_sep,
        len(entry),
    )


def _choose_preferred_path_entry(
    existing: str,
    candidate: str,
    normalizer: Callable[[str], str],
) -> str:
    if existing == candidate:
        return existing

    existing_core = _strip_windows_quotes(existing)
    candidate_core = _strip_windows_quotes(candidate)
    existing_normalized = normalizer(existing_core)
    candidate_normalized = normalizer(candidate_core)

    if existing_normalized == candidate_normalized:
        normalized_candidate = _format_windows_path_entry(candidate)
        if normalized_candidate != candidate:
            candidate = normalized_candidate
            candidate_core = _strip_windows_quotes(candidate)
        existing_requires_quotes = _needs_windows_path_quotes(existing_core)
        candidate_requires_quotes = _needs_windows_path_quotes(candidate_core)
        existing_is_quoted = _is_quoted_windows_value(existing)
        candidate_is_quoted = _is_quoted_windows_value(candidate)
        if existing_requires_quotes and not existing_is_quoted and candidate_requires_quotes:
            return candidate
        if candidate_requires_quotes and not candidate_is_quoted and existing_requires_quotes:
            candidate = _format_windows_path_entry(candidate)
            candidate_core = _strip_windows_quotes(candidate)
        if candidate_requires_quotes and not existing_requires_quotes:
            return candidate
        if existing_requires_quotes and not candidate_requires_quotes:
            return _format_windows_path_entry(existing)
        if existing_core != candidate_core:
            return candidate
        return existing

    existing_score = _score_windows_entry(existing)
    candidate_score = _score_windows_entry(candidate)
    if existing_score == candidate_score and existing_core != candidate_core:
        return candidate
    return existing if existing_score <= candidate_score else candidate


def _gather_existing_path_entries() -> tuple[list[str], dict[str, str], bool]:
    """Collect the current PATH entries de-duplicated by Windows semantics."""

    raw_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    separator = os.pathsep
    if ";" in raw_path and separator != ";":
        parts: Iterable[str] = raw_path.split(";")
    else:
        parts = raw_path.split(separator)
    entries = [entry.strip() for entry in parts if entry and entry.strip()]
    normalizer = _windows_path_normalizer()
    seen: dict[str, str] = {}
    ordered: list[str] = []
    deduplicated = False
    for entry in entries:
        try:
            normalized = normalizer(_strip_windows_quotes(entry))
        except (TypeError, ValueError):
            continue
        existing = seen.get(normalized)
        if existing is not None:
            preferred = _choose_preferred_path_entry(existing, entry, normalizer)
            if preferred != existing:
                try:
                    index = ordered.index(existing)
                except ValueError:
                    ordered.append(preferred)
                else:
                    ordered[index] = preferred
                seen[normalized] = preferred
            deduplicated = True
            continue
        seen[normalized] = entry
        ordered.append(entry)
    return ordered, seen, deduplicated


def _set_windows_path(value: str) -> None:
    os.environ["PATH"] = value
    os.environ["Path"] = value


def _ensure_windows_compatibility() -> None:
    """Augment environment defaults with Windows specific safeguards."""

    if not _is_windows():  # pragma: no cover - exercised via integration tests
        return

    scripts_dirs: list[Path] = []
    normalized_updates: list[tuple[str, str]] = []
    executable = Path(sys.executable)
    candidates = list(_iter_windows_script_candidates(executable))

    ordered_entries, seen, deduplicated = _gather_existing_path_entries()
    normalizer = _windows_path_normalizer()

    updated = False
    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate_resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if not candidate_resolved.exists():
            continue
        key = _format_windows_path_entry(str(candidate_resolved))
        normalized_key = normalizer(_strip_windows_quotes(key))
        existing_entry = seen.get(normalized_key)
        if existing_entry is None:
            seen[normalized_key] = key
            ordered_entries.insert(0, key)
            scripts_dirs.append(candidate_resolved)
            updated = True
            continue

        preferred = _choose_preferred_path_entry(existing_entry, key, normalizer)
        if preferred == existing_entry:
            continue
        try:
            index = ordered_entries.index(existing_entry)
        except ValueError:
            ordered_entries.insert(0, preferred)
        else:
            ordered_entries[index] = preferred
        seen[normalized_key] = preferred
        normalized_updates.append((existing_entry, preferred))
        updated = True

    if updated or deduplicated:
        new_path = os.pathsep.join(ordered_entries)
        _set_windows_path(new_path)
        if updated:
            if scripts_dirs:
                LOGGER.info(
                    "Ensured Windows PATH contains Scripts directories: %s",
                    ", ".join(str(path) for path in scripts_dirs),
                )
            if normalized_updates:
                LOGGER.info(
                    "Normalized existing Windows PATH entries: %s",
                    ", ".join(
                        f"{original!r} -> {updated_entry!r}"
                        for original, updated_entry in normalized_updates
                    ),
                )
        elif deduplicated:
            LOGGER.info("Normalized Windows PATH by removing duplicate entries")

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    pathext = os.environ.get("PATHEXT")
    required_exts = (".COM", ".EXE", ".BAT", ".CMD", ".PY", ".PYW")
    if pathext:
        current = [ext.strip() for ext in pathext.split(os.pathsep) if ext]
        normalized = {ext.upper() for ext in current}
        if not set(required_exts).issubset(normalized):
            updated_exts = list(dict.fromkeys(ext.upper() for ext in current))
            for ext in required_exts:
                if ext.upper() not in normalized:
                    updated_exts.append(ext)
            os.environ["PATHEXT"] = os.pathsep.join(updated_exts)
            LOGGER.info(
                "Augmented PATHEXT with Python aware extensions: %s",
                ", ".join(required_exts),
            )
    else:
        os.environ["PATHEXT"] = os.pathsep.join(required_exts)
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
    _ensure_parent_directory(resolved_env_file)
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
