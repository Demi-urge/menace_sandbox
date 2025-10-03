#!/usr/bin/env python3
"""Bootstrap the Menace environment and verify dependencies.

Run ``python scripts/bootstrap_env.py`` to install required tooling and
configuration.  Pass ``--skip-stripe-router`` to bypass the Stripe router
startup verification when working offline or without Stripe credentials.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from functools import lru_cache
import ntpath
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterable, Mapping, Sequence, Literal

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


_DOCKER_SKIP_ENV = "MENACE_BOOTSTRAP_SKIP_DOCKER_CHECK"
_DOCKER_REQUIRE_ENV = "MENACE_REQUIRE_DOCKER"
_DOCKER_ASSUME_NO_ENV = "MENACE_BOOTSTRAP_ASSUME_NO_DOCKER"


_WINDOWS_ENV_VAR_PATTERN = re.compile(r"%(?P<name>[A-Za-z0-9_]+)%")
_POSIX_ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<simple>[A-Za-z_][A-Za-z0-9_]*))"
)


_BACKOFF_INTERVAL_PATTERN = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ms|msec|milliseconds|s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)?",
    flags=re.IGNORECASE,
)


def _coalesce_iterable(values: Iterable[str]) -> list[str]:
    """Return *values* with duplicates removed while preserving ordering."""

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value)
    return unique


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


def _iter_windows_docker_directories() -> Iterable[Path]:
    """Yield directories that commonly contain Docker Desktop CLIs on Windows."""

    candidates: list[Path] = []
    for env_var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    if not candidates:
        candidates.extend(
            Path(path)
            for path in (r"C:\\Program Files", r"C:\\Program Files (x86)")
        )

    seen: set[str] = set()
    for root in candidates:
        target = root / "Docker" / "Docker" / "resources" / "bin"
        key = os.path.normcase(str(target))
        if key in seen:
            continue
        seen.add(key)
        yield target


def _convert_windows_path_to_wsl(path: Path) -> Path | None:
    """Return the WSL representation of ``path`` if it targets a Windows drive."""

    try:
        windows_path = PureWindowsPath(path)
    except Exception:  # pragma: no cover - defensive
        return None

    drive = windows_path.drive.rstrip(":")
    if not drive:
        return None

    segments = list(windows_path.parts[1:])
    if not segments:
        return None

    converted = Path("/mnt") / drive.lower()
    for segment in segments:
        converted /= segment
    return converted


def _iter_wsl_docker_directories() -> Iterable[Path]:
    """Yield Docker CLI directories exposed via Windows when running inside WSL."""

    for candidate in _iter_windows_docker_directories():
        converted = _convert_windows_path_to_wsl(candidate)
        if converted is not None:
            yield converted


def _is_windows() -> bool:
    return os.name == "nt"


@lru_cache(maxsize=None)
def _is_wsl() -> bool:
    """Return ``True`` when executing inside the Windows Subsystem for Linux."""

    if _is_windows():
        return False

    indicators = []
    for probe in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            with open(probe, "r", encoding="utf-8", errors="ignore") as fh:
                indicators.append(fh.read())
        except OSError:
            continue

    signature = "\n".join(indicators)
    return "Microsoft" in signature or "WSL" in signature


def _detect_container_indicators() -> tuple[bool, str | None, tuple[str, ...]]:
    """Return containerisation hints discovered on the current host."""

    indicators: list[str] = []
    runtime: str | None = None

    for path, label in (
        (Path("/.dockerenv"), "dockerenv"),
        (Path("/run/.containerenv"), "containerenv"),
    ):
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if exists:
            indicator = f"path:{label}"
            indicators.append(indicator)
            if runtime is None:
                runtime = "docker" if label == "dockerenv" else None

    for env_var in ("container", "CONTAINER", "OCI_CONTAINERS", "KUBERNETES_SERVICE_HOST"):
        value = os.getenv(env_var)
        if not value:
            continue
        indicator = f"env:{env_var.lower()}"
        indicators.append(indicator)
        if runtime is None and env_var.lower().startswith("oci"):
            runtime = "oci"
        elif runtime is None and env_var.lower().startswith("kubernetes"):
            runtime = "kubernetes"

    token_runtime_map = {
        "docker": "docker",
        "kubepods": "kubernetes",
        "containerd": "containerd",
        "crio": "cri-o",
        "podman": "podman",
        "libpod": "podman",
        "lxc": "lxc",
        "garden": "garden",
    }

    for probe in ("/proc/1/cgroup", "/proc/self/cgroup"):
        try:
            with open(probe, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    for token, token_runtime in token_runtime_map.items():
                        if token in stripped:
                            indicators.append(f"cgroup:{token}")
                            if runtime is None:
                                runtime = token_runtime
        except OSError:
            continue

    deduped_indicators = tuple(dict.fromkeys(indicators))
    return bool(deduped_indicators), runtime, deduped_indicators


def _detect_ci_indicators() -> tuple[bool, tuple[str, ...]]:
    """Detect whether execution appears to be running under CI orchestration."""

    hints: list[str] = []
    ci_markers = {
        "CI": "ci",
        "GITHUB_ACTIONS": "github-actions",
        "GITLAB_CI": "gitlab-ci",
        "BUILDKITE": "buildkite",
        "CIRCLECI": "circleci",
        "TRAVIS": "travis-ci",
        "APPVEYOR": "appveyor",
        "TF_BUILD": "azure-pipelines",
        "TEAMCITY_VERSION": "teamcity",
        "BITBUCKET_BUILD_NUMBER": "bitbucket-pipelines",
        "JENKINS_URL": "jenkins",
        "CODEBUILD_BUILD_ID": "aws-codebuild",
        "CODESPACES": "github-codespaces",
    }

    for env_var, label in ci_markers.items():
        raw = os.getenv(env_var)
        if not raw:
            continue
        normalized = raw.strip().lower()
        if env_var == "CI" and normalized in {"0", "false", "no", "off"}:
            continue
        hints.append(label)

    deduped_hints = tuple(dict.fromkeys(hints))
    return bool(deduped_hints), deduped_hints


def _detect_runtime_context() -> RuntimeContext:
    """Aggregate runtime heuristics for Docker diagnostics."""

    inside_container, container_runtime, container_indicators = _detect_container_indicators()
    is_ci, ci_indicators = _detect_ci_indicators()
    return RuntimeContext(
        platform=sys.platform,
        is_windows=_is_windows(),
        is_wsl=_is_wsl(),
        inside_container=inside_container,
        container_runtime=container_runtime,
        container_indicators=container_indicators,
        is_ci=is_ci,
        ci_indicators=ci_indicators,
    )


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
    docker_dirs: list[Path] = []
    docker_updates: list[tuple[str, str]] = []
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

    docker_insertion_index = len(scripts_dirs)
    docker_candidates = list(_iter_windows_docker_directories())
    for candidate in docker_candidates:
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
            ordered_entries.insert(docker_insertion_index, key)
            docker_dirs.append(candidate_resolved)
            docker_insertion_index += 1
            updated = True
            continue

        preferred = _choose_preferred_path_entry(existing_entry, key, normalizer)
        if preferred == existing_entry:
            continue
        try:
            index = ordered_entries.index(existing_entry)
        except ValueError:
            ordered_entries.insert(docker_insertion_index, preferred)
            docker_insertion_index += 1
        else:
            ordered_entries[index] = preferred
        seen[normalized_key] = preferred
        docker_updates.append((existing_entry, preferred))
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
            if docker_dirs:
                LOGGER.info(
                    "Ensured Windows PATH contains Docker CLI directories: %s",
                    ", ".join(str(path) for path in docker_dirs),
                )
            if normalized_updates:
                LOGGER.info(
                    "Normalized existing Windows PATH entries: %s",
                    ", ".join(
                        f"{original!r} -> {updated_entry!r}"
                        for original, updated_entry in normalized_updates
                    ),
                )
            if docker_updates:
                LOGGER.info(
                    "Normalized existing Docker PATH entries: %s",
                    ", ".join(
                        f"{original!r} -> {updated_entry!r}"
                        for original, updated_entry in docker_updates
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
        "SANDBOX_DISABLE_CLEANUP": "1",
    }
    if config.skip_stripe_router:
        overrides["MENACE_SKIP_STRIPE_ROUTER"] = "1"
    if resolved_env_file is not None:
        overrides["MENACE_ENV_FILE"] = str(resolved_env_file)
    _apply_environment(overrides)
    _ensure_parent_directory(resolved_env_file)
    _ensure_windows_compatibility()
    return resolved_env_file


@dataclass(frozen=True)
class RuntimeContext:
    """Represents key characteristics of the current execution environment."""

    platform: str
    is_windows: bool
    is_wsl: bool
    inside_container: bool
    container_runtime: str | None
    container_indicators: tuple[str, ...]
    is_ci: bool
    ci_indicators: tuple[str, ...]

    def to_metadata(self) -> dict[str, str]:
        """Return a serialisable representation suitable for diagnostics."""

        metadata: dict[str, str] = {
            "platform": self.platform,
            "is_windows": str(self.is_windows).lower(),
            "is_wsl": str(self.is_wsl).lower(),
            "inside_container": str(self.inside_container).lower(),
        }
        if self.container_runtime or self.container_indicators:
            if self.container_runtime:
                metadata["container_runtime"] = self.container_runtime
            if self.container_indicators:
                metadata["container_indicators"] = ",".join(self.container_indicators)
        if self.is_ci and self.ci_indicators:
            metadata["ci_indicators"] = ",".join(self.ci_indicators)
        return metadata


@dataclass(frozen=True)
class DockerDiagnosticResult:
    """Outcome of Docker environment verification."""

    cli_path: Path | None
    available: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: Mapping[str, str]
    skipped: bool = False
    skip_reason: str | None = None


def _discover_docker_cli() -> tuple[Path | None, list[str]]:
    """Locate the Docker CLI executable if available."""

    warnings: list[str] = []
    for executable in ("docker", "docker.exe", "com.docker.cli", "com.docker.cli.exe"):
        discovered = shutil.which(executable)
        if not discovered:
            continue
        path = Path(discovered)
        if _is_windows() and path.name.lower() == "com.docker.cli.exe":
            warnings.append(
                "Resolved Docker CLI via com.docker.cli.exe shim; docker.exe alias was not present on PATH"
            )
        return path, warnings

    if _is_windows():
        for directory in _iter_windows_docker_directories():
            for candidate in ("docker.exe", "com.docker.cli.exe"):
                target = directory / candidate
                if target.exists():
                    return target, warnings
        warnings.append(
            "Docker Desktop installation was not discovered in standard locations. "
            "Install Docker Desktop or ensure docker.exe is on PATH."
        )
    elif _is_wsl():
        for directory in _iter_wsl_docker_directories():
            for candidate in ("docker.exe", "com.docker.cli.exe"):
                target = directory / candidate
                if target.exists():
                    warnings.append(
                        "Using Windows Docker CLI through WSL interop. Consider enabling the "
                        "Docker Desktop WSL integration for improved reliability."
                    )
                    return target, warnings

    return None, warnings


def _run_docker_command(
    cli_path: Path,
    args: Sequence[str],
    *,
    timeout: float,
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute a Docker CLI command and capture failures as textual diagnostics."""

    command = [str(cli_path), *args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed, None
    except FileNotFoundError:
        return None, f"Docker executable '{cli_path}' is not accessible"
    except subprocess.TimeoutExpired:
        rendered = " ".join(args)
        return None, (
            f"Docker command '{rendered}' timed out after {timeout:.1f}s; "
            "ensure Docker Desktop is running and responsive"
        )
    except OSError as exc:  # pragma: no cover - environment specific
        rendered = " ".join(args)
        return None, f"Failed to execute docker {rendered!s}: {exc}"


def _run_command(command: Sequence[str], *, timeout: float) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """Execute an arbitrary command and capture failures as diagnostics."""

    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed, None
    except FileNotFoundError:
        return None, f"Executable '{command[0]}' is not available on PATH"
    except subprocess.TimeoutExpired:
        rendered_args = " ".join(command[1:])
        if rendered_args:
            command_preview = f"{command[0]} {rendered_args}"
        else:
            command_preview = command[0]
        return None, f"Command '{command_preview}' timed out after {timeout:.1f}s"
    except OSError as exc:  # pragma: no cover - environment specific
        return None, f"Failed to execute {command[0]!s}: {exc}"


def _iter_docker_warning_messages(value: object) -> Iterable[str]:
    """Yield normalized warning strings from Docker diagnostic payloads."""

    if value is None:
        return

    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8", "ignore")
        except Exception:  # pragma: no cover - defensive fallback
            return
        yield from _iter_docker_warning_messages(decoded)
        return

    if isinstance(value, str):
        text = value.replace("\r", "\n")
        for line in text.split("\n"):
            candidate = line.strip()
            if candidate:
                yield candidate
        return

    if isinstance(value, MappingABC):
        iterable: IterableABC[object] = value.values()
    elif isinstance(value, IterableABC):
        iterable = value
    else:
        return

    for item in iterable:
        yield from _iter_docker_warning_messages(item)


_DOCKER_WARNING_PREFIX_PATTERN = re.compile(
    r"""
    ^\s*
    (?:
        warn(?:ing)?
        (?:\[[^\]]+\])?
        (?:[:\-]|::)?
        \s*
        |
        (?:warn|warning)\[[^\]]+\]\s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKER_VALUE_PATTERN = (
    r"(?:\"[^\"]+\"|'[^']+'|[A-Za-z0-9_.:/\\-]+"
    r"(?:\s+(?![A-Za-z0-9_.:/\\-]+\s*(?:=|:))[A-Za-z0-9_.:/\\-]+)*)"
)


_WORKER_CONTEXT_PREFIX_PATTERN = re.compile(
    r"(?P<context>[A-Za-z0-9_.:/\\-]+(?:\s+[A-Za-z0-9_.:/\\-]+)*)\s*(?:[:\-]|::)\s*worker\s+stalled",
    re.IGNORECASE,
)

_WORKER_CONTEXT_KV_PATTERN = re.compile(
    rf"(?P<key>context|component|module|id|name|worker|scope|subsystem|service|pipeline|task|unit|process)\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_CONTEXT_RESTART_PATTERN = re.compile(
    rf"restarting(?:\s+worker)?\s+(?P<context>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_CONTEXT_STALLED_PATTERN = re.compile(
    rf"worker\s+(?P<context>{_WORKER_VALUE_PATTERN})\s+stalled",
    re.IGNORECASE,
)

_WORKER_METADATA_TOKEN_PATTERN = re.compile(
    rf"(?P<key>[A-Za-z0-9_.-]+)\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_RESTART_KEYS = {
    "restart",
    "restarts",
    "restart_count",
    "restartcounts",
    "restartattempt",
    "restartattempts",
    "attempt",
    "attempts",
    "retry",
    "retries",
    "tries",
    "trycount",
}

_WORKER_ERROR_KEYS = {
    "error",
    "err",
    "last_error",
    "lasterror",
    "error_message",
    "failure",
    "failreason",
    "reason",
}

_WORKER_BACKOFF_KEYS = {
    "backoff",
    "delay",
    "wait",
    "cooldown",
    "interval",
    "duration",
}

_WORKER_LAST_SEEN_KEYS = {
    "since",
    "last_restart",
    "lastrestart",
    "last",
    "last_start",
    "laststart",
    "last_seen",
    "lastseen",
}


def _normalize_worker_context_candidate(candidate: str | None) -> str | None:
    """Return a cleaned worker context string if *candidate* is meaningful."""

    if not candidate:
        return None

    cleaned = candidate.strip().strip("\"'()[]{}<>")
    cleaned = re.sub(r"^(?:warn(?:ing)?|err(?:or)?|info|debug)\s*(?:[:\-]|::)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(".,;:")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"worker", "workers", "after", "restart", "restarting"}:
        return None
    if lowered.startswith("after") or lowered.startswith("workerafter"):
        return None
    if not any(char.isalpha() for char in cleaned):
        return None

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    return normalized or None


def _extract_worker_context(message: str, cleaned_message: str) -> str | None:
    """Extract the most meaningful worker context descriptor from *message*."""

    candidates: list[tuple[str, int]] = []

    context_match = re.search(
        r"worker\s+stalled[\s;,:-]*(?:restarting|restart)"
        r"(?:\s*(?:[:\-]\s*|\(\s*)(?P<context>[^)]+?)(?:\s*\)|$))?",
        cleaned_message,
        flags=re.IGNORECASE,
    )
    if context_match:
        candidate = _normalize_worker_context_candidate(context_match.group("context"))
        if candidate:
            candidates.append((candidate, 90))

    for pattern in (
        _WORKER_CONTEXT_PREFIX_PATTERN,
        _WORKER_CONTEXT_KV_PATTERN,
        _WORKER_CONTEXT_RESTART_PATTERN,
        _WORKER_CONTEXT_STALLED_PATTERN,
    ):
        for match in pattern.finditer(message):
            if "value" in match.groupdict():
                raw_candidate = match.group("value")
            else:
                raw_candidate = match.group("context")
            normalized = _normalize_worker_context_candidate(raw_candidate)
            if normalized:
                weight = 20
                key = match.groupdict().get("key", "")
                key_normalized = key.lower() if key else ""
                if key_normalized in {"worker", "id", "name"}:
                    weight = 80
                elif key_normalized in {"context", "component"}:
                    weight = 60
                elif key_normalized in {"module"}:
                    weight = 50
                elif key_normalized in {
                    "subsystem",
                    "service",
                    "scope",
                    "pipeline",
                    "task",
                    "unit",
                    "process",
                }:
                    weight = 55
                elif pattern in {_WORKER_CONTEXT_RESTART_PATTERN, _WORKER_CONTEXT_PREFIX_PATTERN}:
                    weight = 70
                candidates.append((normalized, weight))

    if not candidates:
        return None

    best_candidate, _ = max(
        candidates,
        key=lambda item: (item[1], len(item[0])),
    )

    if best_candidate.lower().startswith("worker "):
        for option, _ in sorted(candidates, key=lambda item: (-item[1], -len(item[0]))):
            if not option.lower().startswith("worker "):
                return option
        return None

    return best_candidate


def _clean_worker_metadata_value(raw_value: str) -> str:
    """Return a sanitised token extracted from worker diagnostic payloads."""

    cleaned = raw_value.strip()
    if cleaned and cleaned[0] in {'"', "'"} and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _normalise_worker_metadata_key(raw_key: str) -> str:
    """Return a canonical lowercase key for worker diagnostic attributes."""

    normalised = raw_key.strip().lower()
    if not normalised:
        return normalised
    return re.sub(r"[^a-z0-9]+", "_", normalised)


def _extract_worker_flapping_descriptors(message: str) -> tuple[list[str], dict[str, str]]:
    """Derive human friendly descriptors for flapping Docker workers."""

    descriptors: list[str] = []
    metadata: dict[str, str] = {}

    restart_count: int | None = None
    last_error: str | None = None
    backoff_hint: str | None = None
    last_seen: str | None = None

    for match in _WORKER_METADATA_TOKEN_PATTERN.finditer(message):
        key = _normalise_worker_metadata_key(match.group("key"))
        value = _clean_worker_metadata_value(match.group("value"))
        if not key or not value:
            continue

        if key in _WORKER_RESTART_KEYS and restart_count is None:
            number_match = re.search(r"(-?\d+)", value)
            if number_match:
                try:
                    restart_count = int(number_match.group(1))
                except ValueError:
                    restart_count = None
            continue

        if key in _WORKER_ERROR_KEYS and last_error is None:
            last_error = value
            continue

        if key in _WORKER_BACKOFF_KEYS and backoff_hint is None:
            backoff_hint = value
            continue

        if key in _WORKER_LAST_SEEN_KEYS and last_seen is None:
            last_seen = value
            continue

    if restart_count is None:
        fallback_restart = re.search(
            r"(?:attempt|retry|restart)[^0-9]*?(?P<count>\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_restart:
            try:
                restart_count = int(fallback_restart.group("count"))
            except ValueError:
                restart_count = None

    if last_error is None:
        fallback_error = re.search(
            r"error\s*[=:]\s*(?P<value>\"[^\"]+\"|'[^']+'|[^;\n]+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_error:
            last_error = _clean_worker_metadata_value(fallback_error.group("value"))

    if backoff_hint is None:
        fallback_backoff = re.search(
            r"backoff\s*[=:]\s*(?P<value>\"[^\"]+\"|'[^']+'|[^;\n]+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_backoff:
            backoff_hint = _clean_worker_metadata_value(fallback_backoff.group("value"))

    if backoff_hint is None:
        interval_match = re.search(
            r"worker\s+stalled;\s*restarting(?:\s+in\s+(?P<interval>[0-9.]+\s*(?:ms|s|sec|seconds|m|min|minutes|h|hours)?))?",
            message,
            flags=re.IGNORECASE,
        )
        if interval_match:
            interval = interval_match.group("interval")
            if interval:
                backoff_hint = interval.strip()

    if restart_count is not None and restart_count >= 0:
        metadata["docker_worker_restart_count"] = str(restart_count)
        plural = "s" if restart_count != 1 else ""
        descriptors.append(f"{restart_count} restart{plural} observed")

    if backoff_hint:
        metadata["docker_worker_backoff"] = backoff_hint
        descriptors.append(f"reported backoff interval: {backoff_hint}")

    if last_seen:
        metadata["docker_worker_last_restart"] = last_seen
        descriptors.append(f"last restart marker: {last_seen}")

    if last_error:
        metadata["docker_worker_last_error"] = last_error
        descriptors.append(f"last reported error: {last_error}")

    return descriptors, metadata


def _normalise_docker_warning(message: str) -> tuple[str | None, dict[str, str]]:
    """Return a cleaned warning and metadata extracted from Docker output."""

    cleaned = _DOCKER_WARNING_PREFIX_PATTERN.sub("", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None, {}

    metadata: dict[str, str] = {}
    lowered = cleaned.lower()
    if "worker stalled" in lowered:
        metadata["docker_worker_health"] = "flapping"
        context = _extract_worker_context(message, cleaned)
        if context:
            metadata["docker_worker_context"] = context

        cleaned = (
            "Docker Desktop reported repeated restarts of a background worker. "
            "Restart Docker Desktop, ensure Hyper-V or WSL 2 virtualization is enabled, and "
            "allocate additional CPU/RAM to the Docker VM before retrying."
        )

        descriptors, worker_metadata = _extract_worker_flapping_descriptors(message)
        metadata.update(worker_metadata)

        detail_segments: list[str] = []
        if context:
            detail_segments.append(f"Affected component: {context}.")
        if descriptors:
            detail_segments.append("Additional context: " + "; ".join(descriptors) + ".")

        if detail_segments:
            cleaned += " " + " ".join(detail_segments)

    return cleaned, metadata


def _normalize_warning_collection(messages: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Normalise warning ``messages`` and capture associated metadata."""

    normalized: list[str] = []
    metadata: dict[str, str] = {}
    seen: set[str] = set()

    for message in messages:
        cleaned, extracted = _normalise_docker_warning(message)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        metadata.update(extracted)

    return normalized, metadata


def _parse_key_value_lines(payload: str) -> dict[str, str]:
    """Return key/value mappings parsed from ``payload`` lines."""

    parsed: dict[str, str] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
        parsed[normalized_key] = value.strip()
    return parsed


def _parse_wsl_distribution_table(payload: str) -> list[dict[str, str | bool]]:
    """Return WSL distribution metadata parsed from ``wsl.exe -l -v`` output."""

    distributions: list[dict[str, str | bool]] = []

    header_consumed = False
    for raw_line in payload.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not header_consumed:
            header_consumed = True
            columns = [column.upper() for column in re.split(r"\s{2,}", line.strip())[:3]]
            if columns == ["NAME", "STATE", "VERSION"]:
                continue
        is_default = line.lstrip().startswith("*")
        normalized = line.lstrip("*").strip()
        if not normalized:
            continue
        parts = re.split(r"\s{2,}", normalized)
        if len(parts) < 3:
            collapsed = re.sub(r"\s+", " ", normalized)
            parts = collapsed.split(" ")
            if len(parts) < 3:
                continue
            name = " ".join(parts[:-2])
            state, version = parts[-2:]
        else:
            name, state, version = parts[0], parts[1], parts[2]

        distributions.append(
            {
                "name": name.strip(),
                "state": state.strip(),
                "version": version.strip(),
                "is_default": is_default,
            }
        )

    return distributions


def _collect_windows_virtualization_insights(timeout: float = 6.0) -> tuple[list[str], list[str], dict[str, str]]:
    """Gather virtualization diagnostics relevant to Docker Desktop on Windows."""

    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, str] = {}

    status_proc, failure = _run_command(["wsl.exe", "--status"], timeout=timeout)
    if failure:
        warnings.append(f"Unable to query WSL status: {failure}")
    elif status_proc is not None:
        if status_proc.stdout.strip():
            metadata["wsl_status_raw"] = status_proc.stdout.strip()
        parsed = _parse_key_value_lines(status_proc.stdout)
        default_version = parsed.get("default_version")
        if default_version:
            metadata["wsl_default_version"] = default_version
            if default_version and not default_version.startswith("2"):
                errors.append(
                    "WSL default version is set to %s. Docker Desktop requires WSL 2 for stable operation. "
                    "Run 'wsl --set-default-version 2' from an elevated PowerShell session and reboot."
                    % default_version
                )
        wsl_version = parsed.get("wsl_version")
        if wsl_version:
            metadata["wsl_version"] = wsl_version
        if not parsed and status_proc.stdout:
            lower = status_proc.stdout.lower()
            if "not installed" in lower or "not enabled" in lower:
                errors.append(
                    "Windows Subsystem for Linux is not fully enabled. Enable the 'Virtual Machine Platform' and 'Windows Subsystem for Linux' optional features and restart."
                )

    list_proc, failure = _run_command(["wsl.exe", "-l", "-v"], timeout=timeout)
    if failure:
        warnings.append(f"Unable to enumerate WSL distributions: {failure}")
    elif list_proc is not None and list_proc.stdout:
        metadata["wsl_list_raw"] = list_proc.stdout.strip()
        distributions = _parse_wsl_distribution_table(list_proc.stdout)
        default_distro: str | None = None
        docker_states: dict[str, str] = {}
        for item in distributions:
            name = str(item.get("name", "")).strip()
            state = str(item.get("state", "")).strip()
            version = str(item.get("version", "")).strip()
            if not name:
                continue
            key_prefix = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
            if key_prefix:
                if state:
                    metadata[f"wsl_distro_{key_prefix}_state"] = state
                if version:
                    metadata[f"wsl_distro_{key_prefix}_version"] = version
            if bool(item.get("is_default")):
                metadata["wsl_default_distribution"] = name
                default_distro = name
                if state.lower() not in {"running", "starting"}:
                    warnings.append(
                        "Default WSL distribution '%s' is %s. Start the distribution or switch Docker Desktop to a running distribution via Settings > Resources > WSL Integration."
                        % (name, state or "stopped")
                    )
            normalized_name = name.lower()
            if normalized_name in {"docker-desktop", "docker-desktop-data"}:
                docker_states[normalized_name] = state
                if state.lower() not in {"running", "starting"}:
                    errors.append(
                        "WSL distribution '%s' is %s. Start Docker Desktop and ensure its WSL integration is healthy."
                        % (name, state or "stopped")
                    )
        if not distributions:
            warnings.append(
                "WSL reported no installed distributions; Docker Desktop cannot operate without the 'docker-desktop' distribution."
            )
        elif {"docker-desktop", "docker-desktop-data"} - set(docker_states):
            missing = sorted({"docker-desktop", "docker-desktop-data"} - set(docker_states))
            errors.append(
                "Required Docker Desktop WSL distributions are missing: %s. Re-run 'wsl --install' or reinstall Docker Desktop."
                % ", ".join(missing)
            )
        if default_distro is None and distributions:
            warnings.append(
                "No default WSL distribution detected. Assign one with 'wsl --set-default <distro>' to avoid Docker context issues."
            )

    hyperv_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "(Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All).State",
    ]
    hyperv_proc, failure = _run_command(hyperv_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect Hyper-V feature state: {failure}")
    elif hyperv_proc is not None:
        lines = [line.strip() for line in hyperv_proc.stdout.splitlines() if line.strip()]
        hyperv_state = lines[-1] if lines else ""
        if hyperv_state:
            metadata["hyper_v_state"] = hyperv_state
            if hyperv_state.lower() not in {"enabled", "enablepending"}:
                errors.append(
                    "Hyper-V is %s. Enable Hyper-V (and its management tools) from Windows Features, reboot, and relaunch Docker Desktop."
                    % hyperv_state
                )

    vmp_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State",
    ]
    vmp_proc, failure = _run_command(vmp_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect Virtual Machine Platform state: {failure}")
    elif vmp_proc is not None:
        lines = [line.strip() for line in vmp_proc.stdout.splitlines() if line.strip()]
        vmp_state = lines[-1] if lines else ""
        if vmp_state:
            metadata["virtual_machine_platform_state"] = vmp_state
            if vmp_state.lower() not in {"enabled", "enablepending"}:
                errors.append(
                    "Windows 'Virtual Machine Platform' feature is %s. Enable it via 'OptionalFeatures.exe', reboot, and restart Docker Desktop."
                    % vmp_state
                )

    vmcompute_cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "(Get-Service -Name vmcompute).Status",
    ]
    vmcompute_proc, failure = _run_command(vmcompute_cmd, timeout=timeout)
    if failure:
        warnings.append(f"Unable to inspect Hyper-V compute service state: {failure}")
    elif vmcompute_proc is not None:
        lines = [line.strip() for line in vmcompute_proc.stdout.splitlines() if line.strip()]
        vmcompute_state = lines[-1] if lines else ""
        if vmcompute_state:
            metadata["vmcompute_status"] = vmcompute_state
            if vmcompute_state.lower() not in {"running", "startpending"}:
                errors.append(
                    "Hyper-V compute service (vmcompute) is %s. Start the service from an elevated PowerShell session with 'Start-Service vmcompute'."
                    % vmcompute_state
                )

    return warnings, errors, metadata


def _coerce_optional_int(value: object) -> int | None:
    """Convert *value* to ``int`` when possible."""

    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _estimate_backoff_seconds(value: str | None) -> float | None:
    """Approximate the restart backoff interval extracted from Docker warnings."""

    if not value:
        return None
    match = _BACKOFF_INTERVAL_PATTERN.search(value)
    if not match:
        return None
    raw = match.group("value")
    unit = match.group("unit") or "s"
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None

    unit_normalized = unit.lower()
    if unit_normalized in {"ms", "msec", "milliseconds"}:
        return numeric / 1000.0
    if unit_normalized in {"s", "sec", "secs", "seconds"}:
        return numeric
    if unit_normalized in {"m", "min", "mins", "minutes"}:
        return numeric * 60.0
    if unit_normalized in {"h", "hr", "hrs", "hours"}:
        return numeric * 3600.0
    return None


@dataclass(frozen=True)
class WorkerRestartTelemetry:
    """Structured representation of Docker worker health metadata."""

    context: str | None
    restart_count: int | None
    backoff_hint: str | None
    last_seen: str | None
    last_error: str | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> "WorkerRestartTelemetry":
        return cls(
            context=metadata.get("docker_worker_context"),
            restart_count=_coerce_optional_int(metadata.get("docker_worker_restart_count")),
            backoff_hint=metadata.get("docker_worker_backoff"),
            last_seen=metadata.get("docker_worker_last_restart"),
            last_error=metadata.get("docker_worker_last_error"),
        )

    @property
    def backoff_seconds(self) -> float | None:
        """Best-effort conversion of the Docker restart backoff to seconds."""

        return _estimate_backoff_seconds(self.backoff_hint)


@dataclass(frozen=True)
class WorkerHealthAssessment:
    """Classification of Docker worker restart telemetry."""

    severity: Literal["warning", "error"]
    headline: str
    details: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()

    def render(self) -> str:
        """Compose a human readable message from the assessment components."""

        segments = [self.headline]
        if self.details:
            segments.append(" ".join(detail.strip() for detail in self.details if detail))
        if self.remediation:
            segments.append(" ".join(hint.strip() for hint in self.remediation if hint))
        return " ".join(segment for segment in segments if segment)


_CRITICAL_WORKER_ERROR_KEYWORDS = (
    "fatal",
    "panic",
    "exhausted",
    "cannot allocate",
    "unrecoverable",
    "out of memory",
    "corrupted",
)


def _is_critical_worker_error(message: str | None) -> bool:
    """Return ``True`` when *message* indicates a severe worker failure."""

    if not message:
        return False
    lowered = message.lower()
    return any(keyword in lowered for keyword in _CRITICAL_WORKER_ERROR_KEYWORDS)


def _classify_worker_flapping(
    telemetry: WorkerRestartTelemetry,
    context: RuntimeContext,
) -> WorkerHealthAssessment:
    """Categorise Docker worker restart telemetry into actionable guidance."""

    details: list[str] = []
    remediation: list[str] = []

    if telemetry.context:
        details.append(f"Affected component: {telemetry.context}.")

    if telemetry.restart_count is not None:
        plural = "s" if telemetry.restart_count != 1 else ""
        details.append(f"Docker reported {telemetry.restart_count} restart{plural} in the last diagnostic window.")

    if telemetry.backoff_hint:
        details.append(f"Backoff interval advertised by Docker: {telemetry.backoff_hint}.")

    if telemetry.last_seen:
        details.append(f"Last restart marker: {telemetry.last_seen}.")

    if telemetry.last_error:
        details.append(f"Most recent worker error: {telemetry.last_error}.")

    severity: Literal["warning", "error"] = "warning"

    if telemetry.restart_count is not None and telemetry.restart_count >= 6:
        severity = "error"
    elif telemetry.restart_count is not None and telemetry.restart_count >= 3:
        # Elevated churn warrants increased attention but can often self-resolve.
        severity = "warning"

    if telemetry.backoff_seconds is not None and telemetry.backoff_seconds >= 60:
        severity = "error"

    if _is_critical_worker_error(telemetry.last_error):
        severity = "error"

    if context.is_wsl:
        remediation.append(
            "Ensure WSL 2 is enabled for the distribution, install the latest WSL kernel update, and enable Docker Desktop's WSL integration for the distribution in settings."
        )
    elif context.is_windows:
        remediation.append(
            "Enable the Hyper-V and Virtual Machine Platform Windows features, allocate sufficient CPU and memory to Docker Desktop, and restart Docker Desktop after applying changes."
        )
    else:
        remediation.append(
            "Restart the Docker daemon and inspect host virtualization services for resource starvation or crashes."
        )

    if severity == "warning":
        headline = (
            "Docker Desktop observed worker restarts but reported they are recovering automatically. Monitor Docker Desktop and re-run bootstrap if instability persists."
        )
    else:
        headline = (
            "Docker Desktop worker processes are repeatedly restarting and may not stabilize without intervention."
        )

    return WorkerHealthAssessment(
        severity=severity,
        headline=headline,
        details=tuple(details),
        remediation=tuple(remediation),
    )


def _post_process_docker_health(
    *,
    metadata: dict[str, str],
    context: RuntimeContext,
    timeout: float = 6.0,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Augment diagnostics when Docker reports unhealthy background workers."""

    worker_health = metadata.get("docker_worker_health")
    if worker_health != "flapping":
        return [], [], {}

    telemetry = WorkerRestartTelemetry.from_metadata(metadata)
    assessment = _classify_worker_flapping(telemetry, context)

    warnings: list[str] = []
    errors: list[str] = []
    additional_metadata: dict[str, str] = {}

    if assessment.severity == "warning":
        warnings.append(assessment.render())
        return warnings, errors, additional_metadata

    errors.append(assessment.render())

    if context.is_wsl or context.is_windows:
        vw_warnings, vw_errors, vw_metadata = _collect_windows_virtualization_insights(timeout=timeout)
        warnings.extend(vw_warnings)
        errors.extend(vw_errors)
        additional_metadata.update(vw_metadata)

    return warnings, errors, additional_metadata


def _normalize_docker_warnings(value: object) -> tuple[list[str], dict[str, str]]:
    """Normalise Docker warnings into unique, user-friendly strings."""

    return _normalize_warning_collection(_iter_docker_warning_messages(value))


def _extract_json_document(stdout: str, stderr: str) -> tuple[str | None, list[str], dict[str, str]]:
    """Extract the JSON payload and structured warnings from Docker command output.

    Docker Desktop – especially on Windows – occasionally prefixes formatted
    JSON output with warning banners such as ``WARNING: worker stalled;
    restarting``.  Earlier bootstrap logic attempted to ``json.loads`` the raw
    ``stdout`` payload which failed in these situations and left the user with a
    cryptic parsing error.  This helper tolerantly searches ``stdout`` for the
    first decodable JSON document while collecting any surrounding text as human
    readable warnings.  ``stderr`` content is also normalised into warnings and
    annotated metadata so that diagnostics remain actionable.
    """

    decoder = json.JSONDecoder()
    warnings: list[str] = []

    def _normalise_stream(value: str | None) -> str:
        if not value:
            return ""
        return value.replace("\r", "\n")

    stdout_normalized = _normalise_stream(stdout)
    stderr_normalized = _normalise_stream(stderr)

    json_fragment: str | None = None

    if stdout_normalized:
        search_text = stdout_normalized
        index = 0
        length = len(search_text)
        while index < length:
            char = search_text[index]
            if char not in "[{":
                index += 1
                continue
            try:
                _, end = decoder.raw_decode(search_text[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            start = index
            finish = index + end
            json_fragment = search_text[start:finish]
            prefix = search_text[:start]
            suffix = search_text[finish:]
            warnings.extend(_iter_docker_warning_messages(prefix))
            warnings.extend(_iter_docker_warning_messages(suffix))
            break
        if json_fragment is None:
            warnings.extend(_iter_docker_warning_messages(search_text))

    if json_fragment is None:
        warnings.extend(_iter_docker_warning_messages(stderr_normalized))
        normalized_warnings, metadata = _normalize_warning_collection(warnings)
        return None, normalized_warnings, metadata

    warnings.extend(_iter_docker_warning_messages(stderr_normalized))
    normalized_warnings, warning_metadata = _normalize_warning_collection(warnings)
    return json_fragment.strip(), normalized_warnings, warning_metadata


def _parse_docker_json(
    proc: subprocess.CompletedProcess[str],
    command: str,
) -> tuple[object | None, list[str], dict[str, str]]:
    """Return decoded JSON output, normalised warnings, and metadata."""

    payload, collected_warnings, warning_metadata = _extract_json_document(
        proc.stdout, proc.stderr
    )

    if not payload:
        collected_warnings.append(f"docker {command} produced no JSON output")
        normalized_warnings, metadata = _normalize_warning_collection(collected_warnings)
        metadata.update(warning_metadata)
        return None, normalized_warnings, metadata

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        collected_warnings.append(f"Failed to parse docker {command} payload: {exc}")
        normalized_warnings, metadata = _normalize_warning_collection(collected_warnings)
        metadata.update(warning_metadata)
        return None, normalized_warnings, metadata

    return decoded, collected_warnings, warning_metadata


def _probe_docker_environment(cli_path: Path, timeout: float) -> tuple[dict[str, str], list[str], list[str]]:
    """Gather Docker daemon metadata and associated warnings/errors."""

    metadata: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []

    version_proc, failure = _run_docker_command(cli_path, ["version", "--format", "{{json .}}"], timeout=timeout)
    if failure:
        errors.append(failure)
        return metadata, warnings, errors

    if version_proc is None:
        return metadata, warnings, errors

    if version_proc.returncode != 0:
        detail = version_proc.stderr.strip() or version_proc.stdout.strip()
        errors.append(
            "docker version returned non-zero exit code "
            f"{version_proc.returncode}: {detail or 'no diagnostic output provided'}"
        )
        return metadata, warnings, errors

    version_data, version_warnings, version_metadata = _parse_docker_json(
        version_proc, "version"
    )
    warnings.extend(version_warnings)
    metadata.update(version_metadata)

    if isinstance(version_data, dict):
        client_data = version_data.get("Client", {}) if isinstance(version_data, dict) else {}
        server_data = version_data.get("Server", {}) if isinstance(version_data, dict) else {}
        client_version = str(client_data.get("Version", "")).strip()
        api_version = str(client_data.get("ApiVersion", "")).strip()
        server_version = str(server_data.get("Version", "")).strip()
        if client_version:
            metadata["client_version"] = client_version
        if api_version:
            metadata["api_version"] = api_version
        if server_version:
            metadata["server_version"] = server_version
        else:
            errors.append(
                "Docker daemon appears to be unavailable; version output omitted server details. "
                "Start Docker Desktop or connect to a reachable daemon."
            )

    if errors:
        return metadata, warnings, errors

    info_proc, failure = _run_docker_command(cli_path, ["info", "--format", "{{json .}}"], timeout=timeout)
    if failure:
        errors.append(failure)
        return metadata, warnings, errors

    if info_proc is None:
        return metadata, warnings, errors

    if info_proc.returncode != 0:
        detail = info_proc.stderr.strip() or info_proc.stdout.strip()
        errors.append(
            "docker info returned non-zero exit code "
            f"{info_proc.returncode}: {detail or 'no diagnostic output provided'}"
        )
        return metadata, warnings, errors

    info_data, info_warnings, info_metadata = _parse_docker_json(info_proc, "info")
    warnings.extend(info_warnings)
    metadata.update(info_metadata)

    if isinstance(info_data, dict):
        for key, metadata_key in (
            ("ServerVersion", "server_version"),
            ("OperatingSystem", "operating_system"),
            ("OSType", "os_type"),
            ("Architecture", "architecture"),
            ("DockerRootDir", "root_dir"),
        ):
            value = str(info_data.get(key, "")).strip()
            if value and metadata_key not in metadata:
                metadata[metadata_key] = value

        warnings_field = info_data.get("Warnings")
        normalized_warnings, warning_metadata = _normalize_docker_warnings(warnings_field)
        warnings.extend(normalized_warnings)
        metadata.update(warning_metadata)

        if _is_windows() or _is_wsl():
            context = str(info_data.get("Name", "")).strip()
            if context and context.lower() not in {"docker-desktop", "desktop-linux"}:
                warnings.append(
                    "Docker context '%s' is active; Docker Desktop typically uses 'docker-desktop'. "
                    "Verify that the desired context is selected before launching sandboxes."
                    % context
                )
    elif info_data is not None:  # pragma: no cover - unexpected payloads
        warnings.append("docker info returned an unexpected payload structure")

    return metadata, warnings, errors


def _infer_missing_docker_skip_reason(context: RuntimeContext) -> str | None:
    """Return a descriptive reason for skipping Docker verification if appropriate."""

    assume_no = os.getenv(_DOCKER_ASSUME_NO_ENV)
    if assume_no and assume_no.strip().lower() in {"1", "true", "yes", "on"}:
        return f"Docker diagnostics disabled via {_DOCKER_ASSUME_NO_ENV}"

    if context.inside_container and not context.is_windows:
        runtime_label = context.container_runtime or "container"
        indicators = ", ".join(context.container_indicators) or "no explicit indicators"
        return (
            "Detected execution inside a %s-managed environment (%s) without Docker CLI access; "
            "assuming the host manages containers and skipping Docker diagnostics."
            % (runtime_label, indicators)
        )

    if context.is_ci:
        ci_label = ", ".join(context.ci_indicators) or "CI"
        return (
            "Detected continuous integration environment (%s) without Docker CLI access; "
            "skipping Docker diagnostics."
            % ci_label
        )

    return None


def _collect_docker_diagnostics(timeout: float = 12.0) -> DockerDiagnosticResult:
    """Inspect the Docker environment and return detailed diagnostics."""

    context = _detect_runtime_context()
    metadata: dict[str, str] = context.to_metadata()

    cli_path, cli_warnings = _discover_docker_cli()
    warnings = list(cli_warnings)
    errors: list[str] = []

    if cli_path is None:
        skip_reason = _infer_missing_docker_skip_reason(context)
        if skip_reason:
            metadata["skip_reason"] = skip_reason
            return DockerDiagnosticResult(
                cli_path=None,
                available=False,
                errors=(),
                warnings=(),
                metadata=metadata,
                skipped=True,
                skip_reason=skip_reason,
            )

        errors.append(
            "Docker CLI executable was not found. Install Docker Desktop or ensure 'docker' is on PATH."
        )
        return DockerDiagnosticResult(
            cli_path=None,
            available=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metadata=metadata,
        )

    metadata["cli_path"] = str(cli_path)

    probe_metadata, probe_warnings, probe_errors = _probe_docker_environment(cli_path, timeout)
    metadata.update(probe_metadata)
    warnings.extend(probe_warnings)
    errors.extend(probe_errors)

    health_warnings, health_errors, health_metadata = _post_process_docker_health(
        metadata=metadata,
        context=context,
    )
    warnings.extend(health_warnings)
    errors.extend(health_errors)
    metadata.update(health_metadata)

    warnings = _coalesce_iterable(warnings)
    errors = _coalesce_iterable(errors)

    available = not errors

    return DockerDiagnosticResult(
        cli_path=cli_path,
        available=available,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _bool_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _verify_docker_environment() -> None:
    """Perform Docker diagnostics and surface actionable guidance to the user."""

    if _bool_env_flag(_DOCKER_SKIP_ENV):
        LOGGER.info(
            "Skipping Docker diagnostics due to %s environment override",
            _DOCKER_SKIP_ENV,
        )
        return

    diagnostics = _collect_docker_diagnostics()

    if diagnostics.skipped:
        message = diagnostics.skip_reason or "Docker diagnostics skipped"
        LOGGER.info("Skipping Docker diagnostics: %s", message)
        extra_context = {
            key: value
            for key, value in diagnostics.metadata.items()
            if key not in {"skip_reason", "cli_path"}
        }
        if extra_context:
            LOGGER.debug("Docker runtime context: %s", extra_context)
        return

    for warning in diagnostics.warnings:
        LOGGER.warning("Docker diagnostic warning: %s", warning)

    if diagnostics.available:
        details = {
            key: value
            for key, value in diagnostics.metadata.items()
            if key in {"server_version", "operating_system", "os_type", "architecture", "cli_path"}
            and value
        }
        if details:
            summary = ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
            LOGGER.info("Docker daemon reachable (%s)", summary)
        return

    for error in diagnostics.errors:
        LOGGER.warning("Docker diagnostic error: %s", error)

    if _bool_env_flag(_DOCKER_REQUIRE_ENV):
        raise BootstrapError(
            "Docker environment verification failed and %s is set. "
            "Review the diagnostic warnings above before retrying." % _DOCKER_REQUIRE_ENV
        )


def _run_bootstrap(config: BootstrapConfig) -> None:
    resolved_env_file = _prepare_environment(config)

    _verify_docker_environment()

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
