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
from collections.abc import (
    Iterable as IterableABC,
    Mapping as MappingABC,
    Sequence as SequenceABC,
)
from functools import lru_cache
import ntpath
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence, Literal

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


_APPROX_PREFIX_PATTERN = re.compile(
    r"^(?P<prefix>about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*",
    flags=re.IGNORECASE,
)

_APPROX_SUFFIX_PATTERN = re.compile(
    r"(about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*$",
    flags=re.IGNORECASE,
)

_BACKOFF_INTERVAL_PATTERN = re.compile(
    r"""
    (?P<prefix>
        (?:about|approx(?:\.|imately)?|approximately|around|roughly|near(?:ly)?|~|≈)\s*
    )?
    (?P<number>[0-9]+(?:\.[0-9]+)?)
    \s*
    (?P<unit>ms|msec|milliseconds|s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_GO_DURATION_PATTERN = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?[hms](?:[0-9]+(?:\.[0-9]+)?[hms]){0,2}\b",
    flags=re.IGNORECASE,
)

_GO_DURATION_COMPONENT_PATTERN = re.compile(
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>[hms])",
    flags=re.IGNORECASE,
)

_CLOCK_DURATION_PATTERN = re.compile(r"^\d+(?::\d+){1,3}(?:\.\d+)?$")

_CLOCK_DURATION_SEARCH_PATTERN = re.compile(r"\b\d+(?::\d+){1,3}(?:\.\d+)?\b")

_CLOCK_DURATION_LAYOUTS: dict[int, tuple[str, ...]] = {
    2: ("minutes", "seconds"),
    3: ("hours", "minutes", "seconds"),
    4: ("days", "hours", "minutes", "seconds"),
}

_CLOCK_DURATION_FACTORS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}

_CLOCK_DURATION_SYMBOLS = {
    "seconds": "s",
    "minutes": "m",
    "hours": "h",
    "days": "d",
}

_DURATION_UNIT_NORMALISATION = {
    "ms": "ms",
    "msec": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "seconds": "s",
    "m": "m",
    "min": "m",
    "mins": "m",
    "minutes": "m",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hours": "h",
}


_DOCKER_LOG_FIELD_PATTERN = re.compile(
    r"""
    (?P<key>[A-Za-z0-9_.-]+)
    =
    (
        "(?P<double>(?:\\.|[^"\\])*)"
        |
        '(?P<single>(?:\\.|[^'\\])*)'
        |
        (?P<bare>[^\s]+)
    )
    """,
    flags=re.VERBOSE,
)

_WORKER_ERROR_NORMALISERS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"worker\s+stalled", flags=re.IGNORECASE),
        "stalled_restart",
        "Docker Desktop automatically restarted a background worker after it stalled",
    ),
    (
        re.compile(r"restart\s+loop", flags=re.IGNORECASE),
        "restart_loop",
        "Docker Desktop detected that a background worker entered a restart loop",
    ),
    (
        re.compile(r"health\s*check\s+(?:failed|timed?\s*out)", flags=re.IGNORECASE),
        "healthcheck_failure",
        "Docker Desktop reported that the worker health check failed",
    ),
)

_WORKER_ERROR_CODE_LABELS: Mapping[str, str] = {
    "stalled_restart": "an automatic restart after a stall",
    "restart_loop": "a restart loop",
    "healthcheck_failure": "a health-check failure",
}


_WORKER_STALLED_VARIATIONS_PATTERN = re.compile(
    r"""
    worker
    (?:
        \s+(?:has|have|had|is|was|are|were)(?:\s+been)?
        |
        \s+(?:appears?|appeared|appearing|seems?|seemed|seeming)(?:\s+to)?(?:\s+have)?(?:\s+been)?
        |
        \s+(?:may|might|could|should|would)\s+(?:have)?(?:\s+been)?
        |
        \s+(?:apparently|reportedly|likely|probably|possibly|potentially|maybe|virtually|nearly|almost|still|persistently|chronically|repeatedly)
    )+
    \s+stalled
    """,
    re.IGNORECASE | re.VERBOSE,
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


def _split_metadata_values(value: str | None) -> list[str]:
    """Return a list of normalised entries parsed from a metadata field."""

    if not value:
        return []

    if not isinstance(value, str):  # pragma: no cover - defensive guardrail
        value = str(value)

    tokens: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None

    pairing = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairing.items()}

    for char in value:
        if quote:
            buffer.append(char)
            if char == quote and (len(buffer) < 2 or buffer[-2] != "\\"):
                quote = None
            continue

        if char in {'"', "'"}:
            quote = char
            buffer.append(char)
            continue

        if char in pairing:
            depth += 1
            buffer.append(char)
            continue

        if char in closing and depth > 0:
            depth = max(0, depth - 1)
            buffer.append(char)
            continue

        if depth == 0 and char in {",", ";", "\n"}:
            token = "".join(buffer).strip()
            if token:
                tokens.append(token)
            buffer.clear()
            continue

        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        tokens.append(tail)

    return tokens


def _parse_int_sequence(value: str | None) -> tuple[int, ...]:
    """Parse a metadata field containing integer samples."""

    samples: list[int] = []
    seen: set[int] = set()
    for token in _split_metadata_values(value):
        try:
            number = int(token)
        except ValueError:
            continue
        if number in seen:
            continue
        seen.add(number)
        samples.append(number)
    return tuple(samples)


def _decode_docker_log_value(value: str) -> str:
    """Best-effort decoding of escaped Docker log field values."""

    if not value:
        return ""

    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:  # pragma: no cover - extremely defensive
        return value


def _stringify_envelope_value(value: Any) -> str | None:
    """Convert structured log payload values into displayable strings."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover - defensive guardrail
            decoded = bytes(value).decode("utf-8", errors="ignore").strip()
        return decoded or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ingest_structured_envelope(
    envelope: dict[str, str], payload: Any, prefix: str | None = None
) -> None:
    """Populate ``envelope`` with data extracted from *payload* recursively."""

    if isinstance(payload, MappingABC):
        _ingest_structured_mapping(envelope, payload, prefix)
        return

    if isinstance(payload, SequenceABC) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            child_prefix = f"{prefix}_{index}" if prefix else str(index)
            _ingest_structured_envelope(envelope, item, child_prefix)
        return

    if prefix and prefix not in envelope:
        text = _stringify_envelope_value(payload)
        if text:
            envelope[prefix] = text


def _ingest_structured_mapping(
    envelope: dict[str, str], mapping: Mapping[Any, Any], prefix: str | None = None
) -> None:
    """Flatten mapping-like Docker payloads into the ``envelope`` dictionary."""

    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            continue
        normalized_key = raw_key.strip()
        if not normalized_key:
            continue

        composite_key = (
            normalized_key if prefix is None else f"{prefix}_{normalized_key}"
        )

        if isinstance(value, MappingABC):
            _ingest_structured_mapping(envelope, value, composite_key)
            continue

        if isinstance(value, SequenceABC) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            inline_parts: list[str] = []
            for index, item in enumerate(value):
                if isinstance(item, MappingABC):
                    _ingest_structured_envelope(
                        envelope, item, f"{composite_key}_{index}"
                    )
                    continue
                text = _stringify_envelope_value(item)
                if not text:
                    continue
                inline_parts.append(text)
                if prefix is not None:
                    indexed_key = f"{composite_key}_{index}"
                    if indexed_key not in envelope:
                        envelope[indexed_key] = text
            if inline_parts:
                joined = ", ".join(inline_parts)
                if normalized_key not in envelope:
                    envelope[normalized_key] = joined
                if (
                    composite_key not in envelope
                    and composite_key != normalized_key
                ):
                    envelope[composite_key] = joined
            continue

        text = _stringify_envelope_value(value)
        if not text:
            continue

        if normalized_key not in envelope:
            envelope[normalized_key] = text
        if composite_key not in envelope and composite_key != normalized_key:
            envelope[composite_key] = text


def _ingest_json_fragments(message: str, envelope: dict[str, str]) -> None:
    """Extract JSON fragments embedded within Docker diagnostic output."""

    decoder = json.JSONDecoder()
    index = 0
    length = len(message)

    while index < length:
        char = message[index]
        if char not in "[{":
            index += 1
            continue
        try:
            payload, end = decoder.raw_decode(message, index)
        except ValueError:
            index += 1
            continue
        _ingest_structured_envelope(envelope, payload)
        index = end


def _parse_docker_log_envelope(message: str) -> dict[str, str]:
    """Extract key/value pairs embedded in structured Docker log lines."""

    if not message:
        return {}

    envelope: dict[str, str] = {}
    _ingest_json_fragments(message, envelope)

    for match in _DOCKER_LOG_FIELD_PATTERN.finditer(message):
        key = match.group("key")
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        decoded = _decode_docker_log_value(value)
        if key not in envelope:
            envelope[key] = decoded
    return envelope


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

    def _unique(paths: Iterable[Path]) -> Iterable[Path]:
        seen: set[str] = set()
        for candidate in paths:
            normalized = os.path.normcase(str(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            yield candidate

    program_roots: list[Path] = []
    for env_var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        value = os.environ.get(env_var)
        if value:
            program_roots.append(Path(value))

    if not program_roots:
        program_roots.extend(
            Path(path)
            for path in (r"C:\\Program Files", r"C:\\Program Files (x86)")
        )

    default_targets = [
        root / "Docker" / "Docker" / "resources" / "bin" for root in program_roots
    ]

    program_data = os.environ.get("ProgramData")
    if program_data:
        default_targets.append(Path(program_data) / "DockerDesktop" / "version-bin")
    else:
        default_targets.append(Path(r"C:\\ProgramData") / "DockerDesktop" / "version-bin")

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        user_root = Path(local_appdata)
        default_targets.append(
            user_root / "Programs" / "Docker" / "Docker" / "resources" / "bin"
        )
        default_targets.append(user_root / "Docker" / "resources" / "bin")
    else:
        default_targets.append(
            Path.home() / "AppData" / "Local" / "Programs" / "Docker" / "Docker" / "resources" / "bin"
        )

    for target in _unique(default_targets):
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
        text = _strip_control_sequences(value).replace("\r", "\n")
        yield from _coalesce_warning_lines(text)
        return

    if isinstance(value, MappingABC):
        iterable: IterableABC[object] = value.values()
    elif isinstance(value, IterableABC):
        iterable = value
    else:
        return

    for item in iterable:
        yield from _iter_docker_warning_messages(item)


_ANSI_ESCAPE_PATTERN = re.compile(
    r"""
    (?:
        \x1B[@-Z\\-_]
        |
        \x1B\[[0-?]*[ -/]*[@-~]
        |
        \x1B\][^\x1B]*\x1B\\
        |
        \x9B[0-?]*[ -/]*[@-~]
    )
    """,
    re.VERBOSE,
)


def _strip_control_sequences(text: str) -> str:
    """Remove ANSI escapes and non-printable control characters from *text*."""

    if not text:
        return ""

    cleaned = _ANSI_ESCAPE_PATTERN.sub("", text)
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = cleaned.translate({
        0x00: None,
        0x01: None,
        0x02: None,
        0x03: None,
        0x04: None,
        0x05: None,
        0x06: None,
        0x07: None,
        0x08: None,
        0x0B: None,
        0x0C: None,
        0x0E: None,
        0x0F: None,
        0x10: None,
        0x11: None,
        0x12: None,
        0x13: None,
        0x14: None,
        0x15: None,
        0x16: None,
        0x17: None,
        0x18: None,
        0x19: None,
        0x1A: None,
        0x1B: None,
        0x1C: None,
        0x1D: None,
        0x1E: None,
        0x1F: None,
        0x7F: None,
        0x80: None,
        0x81: None,
        0x82: None,
        0x83: None,
        0x84: None,
        0x85: None,
        0x86: None,
        0x87: None,
        0x88: None,
        0x89: None,
        0x8A: None,
        0x8B: None,
        0x8C: None,
        0x8D: None,
        0x8E: None,
        0x8F: None,
        0x90: None,
        0x91: None,
        0x92: None,
        0x93: None,
        0x94: None,
        0x95: None,
        0x96: None,
        0x97: None,
        0x98: None,
        0x99: None,
        0x9A: None,
        0x9B: None,
        0x9C: None,
        0x9D: None,
        0x9E: None,
        0x9F: None,
        0x200B: None,
        0x200C: None,
        0x200D: None,
        0x2060: None,
    })
    return cleaned


def _normalise_worker_stalled_phrase(message: str) -> str:
    """Collapse phrasing variants of ``worker has stalled`` into ``worker stalled``."""

    if not message:
        return ""

    return _WORKER_STALLED_VARIATIONS_PATTERN.sub("worker stalled", message)


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
    r"""
    (?P<context>[A-Za-z0-9_.:/\\-]+(?:\s+[A-Za-z0-9_.:/\\-]+)*)
    \s*
    (?:
        [:\-]|::|->|=>|—|–|→|⇒
    )
    \s*
    worker\s+stalled
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKER_CONTEXT_KV_PATTERN = re.compile(
    rf"(?P<key>(?:context|component|module|id|name|worker|scope|subsystem|service|pipeline|task|unit|process|engine|backend|runner|channel|queue|thread|target|namespace|project|group|agent|executor|handler)(?:[._-][A-Za-z0-9]+)*)\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
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

_WORKER_CONTEXT_BRACKET_PATTERN = re.compile(
    r"\[(?P<context>[^\]\s][^\]]*?)\]\s*(?:worker\s+)?stalled",
    re.IGNORECASE,
)

_WORKER_METADATA_TOKEN_PATTERN = re.compile(
    rf"(?P<key>[A-Za-z0-9_.-]+)\s*(?:=|:)\s*(?P<value>{_WORKER_VALUE_PATTERN})",
    re.IGNORECASE,
)

_WORKER_METADATA_HEURISTIC_KEYWORDS = {
    "component",
    "context",
    "module",
    "subsystem",
    "worker",
    "namespace",
    "service",
    "scope",
    "target",
    "restarts",
    "restart",
    "backoff",
    "last_restart",
    "last-restart",
    "last error",
    "last_error",
    "error",
    "reason",
    "err",
    "retry",
    "attempt",
}


def _looks_like_worker_metadata_line(line: str) -> bool:
    """Heuristically determine whether ``line`` contains worker metadata."""

    if not line:
        return False

    token_iter = _WORKER_METADATA_TOKEN_PATTERN.finditer(line)
    for _ in token_iter:
        return True

    lowered = line.casefold()
    return any(keyword in lowered for keyword in _WORKER_METADATA_HEURISTIC_KEYWORDS)


def _coalesce_warning_lines(payload: str) -> Iterable[str]:
    """Combine continuation lines emitted as part of structured warnings."""

    pending: list[str] = []
    pending_is_worker_warning = False

    for raw_line in payload.split("\n"):
        if not raw_line:
            continue
        stripped = raw_line.strip()
        if not stripped:
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        lowered = stripped.casefold()
        line_reports_worker = "worker stalled" in lowered
        looks_like_metadata = _looks_like_worker_metadata_line(stripped)

        if pending:
            if indent > 0:
                pending.append(stripped)
                pending_is_worker_warning = pending_is_worker_warning or line_reports_worker
                continue
            if (
                pending_is_worker_warning
                and looks_like_metadata
                and not line_reports_worker
            ):
                pending.append(stripped)
                continue

            yield " ".join(pending)
            pending = []
            pending_is_worker_warning = False

        pending.append(stripped)
        pending_is_worker_warning = line_reports_worker

    if pending:
        yield " ".join(pending)


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
    "retrycount",
    "retry_count",
    "next_retry_attempt",
    "bouncecount",
}

_WORKER_RESTART_PREFIXES = {
    "restart",
    "retry",
    "attempt",
    "try",
    "bounce",
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

_WORKER_ERROR_PREFIXES = {
    "error",
    "fail",
    "reason",
    "last_error",
}

_WORKER_BACKOFF_KEYS = {
    "backoff",
    "delay",
    "wait",
    "cooldown",
    "interval",
    "duration",
    "next_retry",
    "nextretry",
    "retry_after",
    "next_restart",
    "nextstart",
}

_WORKER_BACKOFF_PREFIXES = {
    "backoff",
    "delay",
    "wait",
    "cooldown",
    "interval",
    "duration",
    "next_retry",
    "retry_after",
    "next_restart",
    "nextstart",
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
    "last_success",
    "lastsuccess",
}

_WORKER_LAST_SEEN_PREFIXES = {
    "last",
    "since",
    "previous",
}


def _classify_worker_metadata_key(key: str) -> str | None:
    """Return a semantic category for a worker telemetry key."""

    if not key:
        return None

    if key in _WORKER_RESTART_KEYS or any(key.startswith(prefix) for prefix in _WORKER_RESTART_PREFIXES):
        return "restart"

    if key in _WORKER_ERROR_KEYS or any(key.startswith(prefix) for prefix in _WORKER_ERROR_PREFIXES):
        return "error"

    if key in _WORKER_BACKOFF_KEYS or any(key.startswith(prefix) for prefix in _WORKER_BACKOFF_PREFIXES):
        return "backoff"

    if key in _WORKER_LAST_SEEN_KEYS or any(key.startswith(prefix) for prefix in _WORKER_LAST_SEEN_PREFIXES):
        return "last_seen"

    return None


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
    if lowered in {"worker", "workers", "after", "restart", "restarting", "in"}:
        return None
    if lowered.startswith("after") or lowered.startswith("workerafter"):
        return None
    if not any(char.isalpha() for char in cleaned):
        return None

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if re.match(r"^[x×]\s*\d+", normalized):
        return None
    lowered_normalized = normalized.lower()
    if lowered_normalized.startswith("x") and any(char.isdigit() for char in normalized):
        return None
    if lowered_normalized.startswith("in ") and any(char.isdigit() for char in normalized):
        return None
    if " because " in lowered_normalized:
        return None
    return normalized or None


def _extract_worker_context(message: str, cleaned_message: str) -> str | None:
    """Extract the most meaningful worker context descriptor from *message*."""

    candidates: list[tuple[str, int]] = []
    candidate_positions: dict[str, int] = {}

    def _record_candidate(text: str, weight: int) -> None:
        key = text.lower()
        existing_index = candidate_positions.get(key)
        if existing_index is None:
            candidate_positions[key] = len(candidates)
            candidates.append((text, weight))
            return
        existing_text, existing_weight = candidates[existing_index]
        if weight > existing_weight or (
            weight == existing_weight and len(text) > len(existing_text)
        ):
            candidates[existing_index] = (text, weight)

    normalized_message = _normalise_worker_stalled_phrase(message)
    normalized_cleaned = _normalise_worker_stalled_phrase(cleaned_message)

    cleaned_candidates: list[str] = []
    for candidate in (cleaned_message, normalized_cleaned):
        if candidate and candidate not in cleaned_candidates:
            cleaned_candidates.append(candidate)

    for candidate_text in cleaned_candidates:
        context_match = re.search(
            r"""
            worker\s+stalled
            (?:(?:\s*(?:[;:,.\-–—]|->|=>|→|⇒)\s*)*)
            restart(?:ing)?
            (?:\s+(?:in|after)\s+[^()]+)?
            (?:\s*(?:[:\-–—]\s*|\(\s*)(?P<context>[^)]+?)(?:\s*\)|$))?
            """,
            candidate_text,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if context_match:
            candidate = _normalize_worker_context_candidate(context_match.group("context"))
            if candidate:
                _record_candidate(candidate, 90)
            break

    message_sources: list[str] = []
    for source in (message, normalized_message):
        if source and source not in message_sources:
            message_sources.append(source)

    for pattern in (
        _WORKER_CONTEXT_PREFIX_PATTERN,
        _WORKER_CONTEXT_KV_PATTERN,
        _WORKER_CONTEXT_RESTART_PATTERN,
        _WORKER_CONTEXT_STALLED_PATTERN,
        _WORKER_CONTEXT_BRACKET_PATTERN,
    ):
        for source in message_sources:
            for match in pattern.finditer(source):
                if "value" in match.groupdict():
                    raw_candidate = match.group("value")
                else:
                    raw_candidate = match.group("context")
                normalized = _normalize_worker_context_candidate(raw_candidate)
                if not normalized:
                    continue
                weight = 20
                key = match.groupdict().get("key", "")
                key_normalized = key.lower() if key else ""
                if key_normalized and any(sep in key_normalized for sep in {".", "-"}):
                    key_normalized = re.split(r"[._-]", key_normalized, 1)[0]
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
                    "channel",
                    "queue",
                    "thread",
                    "engine",
                    "backend",
                    "runner",
                    "target",
                    "namespace",
                    "project",
                    "group",
                    "agent",
                    "executor",
                    "handler",
                }:
                    weight = 55
                elif pattern in {_WORKER_CONTEXT_RESTART_PATTERN, _WORKER_CONTEXT_PREFIX_PATTERN}:
                    weight = 70
                elif pattern is _WORKER_CONTEXT_BRACKET_PATTERN:
                    weight = 65
                _record_candidate(normalized, weight)
            # allow normalized source to contribute when original fails while avoiding duplicate matches

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


def _normalise_worker_error_message(
    raw_value: str,
) -> tuple[str | None, str | None, dict[str, str]]:
    """Return a user-facing worker error description and supplemental metadata."""

    cleaned = _clean_worker_metadata_value(raw_value)
    if not cleaned:
        return None, None, {}

    collapsed = re.sub(r"\s+", " ", cleaned).strip(" .;:,-")
    if not collapsed:
        return None, None, {}

    lowered = collapsed.lower()
    metadata: dict[str, str] = {
        "docker_worker_last_error_original": collapsed,
        "docker_worker_last_error_raw": collapsed,
    }

    for pattern, code, narrative in _WORKER_ERROR_NORMALISERS:
        if pattern.search(lowered):
            metadata["docker_worker_last_error_code"] = code
            metadata["docker_worker_last_error_original"] = narrative
            detail = f"{narrative}."
            return narrative, detail, metadata

    fallback_detail = (
        "Docker Desktop reported the worker error '%s'." % collapsed
    )
    return collapsed, fallback_detail, metadata


def _normalise_worker_metadata_key(raw_key: str) -> str:
    """Return a canonical lowercase key for worker diagnostic attributes."""

    normalised = raw_key.strip().lower()
    if not normalised:
        return normalised
    return re.sub(r"[^a-z0-9]+", "_", normalised)


def _strip_interval_clause_suffix(raw: str) -> str:
    """Trim descriptive tails from interval clauses such as ``"30s due to"``."""

    trimmed = raw.strip()
    if not trimmed:
        return ""

    trimmed = re.split(
        r"\b(?:due|because|caused|cause|owing|reason|while|when|with|after|before|as|pending)\b",
        trimmed,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    if "=" in trimmed:
        key, value = trimmed.split("=", 1)
        key_normalized = key.strip().lower()
        if key_normalized in {
            "backoff",
            "delay",
            "wait",
            "cooldown",
            "interval",
            "duration",
            "next",
            "nextretry",
            "next_restart",
            "nextretryin",
        }:
            trimmed = value

    return trimmed.strip(" \t\n\r,.);:")


def _normalise_approx_prefix(raw: str | None) -> str | None:
    """Return a user-friendly approximation qualifier if *raw* is meaningful."""

    if not raw:
        return None
    lowered = raw.strip().lower()
    if not lowered:
        return None
    if lowered in {"~", "≈"}:
        return "~"
    if lowered.startswith("approx"):
        return "approximately"
    if lowered in {"about", "around", "roughly"}:
        return "about"
    if lowered in {"near", "nearly"}:
        return "nearly"
    return lowered


def _format_go_duration(token: str) -> str | None:
    """Return a human readable representation of Go-style duration *token*."""

    sanitized = token.replace(" ", "")
    if not sanitized:
        return None

    components = list(_GO_DURATION_COMPONENT_PATTERN.finditer(sanitized))
    if not components:
        return None

    reconstructed = "".join(match.group(0) for match in components)
    if reconstructed.lower() != sanitized.lower():
        return None

    normalized_segments: list[str] = []
    for match in components:
        value = match.group("value").lstrip("0") or "0"
        unit = match.group("unit").lower()
        normalized_segments.append(f"{value}{unit}")
    return " ".join(normalized_segments) if normalized_segments else None


def _interpret_clock_duration(value: str) -> tuple[str, float] | None:
    """Return a normalized clock-style duration and its total seconds."""

    candidate = value.strip()
    if not candidate or not _CLOCK_DURATION_PATTERN.match(candidate):
        return None

    parts = candidate.split(":")
    layout = _CLOCK_DURATION_LAYOUTS.get(len(parts))
    if not layout:
        return None

    numeric_parts: list[float] = []
    for token, unit in zip(parts, layout):
        if unit == "seconds":
            try:
                numeric = float(token)
            except ValueError:
                return None
        else:
            try:
                numeric = int(token)
            except ValueError:
                return None
        numeric_parts.append(numeric)

    total_seconds = sum(
        numeric * _CLOCK_DURATION_FACTORS[unit]
        for numeric, unit in zip(numeric_parts, layout)
    )

    segments: list[str] = []
    for numeric, unit in zip(numeric_parts, layout):
        symbol = _CLOCK_DURATION_SYMBOLS[unit]
        if unit == "seconds":
            if numeric == 0 and segments:
                continue
            if abs(numeric - round(numeric)) < 1e-9:
                rendered = str(int(round(numeric)))
            else:
                rendered = ("%g" % numeric).rstrip("0").rstrip(".")
            segments.append(f"{rendered}{symbol}")
        else:
            if numeric == 0:
                continue
            segments.append(f"{int(numeric)}{symbol}")

    if not segments:
        segments.append("0s")

    return " ".join(segments), total_seconds


def _normalise_backoff_hint(value: str) -> str | None:
    """Return a normalised representation of a worker backoff interval."""

    if not value:
        return None

    candidate = value.strip().strip(";.,:)")
    candidate = candidate.strip("()[]{}")
    if not candidate:
        return None

    def _combine(prefix_value: str | None, token: str) -> str:
        if not prefix_value:
            return token
        if prefix_value == "~":
            return f"~{token}".strip()
        return f"{prefix_value} {token}".strip()

    prefix: str | None = None
    prefix_match = _APPROX_PREFIX_PATTERN.match(candidate)
    if prefix_match:
        prefix = _normalise_approx_prefix(prefix_match.group("prefix"))
        candidate = candidate[prefix_match.end() :].lstrip()

    go_candidate = _format_go_duration(candidate)
    if go_candidate:
        normalized = go_candidate
    else:
        clock_candidate = _interpret_clock_duration(candidate)
        if clock_candidate:
            normalized, _ = clock_candidate
        else:
            match = _BACKOFF_INTERVAL_PATTERN.match(candidate)
            if not match:
                combined = _combine(prefix, candidate)
                return combined or None
            number = match.group("number")
            unit = match.group("unit")
            if not number:
                combined = _combine(prefix, candidate)
                return combined or None
            normalized = number
            if unit:
                normalized_unit = _DURATION_UNIT_NORMALISATION.get(unit.lower())
                if normalized_unit:
                    normalized = f"{number}{normalized_unit}"
                else:
                    normalized = f"{number}{unit.strip()}"
    if prefix:
        return _combine(prefix, normalized)
    return normalized.strip() if normalized else None


def _scan_backoff_hint_from_message(message: str) -> str | None:
    """Extract a plausible backoff interval embedded within *message*."""

    if not message:
        return None

    lowered = message.lower()
    if not any(token in lowered for token in {"restart", "retry", "backoff", "stalled"}):
        return None

    for match in _GO_DURATION_PATTERN.finditer(message):
        prefix_fragment = message[: match.start()]
        suffix_match = _APPROX_SUFFIX_PATTERN.search(prefix_fragment)
        if suffix_match:
            candidate_start = suffix_match.start()
        else:
            candidate_start = match.start()
        candidate = message[candidate_start:match.end()]
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    for match in _CLOCK_DURATION_SEARCH_PATTERN.finditer(message):
        prefix_fragment = message[: match.start()]
        suffix_match = _APPROX_SUFFIX_PATTERN.search(prefix_fragment)
        if suffix_match:
            candidate_start = suffix_match.start()
        else:
            candidate_start = match.start()
        candidate = message[candidate_start:match.end()]
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    for match in _BACKOFF_INTERVAL_PATTERN.finditer(message):
        candidate = match.group(0)
        normalized = _normalise_backoff_hint(candidate)
        if normalized and any(char.isalpha() for char in normalized):
            return normalized

    return None


def _extract_worker_flapping_descriptors(
    message: str, *, normalized_message: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Derive human friendly descriptors for flapping Docker workers."""

    context_descriptor: str | None = None
    context_details: list[str] = []
    metadata: dict[str, str] = {}

    restart_count: int | None = None
    last_error: str | None = None
    backoff_hint: str | None = None
    last_seen: str | None = None

    normalized_source = normalized_message or _normalise_worker_stalled_phrase(message)

    envelope = _parse_docker_log_envelope(message)

    context_fields = (
        "context",
        "component",
        "subsystem",
        "worker",
        "module",
        "namespace",
        "service",
        "scope",
        "target",
        "name",
    )
    for field in context_fields:
        candidate = envelope.get(field)
        if not candidate:
            continue
        normalized = _clean_worker_metadata_value(candidate)
        if normalized:
            metadata["docker_worker_context"] = normalized
            break

    def _set_backoff_hint(candidate: str | None) -> None:
        nonlocal backoff_hint
        if backoff_hint is not None or not candidate:
            return
        normalized = _normalise_backoff_hint(candidate)
        if normalized:
            backoff_hint = normalized

    def _ingest_metadata_candidate(key: str | None, value: str | None) -> None:
        nonlocal restart_count, last_error, backoff_hint, last_seen
        if not key or value is None:
            return
        normalized_key = _normalise_worker_metadata_key(key)
        cleaned_value = _clean_worker_metadata_value(value)
        if not normalized_key or not cleaned_value:
            return
        category = _classify_worker_metadata_key(normalized_key)
        if category == "restart" and restart_count is None:
            number_match = re.search(r"(-?\d+)", cleaned_value)
            if number_match:
                try:
                    restart_count = int(number_match.group(1))
                except ValueError:
                    restart_count = None
            return
        if category == "error" and last_error is None:
            last_error = cleaned_value
            return
        if category == "backoff":
            _set_backoff_hint(cleaned_value)
            return
        if category == "last_seen" and last_seen is None:
            last_seen = cleaned_value

    for key, value in envelope.items():
        _ingest_metadata_candidate(key, value)

    for match in _WORKER_METADATA_TOKEN_PATTERN.finditer(message):
        _ingest_metadata_candidate(match.group("key"), match.group("value"))

    if restart_count is None:
        fallback_restart = re.search(
            r"(?:attempts?|retries?|restart(?:s|_count|count)?)(?!ing)\D*(?P<count>\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_restart:
            try:
                restart_count = int(fallback_restart.group("count"))
            except ValueError:
                restart_count = None

    if restart_count is None:
        multiplier_match = re.search(r"(?<![0-9A-Za-z])[x×]\s*(?P<count>\d+)", message)
        if multiplier_match:
            try:
                restart_count = int(multiplier_match.group("count"))
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

    if last_error is None:
        due_match = re.search(
            r"(?:due to|because(?: of)?)\s+(?P<reason>[^;.,()]+)",
            normalized_source,
            flags=re.IGNORECASE,
        )
        if due_match:
            candidate = _clean_worker_metadata_value(due_match.group("reason"))
            if candidate:
                last_error = candidate

    if backoff_hint is None:
        fallback_backoff = re.search(
            r"backoff\s*[=:]\s*(?P<value>\"[^\"]+\"|'[^']+'|[^;\n]+)",
            message,
            flags=re.IGNORECASE,
        )
        if fallback_backoff:
            _set_backoff_hint(_clean_worker_metadata_value(fallback_backoff.group("value")))

    if backoff_hint is None:
        interval_match = re.search(
            r"""
            worker\s+stalled
            (?:(?:\s*(?:[;:,.\-–—]|->|=>|→|⇒)\s*)*)
            restart(?:ing)?
            \s+(?:in|after)\s+
            (?P<interval>[^;.,()\n]+)
            """,
            normalized_source,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if not interval_match:
            interval_match = re.search(
                r"re(?:start(?:ing)?|starting)\s+(?:in|after)\s+(?P<interval>[^;.,()\n]+)",
                normalized_source,
                flags=re.IGNORECASE,
            )
        if interval_match:
            interval = interval_match.group("interval")
            if interval:
                cleaned_interval = _strip_interval_clause_suffix(interval)
                if cleaned_interval:
                    _set_backoff_hint(cleaned_interval)

    if backoff_hint is None:
        derived_backoff = _scan_backoff_hint_from_message(normalized_source)
        if derived_backoff:
            backoff_hint = derived_backoff

    if "docker_worker_context" not in metadata:
        cleaned_message = re.sub(
            r"\s+", " ", _strip_control_sequences(normalized_source)
        ).strip()
        context_candidate = _extract_worker_context(normalized_source, cleaned_message)
        if context_candidate:
            metadata["docker_worker_context"] = context_candidate

    context_value = metadata.get("docker_worker_context")
    if context_value:
        context_descriptor = f"Affected component: {context_value}."

    if restart_count is not None and restart_count >= 0:
        metadata["docker_worker_restart_count"] = str(restart_count)
        plural = "s" if restart_count != 1 else ""
        context_details.append(
            f"Docker reported {restart_count} restart{plural} during diagnostics."
        )

    if backoff_hint:
        metadata["docker_worker_backoff"] = backoff_hint
        context_details.append(
            f"Docker advertised a restart backoff interval of {backoff_hint}."
        )

    if last_seen:
        metadata["docker_worker_last_restart"] = last_seen
        context_details.append(
            f"Last restart marker emitted by Docker: {last_seen}."
        )

    if last_error:
        normalized_error, error_detail, error_metadata = _normalise_worker_error_message(
            last_error
        )
        if normalized_error:
            metadata["docker_worker_last_error"] = normalized_error
            metadata.update(error_metadata)
            context_details.append(
                error_detail
                or f"Most recent worker error: {normalized_error}."
            )
    else:
        fallback_source = normalized_source or message
        fallback_error, fallback_detail, fallback_metadata = (
            _normalise_worker_error_message(fallback_source)
            if fallback_source
            else (None, None, {})
        )
        if fallback_error:
            metadata.setdefault("docker_worker_last_error", fallback_error)
            for key, value in fallback_metadata.items():
                metadata.setdefault(key, value)
            if fallback_detail and fallback_detail not in context_details:
                context_details.append(fallback_detail)

    descriptors: list[str] = []
    if context_descriptor:
        descriptors.append(context_descriptor)
    if context_details:
        if context_descriptor:
            descriptors.append(
                "Additional context: " + " ".join(context_details)
            )
        else:
            descriptors.extend(context_details)

    return descriptors, metadata


def _normalise_docker_warning(message: str) -> tuple[str | None, dict[str, str]]:
    """Return a cleaned warning and metadata extracted from Docker output."""

    cleaned = _DOCKER_WARNING_PREFIX_PATTERN.sub("", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None, {}

    metadata: dict[str, str] = {}
    normalized_cleaned = _normalise_worker_stalled_phrase(cleaned)
    if "worker stalled" in normalized_cleaned.lower():
        metadata["docker_worker_health"] = "flapping"

        normalized_original = _normalise_worker_stalled_phrase(message)
        descriptors, worker_metadata = _extract_worker_flapping_descriptors(
            message, normalized_message=normalized_original
        )
        metadata.update(worker_metadata)

        headline = "Docker Desktop reported repeated restarts of a background worker."
        remediation = (
            "Restart Docker Desktop, ensure Hyper-V or WSL 2 virtualization is enabled, and "
            "allocate additional CPU/RAM to the Docker VM before retrying."
        )

        segments: list[str] = [headline]
        if descriptors:
            segments.extend(descriptors)
        segments.append(remediation)
        cleaned = " ".join(segment.strip() for segment in segments if segment.strip())

    return cleaned, metadata


@dataclass
class _WorkerWarningRecord:
    """Capture restart telemetry for an individual Docker worker."""

    context: str | None
    restart_count: int | None = None
    backoff_hint: str | None = None
    backoff_seconds: float | None = None
    last_seen: str | None = None
    last_error: str | None = None
    last_error_original: str | None = None
    last_error_raw: str | None = None
    occurrences: int = 0
    restart_samples: list[int] = field(default_factory=list)
    backoff_hints: list[str] = field(default_factory=list)
    last_seen_samples: list[str] = field(default_factory=list)
    last_error_samples: list[str] = field(default_factory=list)
    last_error_original_samples: list[str] = field(default_factory=list)
    last_error_raw_samples: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)

    def update(self, metadata: Mapping[str, str]) -> None:
        """Merge ``metadata`` gleaned from a worker warning into the record."""

        self.occurrences += 1

        context = metadata.get("docker_worker_context")
        if context and not self.context:
            self.context = context.strip()

        restart_value = metadata.get("docker_worker_restart_count")
        restart_count = _coerce_optional_int(restart_value)
        if restart_count is not None:
            self.restart_samples.append(restart_count)
            if self.restart_count is None or restart_count > self.restart_count:
                self.restart_count = restart_count

        backoff_hint = metadata.get("docker_worker_backoff")
        if backoff_hint:
            normalized_hint = backoff_hint.strip()
            if normalized_hint:
                self.backoff_hints.append(normalized_hint)
                candidate_seconds = _estimate_backoff_seconds(normalized_hint)
                if candidate_seconds is not None:
                    if (
                        self.backoff_seconds is None
                        or candidate_seconds > self.backoff_seconds
                        or (
                            candidate_seconds == self.backoff_seconds
                            and not self.backoff_hint
                        )
                    ):
                        self.backoff_seconds = candidate_seconds
                        self.backoff_hint = normalized_hint
                elif not self.backoff_hint:
                    self.backoff_hint = normalized_hint

        last_restart = metadata.get("docker_worker_last_restart")
        if last_restart:
            cleaned_restart = last_restart.strip()
            if cleaned_restart:
                self.last_seen_samples.append(cleaned_restart)
                self.last_seen = cleaned_restart

        last_error = metadata.get("docker_worker_last_error")
        if last_error:
            cleaned_error = last_error.strip()
            if cleaned_error:
                self.last_error_samples.append(cleaned_error)
                self.last_error = cleaned_error

        original_error = metadata.get("docker_worker_last_error_original")
        if original_error:
            cleaned_original = original_error.strip()
            if cleaned_original:
                self.last_error_original_samples.append(cleaned_original)
                self.last_error_original = cleaned_original

        raw_error = metadata.get("docker_worker_last_error_raw")
        if raw_error:
            cleaned_raw = raw_error.strip()
            if cleaned_raw:
                self.last_error_raw_samples.append(cleaned_raw)
                self.last_error_raw = cleaned_raw

        error_code = metadata.get("docker_worker_last_error_code")
        if error_code:
            normalized_code = error_code.strip()
            if normalized_code:
                self.error_codes.append(normalized_code)


class _WorkerWarningAggregator:
    """Accumulate worker restart telemetry across multiple Docker warnings."""

    def __init__(self) -> None:
        self._records: dict[str, _WorkerWarningRecord] = {}
        self._order: list[str] = []
        self._health: str | None = None

    def ingest(self, metadata: Mapping[str, str]) -> None:
        """Record ``metadata`` emitted by ``_normalise_docker_warning``."""

        if not metadata:
            return

        health = metadata.get("docker_worker_health")
        if health and not self._health:
            self._health = health

        context = metadata.get("docker_worker_context")
        normalized_context = context.strip() if isinstance(context, str) else None
        key = normalized_context.casefold() if normalized_context else "__anonymous"

        record = self._records.get(key)
        if record is None:
            record = _WorkerWarningRecord(context=normalized_context)
            self._records[key] = record
            self._order.append(key)
        else:
            if normalized_context and not record.context:
                record.context = normalized_context

        record.update(metadata)

    def finalize(self) -> dict[str, str]:
        """Produce a consolidated metadata mapping for downstream diagnostics."""

        result: dict[str, str] = {}
        if self._health:
            result["docker_worker_health"] = self._health

        records = [self._records[key] for key in self._order if self._records[key]]
        if not records:
            return result

        primary = self._select_primary_record(records)

        total_occurrences = sum(record.occurrences for record in records)
        if total_occurrences:
            result["docker_worker_warning_occurrences"] = str(total_occurrences)

        context_occurrence_entries = [
            f"{record.context}:{record.occurrences}"
            for record in records
            if record.context and record.occurrences
        ]
        if context_occurrence_entries:
            result["docker_worker_context_occurrences"] = ", ".join(
                context_occurrence_entries
            )

        contexts = _coalesce_iterable(
            [record.context for record in records if record.context]
        )
        if primary and primary.context:
            result["docker_worker_context"] = primary.context
        elif contexts:
            result["docker_worker_context"] = contexts[0]
        if len(contexts) > 1:
            result["docker_worker_contexts"] = ", ".join(contexts)

        restart_samples = sorted(
            {
                sample
                for record in records
                for sample in record.restart_samples
                if sample is not None
            }
        )
        if primary and primary.restart_count is not None:
            result["docker_worker_restart_count"] = str(primary.restart_count)
        elif restart_samples:
            result["docker_worker_restart_count"] = str(restart_samples[-1])
        if len(restart_samples) > 1:
            result["docker_worker_restart_count_samples"] = ", ".join(
                str(sample) for sample in restart_samples
            )

        backoff_hints = _coalesce_iterable(
            [hint for record in records for hint in record.backoff_hints]
        )
        if primary and primary.backoff_hint:
            result["docker_worker_backoff"] = primary.backoff_hint
        elif backoff_hints:
            backoff_candidates = [
                (hint, _estimate_backoff_seconds(hint)) for hint in backoff_hints
            ]
            chosen_hint, _ = max(
                backoff_candidates,
                key=lambda item: (
                    item[1] is not None,
                    item[1] or 0.0,
                    len(item[0]),
                ),
            )
            result["docker_worker_backoff"] = chosen_hint
        if len(backoff_hints) > 1:
            result["docker_worker_backoff_options"] = ", ".join(backoff_hints)

        last_restart_markers = _coalesce_iterable(
            [marker for record in records for marker in record.last_seen_samples]
        )
        if primary and primary.last_seen:
            result["docker_worker_last_restart"] = primary.last_seen
        elif last_restart_markers:
            result["docker_worker_last_restart"] = last_restart_markers[-1]
        if len(last_restart_markers) > 1:
            result["docker_worker_last_restart_samples"] = ", ".join(
                last_restart_markers
            )

        last_errors = _coalesce_iterable(
            [error for record in records for error in record.last_error_samples]
        )
        if primary and primary.last_error:
            result["docker_worker_last_error"] = primary.last_error
        elif last_errors:
            result["docker_worker_last_error"] = last_errors[-1]
        if len(last_errors) > 1:
            result["docker_worker_last_error_samples"] = "; ".join(last_errors)

        original_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_original_samples
            ]
        )
        if primary and primary.last_error_original:
            result["docker_worker_last_error_original"] = primary.last_error_original
        elif original_errors:
            result["docker_worker_last_error_original"] = original_errors[-1]
        if len(original_errors) > 1:
            result["docker_worker_last_error_original_samples"] = "; ".join(
                original_errors
            )

        raw_errors = _coalesce_iterable(
            [
                error
                for record in records
                for error in record.last_error_raw_samples
            ]
        )
        if primary and primary.last_error_raw:
            result["docker_worker_last_error_raw"] = primary.last_error_raw
        elif raw_errors:
            result["docker_worker_last_error_raw"] = raw_errors[-1]
        if len(raw_errors) > 1:
            result["docker_worker_last_error_raw_samples"] = "; ".join(raw_errors)

        error_codes = _coalesce_iterable(
            [code for record in records for code in record.error_codes]
        )
        if error_codes:
            result["docker_worker_last_error_code"] = error_codes[0]
        if len(error_codes) > 1:
            result["docker_worker_last_error_codes"] = ", ".join(error_codes)

        return result

    @staticmethod
    def _select_primary_record(
        records: Sequence[_WorkerWarningRecord],
    ) -> _WorkerWarningRecord | None:
        if not records:
            return None

        return max(
            records,
            key=lambda record: (
                record.restart_count is not None,
                record.restart_count or 0,
                record.backoff_seconds is not None,
                record.backoff_seconds or 0.0,
                1 if record.last_error else 0,
                len(record.context or ""),
            ),
        )


def _normalize_warning_collection(messages: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Normalise warning ``messages`` and capture associated metadata."""

    normalized: list[str] = []
    metadata: dict[str, str] = {}
    seen: set[str] = set()

    worker_aggregator = _WorkerWarningAggregator()

    for message in messages:
        cleaned, extracted = _normalise_docker_warning(message)
        if extracted:
            worker_aggregator.ingest(extracted)
            for key, value in extracted.items():
                if key.startswith("docker_worker_"):
                    continue
                metadata[key] = value
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)

    worker_metadata = worker_aggregator.finalize()
    metadata.update(worker_metadata)

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
    candidate = value.strip()
    if not candidate:
        return None

    candidate = candidate.strip(";.,:)")
    candidate = candidate.strip("()[]{}")
    candidate = candidate.strip()
    if not candidate:
        return None

    prefix_match = _APPROX_PREFIX_PATTERN.match(candidate)
    if prefix_match:
        candidate = candidate[prefix_match.end() :].lstrip()
    suffix_match = _APPROX_SUFFIX_PATTERN.search(candidate)
    if suffix_match and suffix_match.end() == len(candidate):
        candidate = candidate[: suffix_match.start()].rstrip()

    if not candidate:
        return None

    condensed = candidate.replace(" ", "")
    go_components = list(_GO_DURATION_COMPONENT_PATTERN.finditer(condensed))
    if go_components:
        reconstructed = "".join(match.group(0) for match in go_components)
        if reconstructed.lower() == condensed.lower():
            total = 0.0
            for match in go_components:
                try:
                    numeric = float(match.group("value"))
                except (TypeError, ValueError):
                    return None
                unit = match.group("unit").lower()
                if unit == "h":
                    total += numeric * 3600.0
                elif unit == "m":
                    total += numeric * 60.0
                elif unit == "s":
                    total += numeric
                else:  # pragma: no cover - defensive
                    return None
            return total

    clock_candidate = _interpret_clock_duration(candidate)
    if clock_candidate:
        _, seconds = clock_candidate
        return seconds

    match = _BACKOFF_INTERVAL_PATTERN.search(candidate)
    if not match:
        return None
    raw = match.group("number")
    unit = (match.group("unit") or "s").lower()
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None

    if unit in {"ms", "msec", "milliseconds"}:
        return numeric / 1000.0
    if unit in {"s", "sec", "secs", "seconds"}:
        return numeric
    if unit in {"m", "min", "mins", "minutes"}:
        return numeric * 60.0
    if unit in {"h", "hr", "hrs", "hours"}:
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
    last_error_original: str | None = None
    last_error_raw: str | None = None
    warning_occurrences: int = 0
    context_occurrences: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    contexts: tuple[str, ...] = field(default_factory=tuple)
    restart_samples: tuple[int, ...] = field(default_factory=tuple)
    backoff_options: tuple[str, ...] = field(default_factory=tuple)
    last_restart_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_original_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_raw_samples: tuple[str, ...] = field(default_factory=tuple)
    last_error_codes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> "WorkerRestartTelemetry":
        context = metadata.get("docker_worker_context")
        contexts = _split_metadata_values(metadata.get("docker_worker_contexts"))
        if context:
            contexts = _coalesce_iterable([context, *contexts])
        restart_samples = _parse_int_sequence(
            metadata.get("docker_worker_restart_count_samples")
        )
        raw_backoff_options = [
            _normalise_backoff_hint(option)
            for option in _split_metadata_values(
                metadata.get("docker_worker_backoff_options")
            )
            if option
        ]
        backoff_options = _coalesce_iterable(
            option for option in raw_backoff_options if option
        )
        last_restart_samples = tuple(
            _split_metadata_values(metadata.get("docker_worker_last_restart_samples"))
        )
        last_error_samples = tuple(
            _split_metadata_values(metadata.get("docker_worker_last_error_samples"))
        )
        last_error_original = metadata.get("docker_worker_last_error_original")
        last_error_original_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_original_samples")
            )
        )
        last_error_raw = metadata.get("docker_worker_last_error_raw")
        last_error_raw_samples = tuple(
            _split_metadata_values(
                metadata.get("docker_worker_last_error_raw_samples")
            )
        )
        error_codes = _split_metadata_values(
            metadata.get("docker_worker_last_error_codes")
        )
        primary_code = metadata.get("docker_worker_last_error_code")
        if primary_code:
            error_codes = _coalesce_iterable([primary_code, *error_codes])
        else:
            error_codes = _coalesce_iterable(error_codes)

        normalized_backoff = metadata.get("docker_worker_backoff")
        if normalized_backoff:
            normalized_backoff = _normalise_backoff_hint(normalized_backoff)

        warning_occurrences = _coerce_optional_int(
            metadata.get("docker_worker_warning_occurrences")
        ) or 0

        raw_context_occurrences = metadata.get("docker_worker_context_occurrences")
        context_occurrence_pairs: list[tuple[str, int]] = []
        for token in _split_metadata_values(raw_context_occurrences):
            if ":" not in token:
                continue
            name, raw_count = token.split(":", 1)
            cleaned_name = _clean_worker_metadata_value(name)
            if not cleaned_name:
                continue
            try:
                count_value = int(raw_count.strip())
            except ValueError:
                continue
            context_occurrence_pairs.append((cleaned_name, count_value))

        return cls(
            context=context,
            restart_count=_coerce_optional_int(
                metadata.get("docker_worker_restart_count")
            ),
            backoff_hint=normalized_backoff,
            last_seen=metadata.get("docker_worker_last_restart"),
            last_error=metadata.get("docker_worker_last_error"),
            last_error_original=last_error_original,
            last_error_raw=last_error_raw,
            warning_occurrences=warning_occurrences,
            context_occurrences=tuple(context_occurrence_pairs),
            contexts=tuple(contexts),
            restart_samples=restart_samples,
            backoff_options=tuple(backoff_options),
            last_restart_samples=last_restart_samples,
            last_error_samples=last_error_samples,
            last_error_original_samples=tuple(last_error_original_samples),
            last_error_raw_samples=tuple(last_error_raw_samples),
            last_error_codes=tuple(error_codes),
        )

    @property
    def backoff_seconds(self) -> float | None:
        """Best-effort conversion of the Docker restart backoff to seconds."""

        return _estimate_backoff_seconds(self.backoff_hint)

    @property
    def max_restart_count(self) -> int | None:
        """Return the highest restart count observed across metadata samples."""

        candidates: list[int] = []
        if self.restart_count is not None:
            candidates.append(self.restart_count)
        candidates.extend(self.restart_samples)
        if not candidates:
            return None
        return max(candidates)

    @property
    def all_contexts(self) -> tuple[str, ...]:
        """Return unique worker contexts with the primary context prioritised."""

        contexts: list[str] = []
        if self.context:
            contexts.append(self.context)
        contexts.extend(self.contexts)
        if not contexts:
            return ()
        return tuple(_coalesce_iterable(contexts))

    @property
    def all_last_restarts(self) -> tuple[str, ...]:
        """Return the set of observed restart markers."""

        markers: list[str] = []
        if self.last_seen:
            markers.append(self.last_seen)
        markers.extend(self.last_restart_samples)
        if not markers:
            return ()
        return tuple(_coalesce_iterable(markers))

    @property
    def all_last_errors(self) -> tuple[str, ...]:
        """Return the set of observed error messages for the worker."""

        errors: list[str] = []
        if self.last_error:
            errors.append(self.last_error)
        errors.extend(self.last_error_samples)
        if not errors:
            return ()
        return tuple(_coalesce_iterable(errors))

    @property
    def max_backoff_seconds(self) -> float | None:
        """Return the slowest restart backoff advertised by Docker."""

        candidates: list[float] = []
        primary = self.backoff_seconds
        if primary is not None:
            candidates.append(primary)
        for option in self.backoff_options:
            seconds = _estimate_backoff_seconds(option)
            if seconds is not None:
                candidates.append(seconds)
        if not candidates:
            return None
        return max(candidates)


@dataclass(frozen=True)
class WorkerHealthAssessment:
    """Classification of Docker worker restart telemetry."""

    severity: Literal["warning", "error"]
    headline: str
    details: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def render(self) -> str:
        """Compose a human readable message from the assessment components."""

        segments = [self.headline]
        detail_segments: list[str] = []
        if self.details:
            detail_segments.extend(detail.strip() for detail in self.details if detail)
        if self.reasons:
            detail_segments.extend(reason.strip() for reason in self.reasons if reason)
        if detail_segments:
            segments.append(" ".join(detail_segments))
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
    severity_reasons: list[str] = []
    severity_reason_keys: set[str] = set()

    def _register_reason(key: str, message: str) -> None:
        if key in severity_reason_keys:
            return
        severity_reason_keys.add(key)
        normalized = message.rstrip()
        if not normalized.endswith("."):
            normalized += "."
        severity_reasons.append(normalized)

    contexts = telemetry.all_contexts
    if contexts:
        if len(contexts) == 1:
            details.append(f"Affected component: {contexts[0]}.")
        else:
            joined = ", ".join(contexts)
            details.append(f"Affected components: {joined}.")

    occurrence_count = telemetry.warning_occurrences
    if occurrence_count:
        plural = "s" if occurrence_count != 1 else ""
        details.append(
            f"Docker emitted {occurrence_count} worker stall warning{plural} during diagnostics."
        )
        if occurrence_count >= 4:
            _register_reason(
                "warning_frequency",
                "Docker emitted four or more worker stall warnings during a single diagnostics run",
            )

    if telemetry.context_occurrences:
        rendered_context_occurrences = ", ".join(
            f"{name} ({count})" for name, count in telemetry.context_occurrences
        )
        details.append(
            f"Warning frequency by component: {rendered_context_occurrences}."
        )

    max_restart = telemetry.max_restart_count
    if max_restart is not None:
        plural = "s" if max_restart != 1 else ""
        details.append(
            f"Docker recorded up to {max_restart} restart{plural} during diagnostics."
        )
        additional_samples = [
            sample
            for sample in telemetry.restart_samples
            if sample != max_restart
        ]
        if additional_samples:
            rendered = ", ".join(str(sample) for sample in additional_samples)
            details.append(
                f"Additional restart counts observed across repeated runs: {rendered}."
            )
        if max_restart >= 6:
            _register_reason(
                "excessive_restarts",
                "Docker recorded at least six worker restarts during diagnostics",
            )
    elif telemetry.restart_samples:
        rendered = ", ".join(str(sample) for sample in telemetry.restart_samples)
        details.append(f"Docker reported restart counts during diagnostics: {rendered}.")

    backoff_hint = telemetry.backoff_hint
    if backoff_hint:
        details.append(f"Backoff interval advertised by Docker: {backoff_hint}.")
    extra_backoff = [
        option
        for option in telemetry.backoff_options
        if option and option != backoff_hint
    ]
    if extra_backoff:
        rendered = ", ".join(extra_backoff)
        details.append(f"Additional backoff intervals observed: {rendered}.")

    max_backoff_seconds = telemetry.max_backoff_seconds
    if max_backoff_seconds is not None and max_backoff_seconds >= 60:
        descriptor = backoff_hint or extra_backoff[0] if extra_backoff else None
        if descriptor is None:
            descriptor = f"approximately {int(round(max_backoff_seconds))}s"
        _register_reason(
            "prolonged_backoff",
            f"Docker advertised a restart backoff of at least {descriptor}, indicating sustained recovery attempts",
        )

    restart_markers = telemetry.all_last_restarts
    if restart_markers:
        if len(restart_markers) == 1:
            details.append(f"Last restart marker: {restart_markers[0]}.")
        else:
            preview = restart_markers[:3]
            rendered = ", ".join(preview)
            if len(restart_markers) > len(preview):
                rendered += ", …"
            details.append(f"Restart markers captured from Docker diagnostics: {rendered}.")

    last_errors = telemetry.all_last_errors
    if last_errors:
        details.append(f"Most recent worker error: {last_errors[0]}.")
        if len(last_errors) > 1:
            preview = last_errors[1:3]
            rendered = ", ".join(preview)
            if len(last_errors) > 3:
                rendered += ", …"
            details.append(f"Additional errors encountered: {rendered}.")

    critical_errors = [message for message in last_errors if _is_critical_worker_error(message)]
    if critical_errors:
        _register_reason(
            "critical_error",
            f"Docker reported a critical worker error: {critical_errors[0]}",
        )

    error_codes = tuple(_coalesce_iterable(telemetry.last_error_codes))
    if error_codes:
        primary_code = error_codes[0]
        label = _WORKER_ERROR_CODE_LABELS.get(
            primary_code,
            primary_code.replace("_", " "),
        )
        details.append(
            f"Docker categorised the recent worker issue as {label}."
        )

        code_reason_map = {
            "restart_loop": "Docker classified the worker as being stuck in a restart loop",
            "healthcheck_failure": "Docker reported repeated health-check failures for the worker",
        }
        for code in error_codes:
            reason = code_reason_map.get(code)
            if reason:
                _register_reason(f"code:{code}", reason)

        sustained_backoff = max_backoff_seconds is not None and max_backoff_seconds >= 30
        if "stalled_restart" in error_codes and (
            (max_restart is not None and max_restart >= 3)
            or occurrence_count >= 2
            or sustained_backoff
        ):
            _register_reason(
                "persistent_stalls",
                "Docker repeatedly restarted the worker after stalls, indicating instability",
            )

    severity: Literal["warning", "error"] = "error" if severity_reasons else "warning"

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
        reasons=tuple(severity_reasons),
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
    additional_metadata: dict[str, str] = {
        "docker_worker_health_severity": assessment.severity,
    }
    if assessment.reasons:
        additional_metadata["docker_worker_health_reasons"] = "; ".join(
            assessment.reasons
        )

    virtualization_warnings: list[str] = []
    virtualization_errors: list[str] = []
    virtualization_metadata: dict[str, str] = {}

    if context.is_wsl or context.is_windows:
        vw_warnings, vw_errors, vw_metadata = _collect_windows_virtualization_insights(
            timeout=timeout
        )
        virtualization_warnings.extend(vw_warnings)
        virtualization_errors.extend(vw_errors)
        virtualization_metadata.update(vw_metadata)

    if assessment.severity == "warning":
        warnings.append(assessment.render())
        if virtualization_warnings:
            warnings.extend(virtualization_warnings)
        if virtualization_errors:
            warnings.extend(
                f"Virtualization issue detected: {message}"
                for message in virtualization_errors
            )
        if virtualization_metadata:
            additional_metadata.update(virtualization_metadata)
        return warnings, errors, additional_metadata

    errors.append(assessment.render())

    if virtualization_warnings:
        warnings.extend(virtualization_warnings)
    if virtualization_errors:
        errors.extend(virtualization_errors)
    if virtualization_metadata:
        additional_metadata.update(virtualization_metadata)

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
