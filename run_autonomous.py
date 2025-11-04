from __future__ import annotations

# flake8: noqa

"""Wrapper for running the autonomous sandbox loop after dependency checks.

Initialises :data:`GLOBAL_ROUTER` via :func:`init_db_router` before importing
modules that touch the database.
"""

import os

# ``knowledge.local_knowledge`` was used in historical deployments, however the
# sandbox now ships ``local_knowledge_module`` directly within this repository.
# Import from the local module first and gracefully degrade when the legacy
# package is unavailable to keep the script usable in both layouts.
_LOCAL_KNOWLEDGE_LOADER = None


def _resolve_local_knowledge_loader():
    global _LOCAL_KNOWLEDGE_LOADER
    if _LOCAL_KNOWLEDGE_LOADER is not None:
        return _LOCAL_KNOWLEDGE_LOADER
    loaders = (
        "local_knowledge_module",
        "menace_sandbox.local_knowledge_module",
        "knowledge.local_knowledge",
    )
    for module_name in loaders:
        try:  # pragma: no cover - exercised in integration environments
            module = __import__(module_name, fromlist=["init_local_knowledge"])
        except ModuleNotFoundError:
            continue
        init_func = getattr(module, "init_local_knowledge", None)
        if callable(init_func):
            _LOCAL_KNOWLEDGE_LOADER = init_func
            return init_func
    raise ModuleNotFoundError(
        "Unable to locate a local knowledge module. Checked: "
        + ", ".join(loaders)
    )


def init_local_knowledge(mem_db: str | os.PathLike[str] | None = None):
    loader = _resolve_local_knowledge_loader()
    if mem_db is None:
        mem_db = os.getenv("GPT_MEMORY_DB", "gpt_memory.db")
    resolved_path = _expand_path(mem_db)
    return loader(resolved_path)

import argparse
import atexit
import contextlib
import importlib
import importlib.util
import inspect
import json
import logging
from collections.abc import Iterable
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import _thread
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Callable, ClassVar, List, Mapping
import math
import uuid
# ``scipy`` is optional in lightweight developer environments (especially on
# Windows) and can be expensive to build.  The runner only needs access to the
# cumulative distribution function for the Student's t-distribution when
# estimating convergence confidence.  Fall back to a normal approximation when
# ``scipy`` is unavailable so the script remains runnable on machines without
# the dependency.
try:  # pragma: no cover - exercised in integration tests
    from scipy.stats import t as _scipy_student_t
except Exception:  # pragma: no cover - optional dependency missing
    _scipy_student_t = None
    try:
        from statistics import NormalDist as _NormalDist
    except Exception:  # pragma: no cover - Python < 3.8 or minimal builds
        _NormalDist = None
else:
    _NormalDist = None


def _student_t_cdf(x: float, df: float) -> float:
    """Return the two-tailed CDF for a Student's t-statistic.

    ``scipy`` provides the most accurate calculation.  When it is unavailable
    (common on Windows workstations) approximate the distribution using the
    normal distribution scaled by ``sqrt(df / max(df - 2, 1))`` which keeps the
    tails reasonably close for moderate degrees of freedom.
    """

    if _scipy_student_t is not None:
        return float(_scipy_student_t.cdf(x, df))

    if _NormalDist is None:
        # Last resort approximation using the error function; ``NormalDist`` is
        # unavailable on very old Python versions.  ``math.erf`` is always
        # present and provides a usable estimate.
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    scale = math.sqrt(df / max(df - 2.0, 1.0))
    return _NormalDist().cdf(x / scale)
from db_router import init_db_router

try:  # pragma: no cover - exercised indirectly in tests
    import dynamic_path_router as _dynamic_path_router
except Exception as exc:  # pragma: no cover - fail fast when routing unavailable
    raise ImportError("dynamic_path_router is required to locate repository paths") from exc


def _callable(attr, default):
    if callable(attr):
        return attr
    if attr is None:
        return default
    return lambda *_, **__: attr


resolve_path = getattr(_dynamic_path_router, "resolve_path")
_fallback_root = lambda: Path(__file__).resolve().parent  # noqa: E731
get_project_root = _callable(
    getattr(_dynamic_path_router, "get_project_root", None), _fallback_root
)
repo_root = _callable(getattr(_dynamic_path_router, "repo_root", None), _fallback_root)
if repo_root is _fallback_root and get_project_root is not _fallback_root:
    repo_root = get_project_root
path_for_prompt = getattr(
    _dynamic_path_router,
    "path_for_prompt",
    lambda name: resolve_path(name).as_posix(),
)
from sandbox_settings import SandboxSettings
from dependency_hints import format_system_package_instructions


def _extract_flag_value(argv: List[str], flag: str) -> str | None:
    """Return the value provided for ``flag`` in ``argv`` if present."""

    flag_eq = f"{flag}="
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if item.startswith(flag_eq):
            return item[len(flag_eq) :]
    return None


def _sync_sandbox_environment(base_path: Path) -> None:
    """Update ``sandbox_runner.environment`` paths to point under ``base_path``."""

    env_mod = sys.modules.get("sandbox_runner.environment")
    if env_mod is None:
        return

    default_active = str(base_path / "active_overlays.json")
    default_failed = str(base_path / "failed_overlays.json")

    env_path_func = getattr(env_mod, "_env_path", None)
    log = logging.getLogger(__name__)

    def _resolve_env_path(name: str, default: str) -> Path:
        raw_value: str | os.PathLike[str] | None
        if callable(env_path_func):
            try:  # pragma: no cover - exercised in integration tests
                raw_value = env_path_func(name, default)
            except FileNotFoundError:
                raw_value = os.environ.get(name, default)
            except Exception:  # pragma: no cover - defensive guard
                log.debug(
                    "sandbox environment shim rejected %s=%r; using default",
                    name,
                    os.environ.get(name, default),
                    exc_info=True,
                )
                raw_value = os.environ.get(name, default)
        else:
            raw_value = os.environ.get(name, default)

        if raw_value is None:
            raw_value = default

        try:
            return _expand_path(raw_value)
        except Exception:
            log.debug(
                "unable to expand sandbox path %s=%r; falling back to %s",
                name,
                raw_value,
                default,
                exc_info=True,
            )
            return _expand_path(default)

    active_path = _resolve_env_path("SANDBOX_ACTIVE_OVERLAYS", default_active)
    failed_path = _resolve_env_path("SANDBOX_FAILED_OVERLAYS", default_failed)

    env_mod._ACTIVE_OVERLAYS_FILE = active_path
    env_mod._FAILED_OVERLAYS_FILE = failed_path

    filelock_cls = getattr(env_mod, "FileLock", None)
    if filelock_cls is not None:
        lock_path = str(env_mod._ACTIVE_OVERLAYS_FILE) + ".lock"
        try:
            env_mod._ACTIVE_OVERLAYS_LOCK = filelock_cls(lock_path)
        except Exception:
            env_mod._ACTIVE_OVERLAYS_LOCK = None


def _console(message: str) -> None:
    """Emit high-visibility progress output for ``run_autonomous`` steps."""

    print(f"[RUN_AUTONOMOUS] {message}", flush=True)


def _basic_setup_logging(*, level: str | int | None = None) -> None:
    """Fallback logger configuration when runtime imports are deferred."""

    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        resolved = level
    else:
        resolved = logging.INFO
    logging.basicConfig(level=resolved)


def _expand_path(value: str | os.PathLike[str]) -> Path:
    """Return ``value`` as a :class:`Path` with user and env vars expanded."""

    original_value = os.fspath(value)
    raw = original_value
    if raw.startswith("~\\"):
        # Windows shells commonly emit ``~\`` when expanding home shortcuts.
        # Convert the leading separator to a forward slash so ``expanduser``
        # produces a canonical path even when executed on POSIX hosts.  UNC
        # paths (``~\\\\server``) retain their leading double backslash while
        # conventional paths strip redundant separators.
        tail = raw[2:]
        if tail.startswith("\\\\"):
            raw = "~/" + tail
        else:
            raw = "~/" + tail.lstrip(r"\\/")
    escaped_pattern = r"%%([^%]+)%%"
    token_pattern = r"%(?!%)([^%]+?)(?<!\\)%"

    # Protect escaped tokens (``%%VAR%%``) by substituting temporary sentinels
    # before performing any environment or user expansions.  ``expandvars`` and
    # ``expanduser`` can change the length of the string which made the previous
    # span-based detection unreliable on Windows where paths such as
    # ``~\\%%USERPROFILE%%`` are common.  Using sentinels keeps the placeholders
    # intact regardless of earlier substitutions.
    sentinel_map: dict[str, str] = {}

    def _protect(match: re.Match[str]) -> str:
        token = f"__RUN_AUTONOMOUS_ESC_{len(sentinel_map)}__"
        sentinel_map[token] = f"%{match.group(1)}%"
        return token

    protected = re.sub(escaped_pattern, _protect, raw)
    expanded = os.path.expandvars(os.path.expanduser(protected))

    def _extract_windows_path(candidate: str, *, original: str) -> str:
        """Return a canonical Windows path when ``candidate`` embeds one.

        Windows callers frequently combine ``~`` expansion with environment
        variables (for example ``~\\%USERPROFILE%\\data``).  When the
        environment variable resolves to an absolute Windows path the default
        :func:`os.path.expanduser` implementation (executed on POSIX during
        tests) prefixes it with the current home directory, yielding strings such
        as ``/root/C:\\Users\\Example``.  Detect embedded drive letters or UNC
        prefixes and slice away the spurious prefix so Windows hosts receive the
        intended path once this logic runs there.
        """

        # Normal drive letter (``C:\``) handling.  ``expanduser`` running on
        # POSIX platforms sometimes normalises the separator after the drive to
        # a forward slash (``C:/``) which previously slipped past the regex and
        # left spurious ``/home`` prefixes in place.  Accept either slash style
        # so mixed-separator paths coming from Windows config files are
        # recognised and trimmed correctly.
        drive_match = re.search(r"(?i)[A-Z]:(?:\\|/)", candidate)
        if drive_match:
            segment = candidate[drive_match.start() :]
            # Maintain the separator style used in ``candidate`` so that callers
            # running on Windows receive paths that respect the original
            # environment variable formatting (for example preserving doubled
            # backslashes) while POSIX hosts executing unit tests continue to
            # observe forward slash separators when provided.  ``PureWindowsPath``
            # normalises separators which collapsed intentional escaping in the
            # Windows-specific tests.  Convert mixed separators only when both
            # variants appear so the result remains deterministic.
            if "\\" in segment and "/" in segment:
                backslash = segment.count("\\")
                forward = segment.count("/")
                if backslash >= forward:
                    segment = segment.replace("/", "\\")
                else:
                    segment = segment.replace("\\", "/")
            return segment

        # UNC paths begin with two backslashes.  ``expanduser`` on POSIX hosts
        # converts ``~\\\\server\\share`` into ``/home/user/\\\\server\\share``;
        # trim the home prefix in that scenario.  Handle both backslash and
        # forward slash separators to support callers that pass ``//server``
        # style UNC paths in configuration files.
        orig_stripped = original.lstrip()
        is_unc_source = bool(re.match(r"^~?[/\\]{2}", orig_stripped))
        unc_match = re.search(r"(?:\\\\|//)[^\\/]+[\\/][^\\/]+.*", candidate)
        if is_unc_source and unc_match:
            segment = unc_match.group(0)
            segment = segment.lstrip("/\\")
            if segment:
                return "\\\\" + segment.replace("/", "\\")

        if is_unc_source and "\\" in candidate:
            first_backslash = candidate.find("\\")
            if first_backslash != -1:
                start = candidate.rfind("/", 0, first_backslash)
                if start != -1:
                    segment = candidate[start:]
                    segment = segment.lstrip("/\\")
                    if segment:
                        return "\\\\" + segment.replace("/", "\\")

        if is_unc_source and "//" in candidate:
            first_forward = candidate.find("//")
            if first_forward != -1:
                segment = candidate[first_forward:]
                segment = segment.lstrip("/\\")
                if segment:
                    return "\\\\" + segment.replace("/", "\\")

        return candidate

    if "%" in expanded:
        # ``os.path.expandvars`` ignores ``%VAR%`` placeholders on non-Windows
        # hosts and is case-sensitive on Windows.  Retry unresolved tokens with
        # a case-insensitive lookup while respecting escaped literals (``%%``).
        # Environment variables on Windows frequently reference other
        # variables (``%OUTER%`` -> ``%INNER%\bin``).  Resolve recursively
        # until the string stabilises so nested references expand just like
        # they would in ``cmd.exe``.
        env_lower = {key.lower(): value for key, value in os.environ.items()}

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            lowered = name.lower()
            if any(ch in name for ch in "\\/%"):
                return match.group(0)
            direct = os.environ.get(name)
            if direct is not None:
                return direct
            return env_lower.get(lowered, match.group(0))

        previous = None
        current = expanded
        for _ in range(10):
            if "%" not in current or current == previous:
                break
            previous = current
            current = re.sub(token_pattern, _replace, current)
        expanded = current

    for sentinel, literal in sentinel_map.items():
        expanded = expanded.replace(sentinel, literal)

    expanded = _extract_windows_path(expanded, original=original_value)

    return Path(expanded)


def _configured_sandbox_data_dir() -> Path | None:
    """Return the sandbox data directory configured in settings when available."""

    log = logging.getLogger(__name__)

    try:
        configured = settings.sandbox_data_dir  # type: ignore[name-defined]
    except NameError:
        try:
            configured = SandboxSettings().sandbox_data_dir
        except Exception:  # pragma: no cover - transient settings bootstrap issues
            log.debug(
                "sandbox settings unavailable during early bootstrap; "
                "deferring to runtime defaults",
                exc_info=True,
            )
            return None
    except Exception:  # pragma: no cover - defensive guard for bad settings modules
        log.debug(
            "failed to read sandbox_data_dir from sandbox settings", exc_info=True
        )
        return None

    if not configured:
        return None

    try:
        return _expand_path(configured)
    except Exception:  # pragma: no cover - unexpected expansion failure
        log.debug(
            "unable to expand configured sandbox data directory %r", configured,
            exc_info=True,
        )
        return None


def _prepare_sandbox_data_dir_environment(argv: List[str] | None = None) -> None:
    """Populate sandbox data directory environment variables early."""

    _console("preparing sandbox data directory environment variables")

    if argv is None:
        argv = sys.argv[1:]
        _console("no argv provided; defaulting to sys.argv slice")

    override_raw = _extract_flag_value(argv, "--sandbox-data-dir")
    override_path = _expand_path(override_raw) if override_raw else None
    if override_path is not None:
        _console(f"sandbox data dir override detected: {override_path}")

    log = logging.getLogger(__name__)
    configured_dir = _configured_sandbox_data_dir()

    base_path: Path | None = None
    base_source = "environment"

    if override_path is not None:
        base_path = override_path
        base_source = "override"
    else:
        data_dir_value = os.environ.get("SANDBOX_DATA_DIR")
        env_path: Path | None = None
        if data_dir_value:
            try:
                env_path = _expand_path(data_dir_value)
            except Exception:
                log.debug(
                    "invalid SANDBOX_DATA_DIR value %r; ignoring", data_dir_value,
                    exc_info=True,
                )
        if env_path is not None:
            base_path = env_path
        elif configured_dir is not None:
            base_path = configured_dir
            base_source = "settings"
        else:
            base_path = Path.cwd() / "sandbox_data"
            base_source = "default"

    assert base_path is not None

    os.environ["SANDBOX_DATA_DIR"] = str(base_path)

    overlays_path = base_path / "active_overlays.json"
    failed_path = base_path / "failed_overlays.json"

    if base_source == "override":
        os.environ["SANDBOX_ACTIVE_OVERLAYS"] = str(overlays_path)
        os.environ["SANDBOX_FAILED_OVERLAYS"] = str(failed_path)
        _console("applied explicit overlay paths from override")
    else:
        os.environ.setdefault("SANDBOX_ACTIVE_OVERLAYS", str(overlays_path))
        os.environ.setdefault("SANDBOX_FAILED_OVERLAYS", str(failed_path))
        _console("ensured default overlay paths are configured")

    try:
        base_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        log.debug("unable to ensure sandbox data directory exists", exc_info=True)

    if base_source == "default":
        _console(
            "no sandbox data dir specified; initialised default at "
            f"{base_path}"
        )
    elif base_source == "settings":
        _console(f"sandbox data dir resolved from settings: {base_path}")
    else:
        _console(f"sandbox data dir resolved to {base_path}")

    _sync_sandbox_environment(base_path)
    _console("sandbox environment paths synchronised")


def _should_defer_bootstrap(argv: List[str] | None = None) -> bool:
    """Return ``True`` when the current invocation should skip heavy bootstrap."""

    if argv is None:
        argv = sys.argv[1:]
    for token in argv:
        if token in {"-h", "--help", "--version", "--check-settings"}:
            return True
        if token == "--smoke-test":
            return True
    runs_val = _extract_flag_value(argv, "--runs")
    if runs_val is not None:
        try:
            if int(runs_val) <= 0:
                return True
        except ValueError:
            pass
    max_iters_val = _extract_flag_value(argv, "--max-iterations")
    if max_iters_val is not None:
        try:
            if int(max_iters_val) <= 0:
                return True
        except ValueError:
            pass
    return False


_ORIGINAL_LOGGER_ERROR = logging.Logger.error
_ORIGINAL_LOGGER_WARNING = logging.Logger.warning


def _dependency_aware_logger_error(
    self: logging.Logger, message: object, *args, **kwargs
) -> None:  # pragma: no cover - logging glue
    """Treat dependency failures as warnings when we intend to relax them."""

    try:
        text = str(message)
    except Exception:
        text = ""

    lowered = text.lower()
    if (
        not _STRICT_DEPENDENCIES
        and (
            "missing system packages" in lowered
            or "missing python packages" in lowered
        )
    ):
        _ORIGINAL_LOGGER_WARNING(self, message, *args, **kwargs)
        return

    _ORIGINAL_LOGGER_ERROR(self, message, *args, **kwargs)


_INITIAL_ARGV = sys.argv[1:]
_STRICT_DEPENDENCIES = "--strict-dependencies" in _INITIAL_ARGV
_DEFER_BOOTSTRAP = _should_defer_bootstrap(_INITIAL_ARGV)
_LIGHTWEIGHT_IMPORT = os.getenv("RUN_AUTONOMOUS_LIGHTWEIGHT_IMPORT", "").lower() in {
    "1",
    "true",
    "yes",
}

# Importing ``run_autonomous`` inside a test or utility module should not trigger
# the expensive sandbox bootstrap.  Historically the file assumed it was always
# executed as ``__main__`` which caused helpers such as ``_expand_path`` to
# eagerly import the entire sandbox stack when accessed from ``python -c`` or
# unit tests.  Treat regular imports as lightweight invocations unless the
# caller explicitly opts into the full bootstrap via ``RUN_AUTONOMOUS_LIGHTWEIGHT_IMPORT``.
if __name__ != "__main__" and not _LIGHTWEIGHT_IMPORT:
    _LIGHTWEIGHT_IMPORT = True


if not _STRICT_DEPENDENCIES:
    logging.Logger.error = _dependency_aware_logger_error  # type: ignore[assignment]

if _STRICT_DEPENDENCIES:
    # Explicit strict mode should honour an existing environment toggle but
    # default to enforcing dependency checks.  ``pop`` avoids inadvertently
    # leaving a previous relaxed flag in place when the user opts-in to strict
    # enforcement.
    os.environ.pop("SANDBOX_SKIP_DEPENDENCY_CHECKS", None)
elif not os.environ.get("SANDBOX_SKIP_DEPENDENCY_CHECKS"):
    # Relax dependency checks by default so partially provisioned environments
    # (including Windows developer machines) can still execute the runner.
    os.environ["SANDBOX_SKIP_DEPENDENCY_CHECKS"] = "1"

if not (_DEFER_BOOTSTRAP or _LIGHTWEIGHT_IMPORT):
    _prepare_sandbox_data_dir_environment(_INITIAL_ARGV)
else:  # pragma: no cover - convenience path used during CLI discovery
    os.environ.setdefault("SANDBOX_SKIP_DEPENDENCY_CHECKS", "1")

_BOOTSTRAP_HELPERS: tuple[Callable[..., object], Callable[..., object]] | None = None
_BOOTSTRAP_SUPPORTS_ENFORCE: bool | None = None
_PREPARSED_ARGS: argparse.Namespace | None = None
_PREPARSED_ARGV: list[str] | None = None


def _get_bootstrap_helpers() -> tuple[Callable[..., object], Callable[..., object]]:
    """Lazily import heavy bootstrap helpers only when required."""

    global _BOOTSTRAP_HELPERS
    if _BOOTSTRAP_HELPERS is None:
        from sandbox_runner.bootstrap import (  # pragma: no cover - import side effect
            bootstrap_environment as _bootstrap_environment,
            _verify_required_dependencies as _verify_deps,
        )

        _BOOTSTRAP_HELPERS = (_bootstrap_environment, _verify_deps)
    return _BOOTSTRAP_HELPERS


def _call_bootstrap_environment(
    bootstrap_func: Callable[..., object],
    settings_obj: SandboxSettings,
    verifier: Callable[..., object],
    *,
    enforce_dependencies: bool,
) -> object:
    """Invoke ``bootstrap_func`` with ``enforce_dependencies`` when supported.

    Earlier sandbox builds exposed :func:`sandbox_runner.bootstrap.bootstrap_environment`
    without the ``enforce_dependencies`` keyword.  Test suites (and Windows
    developer machines with locally patched shims) still rely on that legacy
    signature.  Inspect the callable once and only include the keyword argument
    when the target advertises support or accepts arbitrary keyword arguments.
    """

    global _BOOTSTRAP_SUPPORTS_ENFORCE

    if _BOOTSTRAP_SUPPORTS_ENFORCE is None:
        try:
            signature = inspect.signature(bootstrap_func)
        except (TypeError, ValueError):  # pragma: no cover - builtin / C callables
            supports = True
        else:
            supports = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                or parameter.name == "enforce_dependencies"
                for parameter in signature.parameters.values()
            )
        _BOOTSTRAP_SUPPORTS_ENFORCE = supports

    if _BOOTSTRAP_SUPPORTS_ENFORCE:
        return bootstrap_func(
            settings_obj,
            verifier,
            enforce_dependencies=enforce_dependencies,
        )
    return bootstrap_func(settings_obj, verifier)


def _ensure_local_package() -> None:
    """Register the local ``menace_sandbox`` package for absolute imports."""

    if "menace_sandbox" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox", resolve_path("__init__.py")
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to load menace_sandbox package from local path")
    menace_pkg = importlib.util.module_from_spec(spec)
    sys.modules["menace_sandbox"] = menace_pkg
    spec.loader.exec_module(menace_pkg)
    sys.modules.setdefault("menace", menace_pkg)

logger = logging.getLogger(__name__)


def _log_extra(**fields: object) -> dict[str, object]:
    """Return structured logging extras when ``log_record`` is available."""

    if not callable(log_record):
        return {}

    try:
        record = log_record(**fields)  # type: ignore[call-arg]
    except Exception:
        logger.debug(
            "log_record helper rejected payload", extra={"fields": fields}, exc_info=True
        )
        return {}

    if isinstance(record, Mapping):
        return {"extra": record}

    if record is not None:
        logger.debug(
            "log_record returned non-mapping payload", extra={"payload": record}
        )

    return {}


def _normalise_actions(raw: object) -> list[str]:
    """Return a list of stringified actions from ``raw`` without raising."""

    if raw is None:
        return []

    if isinstance(raw, Mapping):
        return [f"{key}={value}" for key, value in raw.items()]

    if isinstance(raw, (str, bytes)):
        return [raw]

    if isinstance(raw, Iterable):
        try:
            return [str(item) for item in raw]
        except TypeError:
            pass

    return [str(raw)]


_DEPENDENCY_WARNINGS: list[str] = []
_DEPENDENCY_SUMMARY_EMITTED = False
_REPO_PATH_HINT: str | None = None


_WINDOWS_INCOMPATIBLE_PYTHON_DEPS = {
    "pyroute2",
}


def _format_dependency_warnings(line: str) -> list[str]:
    """Return ``line`` plus any platform specific remediation hints."""

    messages = [line]
    if os.name != "nt":
        return messages

    prefix = line.split(":", 1)[0].lower()
    if ":" in line:
        package_segment = line.split(":", 1)[1].split(".", 1)[0]
        packages = [pkg.strip() for pkg in package_segment.split(",") if pkg.strip()]
    else:  # pragma: no cover - defensive guard for unexpected formats
        packages = []

    if packages and prefix.startswith("missing system packages"):
        joined = " ".join(packages)
        messages.append(
            "Windows hint: install the system packages via Chocolatey "
            f"(e.g. 'choco install {joined}') or download the official installers."
        )
    elif packages and (
        prefix.startswith("missing python packages")
        or prefix.startswith("missing optional python packages")
    ):
        joined = " ".join(packages)
        optional_label = " optional" if "optional" in prefix else ""
        messages.append(
            f"Windows hint: install the{optional_label} Python packages "
            f"with 'python -m pip install {joined}'."
        )

    return messages


def _record_dependency_warning(line: str) -> tuple[str, list[str], bool]:
    """Store ``line`` and return the primary message, hints and freshness."""

    global _DEPENDENCY_SUMMARY_EMITTED
    messages = _format_dependency_warnings(line)
    is_new = False
    for message in messages:
        if message not in _DEPENDENCY_WARNINGS:
            _DEPENDENCY_WARNINGS.append(message)
            _DEPENDENCY_SUMMARY_EMITTED = False
            is_new = True
    primary = messages[0] if messages else line
    hints = messages[1:] if len(messages) > 1 else []
    return primary, hints, is_new


def _filter_dependency_errors(
    errors: Mapping[str, list[str]],
    *,
    platform: str | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Normalise ``errors`` for the current platform and return skip notes.

    ``sandbox_runner.bootstrap`` reports missing dependencies without any
    awareness of the host OS.  Some Python packages such as ``pyroute2`` are
    only available on POSIX platforms and cause Windows setups to emit
    misleading remediation hints on every run.  Filter those entries out so the
    dependency summary reflects what the user can actually install while still
    preserving the original ``errors`` mapping for other platforms.
    """

    platform = platform or os.name
    filtered: dict[str, list[str]] = {
        key: list(values) for key, values in errors.items()
    }
    skip_notes: list[str] = []

    if platform == "nt":
        removed: list[str] = []
        for bucket in ("python", "optional"):
            values = filtered.get(bucket)
            if not values:
                continue
            kept: list[str] = []
            for pkg in values:
                if pkg.lower() in _WINDOWS_INCOMPATIBLE_PYTHON_DEPS:
                    removed.append(pkg)
                    continue
                kept.append(pkg)
            filtered[bucket] = kept
        if removed:
            dedup: dict[str, str] = {}
            for pkg in removed:
                dedup.setdefault(pkg.lower(), pkg)
            skip_notes.append(
                "Skipping Windows-incompatible Python packages: "
                + ", ".join(dedup.values())
            )

    return filtered, skip_notes


def _emit_dependency_summary() -> None:
    """Emit a consolidated summary of dependency warnings collected so far."""

    global _DEPENDENCY_SUMMARY_EMITTED
    if not _DEPENDENCY_WARNINGS or _DEPENDENCY_SUMMARY_EMITTED:
        return

    _DEPENDENCY_SUMMARY_EMITTED = True
    _console("dependency requirements detected during startup:")
    for message in dict.fromkeys(_DEPENDENCY_WARNINGS):
        _console(f"  {message}")


atexit.register(_emit_dependency_summary)


def _initialise_settings() -> SandboxSettings:
    """Initialise :class:`SandboxSettings` with graceful dependency handling."""

    bootstrap_environment, _verify_required_dependencies = _get_bootstrap_helpers()
    base_settings = SandboxSettings()

    skip_strict = (
        not _STRICT_DEPENDENCIES
        and os.environ.get("SANDBOX_SKIP_DEPENDENCY_CHECKS") in {"1", "true", "True"}
    )

    if skip_strict:
        def _format_lines(errors: dict[str, list[str]]) -> list[str]:
            messages: list[str] = []
            if errors.get("system"):
                system_packages = list(dict.fromkeys(errors["system"]))
                messages.append(
                    "Missing system packages: "
                    + ", ".join(system_packages)
                    + "."
                )
                messages.extend(
                    format_system_package_instructions(system_packages)
                )
            if errors.get("python"):
                messages.append(
                    "Missing Python packages: "
                    + ", ".join(errors["python"])
                    + ". Install them with 'pip install <package>'."
                )
            if errors.get("optional"):
                messages.append(
                    "Missing optional Python packages: "
                    + ", ".join(errors["optional"])
                    + ". Install them with 'pip install <package>'."
                )
            return messages

        def _relaxed_verifier(settings_obj: SandboxSettings, *_: object) -> dict[str, list[str]]:
            errors = _verify_required_dependencies(settings_obj, strict=False)  # type: ignore[misc]
            errors, skip_notes = _filter_dependency_errors(errors)
            for note in skip_notes:
                logger.info("dependency check (relaxed) platform skip: %s", note)
                _console(f"dependency check (relaxed) platform skip: {note}")
            for line in _format_lines(errors):
                primary, hints, is_new = _record_dependency_warning(line)
                if not is_new:
                    continue
                logger.warning("dependency check (relaxed) flagged: %s", primary)
                _console(f"dependency check (relaxed) flagged: {primary}")
                for hint in hints:
                    logger.info("dependency remediation hint: %s", hint)
                    _console(f"dependency remediation hint: {hint}")
            return errors

        relaxed = _call_bootstrap_environment(
            bootstrap_environment,
            base_settings,
            _relaxed_verifier,
            enforce_dependencies=False,
        )
        if _DEPENDENCY_WARNINGS:
            _console("continuing with relaxed dependency checks")
        _emit_dependency_summary()
        return relaxed

    try:
        return _call_bootstrap_environment(
            bootstrap_environment,
            base_settings,
            _verify_required_dependencies,
            enforce_dependencies=True,
        )
    except SystemExit as exc:
        message = str(exc).strip()
        if message:
            for line in message.splitlines():
                line = line.strip()
                if not line:
                    continue
                primary, hints, is_new = _record_dependency_warning(line)
                if is_new:
                    logger.warning("dependency enforcement failed: %s", primary)
                    _console(f"dependency enforcement failed: {primary}")
                    for hint in hints:
                        logger.info("dependency remediation hint: %s", hint)
                        _console(f"dependency remediation hint: {hint}")
        else:
            logger.warning(
                "dependency enforcement failed; retrying without strict checks",
            )
            _console("dependency enforcement failed; retrying without strict checks")

        if _STRICT_DEPENDENCIES:
            _emit_dependency_summary()
            raise

        # Allow downstream imports to continue even when optional dependencies
        # are unavailable.  ``sandbox_runner`` exits eagerly during import when
        # required tools are missing which makes it difficult to execute the
        # runner in partially provisioned environments (common on Windows).
        # Opt-in to the relaxed behaviour for the remainder of the process.
        os.environ["SANDBOX_SKIP_DEPENDENCY_CHECKS"] = "1"

        relaxed_settings = _call_bootstrap_environment(
            bootstrap_environment,
            SandboxSettings(),
            _verify_required_dependencies,
            enforce_dependencies=False,
        )
        if message:
            _console("continuing with relaxed dependency checks")
        _emit_dependency_summary()
        return relaxed_settings


if _DEFER_BOOTSTRAP or _LIGHTWEIGHT_IMPORT:
    settings = SandboxSettings()
else:
    settings = _initialise_settings()
os.environ["SANDBOX_CENTRAL_LOGGING"] = "1" if settings.sandbox_central_logging else "0"
def _normalise_refresh_interval(value: float | int | None) -> float:
    """Return a sane refresh interval even when settings provide bad values."""

    try:
        interval = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        interval = float("nan")

    if not math.isfinite(interval) or interval <= 0.0:
        # The sandbox historically defaulted to a ten minute cadence.  Retain
        # that behaviour so that developer workstations (including Windows
        # environments where settings files are sometimes sparse) continue to
        # refresh on a predictable schedule.
        interval = 600.0

    return interval


LOCAL_KNOWLEDGE_REFRESH_INTERVAL = _normalise_refresh_interval(
    settings.local_knowledge_refresh_interval
)
_LKM_REFRESH_STOP = threading.Event()
_LKM_REFRESH_THREAD: threading.Thread | None = None


def _build_argument_parser(settings: SandboxSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full autonomous sandbox with environment presets",
    )
    parser.add_argument(
        "--preset-count",
        type=int,
        default=3,
        help="number of presets per iteration",
    )
    default_max_iterations = getattr(settings, "max_iterations", None)
    if default_max_iterations is None:
        default_max_iterations = 1
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=default_max_iterations,
        help="maximum iterations (defaults to 1 when unspecified)",
    )
    parser.add_argument("--sandbox-data-dir", help="override sandbox data directory")
    parser.add_argument(
        "--memory-db",
        dest="memory_db",
        help="path to GPT memory database (overrides GPT_MEMORY_DB)",
    )
    parser.add_argument(
        "--memory-compact-interval",
        type=float,
        help=(
            "seconds between GPT memory compaction cycles "
            "(overrides GPT_MEMORY_COMPACT_INTERVAL)"
        ),
    )
    parser.add_argument(
        "--memory-retention",
        help=(
            "comma separated tag=limit pairs controlling memory retention "
            "(overrides GPT_MEMORY_RETENTION)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="maximum number of full sandbox runs to execute",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="bootstrap dependencies and exit before launching the sandbox",
    )
    parser.add_argument(
        "--strict-dependencies",
        action="store_true",
        help="fail fast instead of relaxing dependency checks",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        help=(
            "start MetricsDashboard on this port for each run"
            " (overrides AUTO_DASHBOARD_PORT)"
        ),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="start Prometheus metrics server on this port",
    )
    parser.add_argument(
        "--roi-cycles",
        type=int,
        default=3,
        help="cycles below threshold before module convergence",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        help="override ROI delta threshold",
    )
    parser.add_argument(
        "--roi-confidence",
        type=float,
        help="confidence level for ROI convergence",
    )
    parser.add_argument(
        "--entropy-plateau-threshold",
        type=float,
        help="threshold for entropy delta plateau detection",
    )
    parser.add_argument(
        "--entropy-plateau-consecutive",
        type=int,
        help="entropy delta samples below threshold before module convergence",
    )
    parser.add_argument(
        "--synergy-cycles",
        type=int,
        help="cycles below threshold before synergy convergence",
    )
    parser.add_argument(
        "--synergy-threshold",
        type=float,
        help="override synergy threshold",
    )
    parser.add_argument(
        "--synergy-threshold-window",
        type=int,
        help="window size for adaptive synergy threshold",
    )
    parser.add_argument(
        "--synergy-threshold-weight",
        type=float,
        help="exponential weight for adaptive synergy threshold",
    )
    parser.add_argument(
        "--synergy-confidence",
        type=float,
        help="confidence level for synergy convergence",
    )
    parser.add_argument(
        "--synergy-ma-window",
        type=int,
        help="window size for synergy moving average",
    )
    parser.add_argument(
        "--synergy-stationarity-confidence",
        type=float,
        help="confidence level for synergy stationarity test",
    )
    parser.add_argument(
        "--synergy-std-threshold",
        type=float,
        help="standard deviation threshold for synergy convergence",
    )
    parser.add_argument(
        "--synergy-variance-confidence",
        type=float,
        help="confidence level for variance change test",
    )
    parser.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="compute convergence thresholds adaptively",
    )
    parser.add_argument(
        "--preset-file",
        action="append",
        dest="preset_files",
        help="JSON file defining environment presets; can be repeated",
    )
    parser.add_argument(
        "--no-preset-evolution",
        action="store_true",
        dest="disable_preset_evolution",
        help="disable adapting presets from previous run history",
    )
    parser.add_argument(
        "--preset-debug",
        action="store_true",
        help="enable verbose preset adaptation logs",
    )
    parser.add_argument(
        "--debug-log-file",
        help="write verbose logs to this file when --preset-debug is enabled",
    )
    parser.add_argument(
        "--forecast-log",
        help="write ROI forecast and threshold details to this file",
    )
    parser.add_argument(
        "--preset-log-file",
        help="write preset source and actions to this JSONL file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging (overrides --log-level)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.sandbox_log_level or settings.log_level,
        help="logging level for console output",
    )
    parser.add_argument(
        "--save-synergy-history",
        dest="save_synergy_history",
        action="store_true",
        default=None,
        help="persist synergy metrics across runs",
    )
    parser.add_argument(
        "--no-save-synergy-history",
        dest="save_synergy_history",
        action="store_false",
        default=None,
        help="do not persist synergy metrics",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="reload last ROI and synergy histories before running",
    )
    parser.add_argument(
        "--recursive-orphans",
        "--recursive-include",
        action="store_true",
        dest="recursive_orphans",
        default=None,
        help=(
            "recursively integrate orphan dependency chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=1; alias: --recursive-include)"
        ),
    )
    parser.add_argument(
        "--no-recursive-orphans",
        "--no-recursive-include",
        action="store_false",
        dest="recursive_orphans",
        help=(
            "disable recursive integration of orphan dependency chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=0)"
        ),
    )
    parser.add_argument(
        "--include-orphans",
        action="store_false",
        dest="include_orphans",
        default=None,
        help="disable running orphan modules during sandbox runs",
    )
    parser.add_argument(
        "--discover-orphans",
        action="store_false",
        dest="discover_orphans",
        default=None,
        help="disable automatic orphan discovery",
    )
    parser.add_argument(
        "--discover-isolated",
        action="store_true",
        dest="discover_isolated",
        default=None,
        help="automatically run discover_isolated_modules before the orphan scan",
    )
    parser.add_argument(
        "--no-discover-isolated",
        action="store_false",
        dest="discover_isolated",
        help="disable discover_isolated_modules during the orphan scan",
    )
    parser.add_argument(
        "--recursive-isolated",
        action="store_true",
        dest="recursive_isolated",
        default=None,
        help="recurse through dependencies of isolated modules (default)",
    )
    parser.add_argument(
        "--no-recursive-isolated",
        action="store_false",
        dest="recursive_isolated",
        help="disable recursively processing modules from discover_isolated_modules",
    )
    parser.add_argument(
        "--auto-include-isolated",
        action="store_true",
        help=(
            "automatically include isolated modules recursively (sets "
            "SANDBOX_AUTO_INCLUDE_ISOLATED=1 and SANDBOX_RECURSIVE_ISOLATED=1)"
        ),
    )
    parser.add_argument(
        "--foresight-trend",
        nargs=2,
        metavar=("FILE", "WORKFLOW_ID"),
        help="show ROI trend metrics from foresight history file",
    )
    parser.add_argument(
        "--foresight-stable",
        nargs=2,
        metavar=("FILE", "WORKFLOW_ID"),
        help="check workflow stability from foresight history file",
    )
    parser.add_argument(
        "--check-settings",
        action="store_true",
        help="validate environment settings and exit",
    )
    return parser


if __name__ == "__main__" and (_DEFER_BOOTSTRAP or _LIGHTWEIGHT_IMPORT):
    parser = _build_argument_parser(settings)
    try:
        parsed = parser.parse_args(sys.argv[1:])
    except SystemExit:
        # ``argparse`` has already emitted the relevant help or error output.
        # Re-raise so the interpreter terminates without importing heavyweight
        # modules.  This keeps ``python run_autonomous.py --help`` snappy while
        # still allowing other lightweight invocations (``--runs 0`` for
        # example) to flow through :func:`main` for their tailored output.
        raise
    else:
        _PREPARSED_ARGS = parsed
        _PREPARSED_ARGV = list(sys.argv[1:])


# Initialise database router with a unique menace_id. All DB access must go
# through the router.  Import modules requiring database access afterwards so
# they can rely on ``GLOBAL_ROUTER``.  When running in help/metadata mode we
# skip the connection setup entirely to avoid touching the filesystem.
if not (_DEFER_BOOTSTRAP or _LIGHTWEIGHT_IMPORT):
    MENACE_ID = uuid.uuid4().hex
    LOCAL_DB_PATH = settings.menace_local_db_path or str(
        resolve_path(f"menace_{MENACE_ID}_local.db")
    )
    SHARED_DB_PATH = settings.menace_shared_db_path or str(
        resolve_path("shared/global.db")
    )
    GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)
else:  # pragma: no cover - help/metadata path
    MENACE_ID = "bootstrap-preview"
    LOCAL_DB_PATH = ""
    SHARED_DB_PATH = ""
    GLOBAL_ROUTER = None

if not (_DEFER_BOOTSTRAP or _LIGHTWEIGHT_IMPORT):
    _ensure_local_package()
    from menace_sandbox.gpt_memory import GPTMemoryManager as _GPTMemoryManager
    from memory_maintenance import (
        MemoryMaintenance as _MemoryMaintenance,
        _load_retention_rules,
    )
    from gpt_knowledge_service import GPTKnowledgeService as _GPTKnowledgeService
    from local_knowledge_module import LocalKnowledgeModule as _LocalKnowledgeModule

    GPTMemoryManager = _GPTMemoryManager
    MemoryMaintenance = _MemoryMaintenance
    GPTKnowledgeService = _GPTKnowledgeService
else:  # pragma: no cover - help/metadata path
    if TYPE_CHECKING:  # pragma: no cover - typing aid
        from local_knowledge_module import LocalKnowledgeModule as _LocalKnowledgeModule
    else:
        _LocalKnowledgeModule = None  # type: ignore[assignment]
    GPTMemoryManager = None  # type: ignore[assignment]
    MemoryMaintenance = None  # type: ignore[assignment]
    _load_retention_rules = lambda *_args, **_kwargs: []  # type: ignore[assignment]
    GPTKnowledgeService = None  # type: ignore[assignment]

LocalKnowledgeModule = _LocalKnowledgeModule

from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)

# Default to test mode when using the bundled SQLite database.
if settings.menace_mode.lower() == "production" and settings.database_url.startswith(
    "sqlite"
):
    logger.warning(
        "MENACE_MODE=production with SQLite database; switching to test mode"
    )
    os.environ["MENACE_MODE"] = "test"
    settings = SandboxSettings()

# Ensure repository root on sys.path when running as a script
if "menace" not in sys.modules:
    sys.path.insert(0, str(get_project_root()))

# Repository root used by background services like the RelevancyRadarService.
# Default to ``SANDBOX_REPO_PATH`` when provided, otherwise fall back to the
# directory containing this file.  ``SANDBOX_REPO_PATH`` is required in most
# environments but this fallback keeps unit tests and ad-hoc scripts working.
REPO_ROOT = resolve_path(settings.sandbox_repo_path or ".")


if TYPE_CHECKING:  # pragma: no cover - static typing helpers
    import menace.environment_generator as environment_generator
    import sandbox_runner
    import sandbox_runner.cli as cli
    from logging_utils import (
        get_logger,
        setup_logging,
        log_record,
        set_correlation_id,
    )
    from menace.audit_trail import AuditTrail
    from menace.environment_generator import generate_presets
    from menace.roi_tracker import ROITracker
    from foresight_tracker import ForesightTracker
    from menace.synergy_exporter import SynergyExporter
    from menace.synergy_history_db import (
        migrate_json_to_db,
        insert_entry,
        connect_locked,
    )
    import menace.synergy_history_db as shd
    from metrics_exporter import (
        start_metrics_server,
        roi_threshold_gauge,
        synergy_threshold_gauge,
        roi_forecast_gauge,
        synergy_forecast_gauge,
        synergy_adaptation_actions_total,
    )
    import metrics_exporter
    from synergy_monitor import ExporterMonitor, AutoTrainerMonitor
    from sandbox_recovery_manager import SandboxRecoveryManager
    from sandbox_runner.cli import full_autonomous_run
    from threshold_logger import ThresholdLogger
    from forecast_logger import ForecastLogger
    from preset_logger import PresetLogger
    from relevancy_radar_service import RelevancyRadarService
else:
    environment_generator = sys.modules.get(  # type: ignore[assignment]
        "menace.environment_generator"
    )
    sandbox_runner = sys.modules.get("sandbox_runner")  # type: ignore[assignment]
    cli = (
        getattr(sandbox_runner, "cli", None)
        if sandbox_runner is not None
        else None
    )  # type: ignore[assignment]
    get_logger = logging.getLogger  # type: ignore[assignment]
    setup_logging = None  # type: ignore[assignment]
    log_record = None  # type: ignore[assignment]
    set_correlation_id = lambda *_a, **_k: None  # type: ignore[assignment]
    AuditTrail = None  # type: ignore[assignment]
    generate_presets = None  # type: ignore[assignment]
    ROITracker = None  # type: ignore[assignment]
    ForesightTracker = None  # type: ignore[assignment]
    SynergyExporter = None  # type: ignore[assignment]
    migrate_json_to_db = None  # type: ignore[assignment]
    insert_entry = None  # type: ignore[assignment]
    connect_locked = None  # type: ignore[assignment]
    shd = None  # type: ignore[assignment]
    start_metrics_server = None  # type: ignore[assignment]
    roi_threshold_gauge = None  # type: ignore[assignment]
    synergy_threshold_gauge = None  # type: ignore[assignment]
    roi_forecast_gauge = None  # type: ignore[assignment]
    synergy_forecast_gauge = None  # type: ignore[assignment]
    synergy_adaptation_actions_total = None  # type: ignore[assignment]
    metrics_exporter = None  # type: ignore[assignment]
    ExporterMonitor = None  # type: ignore[assignment]
    AutoTrainerMonitor = None  # type: ignore[assignment]
    SandboxRecoveryManager = None  # type: ignore[assignment]
    full_autonomous_run = None  # type: ignore[assignment]
    ThresholdLogger = None  # type: ignore[assignment]
    ForecastLogger = None  # type: ignore[assignment]
    PresetLogger = None  # type: ignore[assignment]
    RelevancyRadarService = None  # type: ignore[assignment]

_RUNTIME_IMPORTS_LOADED = False


def _ensure_runtime_imports() -> None:
    """Load heavyweight modules when running the sandbox."""

    global _RUNTIME_IMPORTS_LOADED
    if _RUNTIME_IMPORTS_LOADED:
        return

    if "menace_sandbox" not in sys.modules:
        # Lightweight invocations (``--runs 0`` or ``--smoke-test``) defer the
        # bootstrap that normally wires up the local package.  Ensure the shim
        # is registered before attempting the heavyweight imports so auxiliary
        # commands executed from Windows terminals resolve modules consistently
        # with the full bootstrap path.
        _ensure_local_package()

    global GPTMemoryManager, MemoryMaintenance, _load_retention_rules, GPTKnowledgeService
    global LocalKnowledgeModule
    if (
        GPTMemoryManager is None
        or MemoryMaintenance is None
        or GPTKnowledgeService is None
        or LocalKnowledgeModule is None
        or getattr(_load_retention_rules, "__module__", __name__) == __name__
    ):
        from menace_sandbox.gpt_memory import GPTMemoryManager as _GPTMemoryManager
        from memory_maintenance import (
            MemoryMaintenance as _MemoryMaintenance,
            _load_retention_rules as _load_retention_rules_impl,
        )
        from gpt_knowledge_service import (
            GPTKnowledgeService as _GPTKnowledgeService,
        )
        from local_knowledge_module import (
            LocalKnowledgeModule as _LocalKnowledgeModule,
        )

        GPTMemoryManager = _GPTMemoryManager  # type: ignore[assignment]
        MemoryMaintenance = _MemoryMaintenance  # type: ignore[assignment]
        _load_retention_rules = _load_retention_rules_impl  # type: ignore[assignment]
        GPTKnowledgeService = _GPTKnowledgeService  # type: ignore[assignment]
        LocalKnowledgeModule = _LocalKnowledgeModule  # type: ignore[assignment]

    global environment_generator, sandbox_runner, cli
    global get_logger, setup_logging, log_record, set_correlation_id
    global AuditTrail, generate_presets, ROITracker, ForesightTracker
    global SynergyExporter, migrate_json_to_db, insert_entry, connect_locked, shd
    global start_metrics_server, roi_threshold_gauge, synergy_threshold_gauge
    global roi_forecast_gauge, synergy_forecast_gauge, synergy_adaptation_actions_total
    global metrics_exporter, ExporterMonitor, AutoTrainerMonitor
    global SandboxRecoveryManager, full_autonomous_run
    global ThresholdLogger, ForecastLogger, PresetLogger, RelevancyRadarService
    global logger

    import menace.environment_generator as environment_generator_module
    import sandbox_runner as sandbox_runner_module
    import sandbox_runner.cli as cli_module
    from logging_utils import (
        get_logger as _get_logger,
        setup_logging as _setup_logging,
        log_record as _log_record,
        set_correlation_id as _set_correlation_id,
    )
    from menace.audit_trail import AuditTrail as _AuditTrail
    from menace.environment_generator import generate_presets as _generate_presets
    from menace.roi_tracker import ROITracker as _ROITracker
    from foresight_tracker import ForesightTracker as _ForesightTracker
    from menace.synergy_exporter import SynergyExporter as _SynergyExporter
    from menace.synergy_history_db import (
        migrate_json_to_db as _migrate_json_to_db,
        insert_entry as _insert_entry,
        connect_locked as _connect_locked,
    )
    import menace.synergy_history_db as _shd

    try:  # pragma: no cover - executed when run as a script
        from metrics_exporter import (
            start_metrics_server as _start_metrics_server,
            roi_threshold_gauge as _roi_threshold_gauge,
            synergy_threshold_gauge as _synergy_threshold_gauge,
            roi_forecast_gauge as _roi_forecast_gauge,
            synergy_forecast_gauge as _synergy_forecast_gauge,
            synergy_adaptation_actions_total as _synergy_adaptation_actions_total,
        )
        import metrics_exporter as _metrics_exporter
    except ImportError:  # pragma: no cover - executed when run as a module
        from .metrics_exporter import (
            start_metrics_server as _start_metrics_server,
            roi_threshold_gauge as _roi_threshold_gauge,
            synergy_threshold_gauge as _synergy_threshold_gauge,
            roi_forecast_gauge as _roi_forecast_gauge,
            synergy_forecast_gauge as _synergy_forecast_gauge,
            synergy_adaptation_actions_total as _synergy_adaptation_actions_total,
        )
        from . import metrics_exporter as _metrics_exporter

    try:  # pragma: no cover - exercised implicitly in integration tests
        synergy_monitor_module = importlib.import_module("synergy_monitor")
    except ModuleNotFoundError:  # pragma: no cover - executed when not installed
        from . import synergy_monitor as synergy_monitor_module

    try:  # pragma: no cover - exercised implicitly in integration tests
        from sandbox_recovery_manager import SandboxRecoveryManager as _SandboxRecoveryManager
    except ImportError:  # pragma: no cover - executed when not installed or run directly
        from .sandbox_recovery_manager import (
            SandboxRecoveryManager as _SandboxRecoveryManager,
        )

    from sandbox_runner.cli import full_autonomous_run as _full_autonomous_run
    from threshold_logger import ThresholdLogger as _ThresholdLogger
    from forecast_logger import ForecastLogger as _ForecastLogger
    from preset_logger import PresetLogger as _PresetLogger

    try:  # pragma: no cover - simple import shim
        from relevancy_radar_service import (
            RelevancyRadarService as _RelevancyRadarService,
        )
    except Exception:  # pragma: no cover - executed when run via package
        from .relevancy_radar_service import (
            RelevancyRadarService as _RelevancyRadarService,
        )

    environment_generator = environment_generator_module
    sandbox_runner = sandbox_runner_module
    cli = cli_module
    get_logger = _get_logger  # type: ignore[assignment]
    setup_logging = _setup_logging  # type: ignore[assignment]
    log_record = _log_record  # type: ignore[assignment]
    set_correlation_id = _set_correlation_id  # type: ignore[assignment]
    AuditTrail = _AuditTrail  # type: ignore[assignment]
    generate_presets = _generate_presets  # type: ignore[assignment]
    ROITracker = _ROITracker  # type: ignore[assignment]
    ForesightTracker = _ForesightTracker  # type: ignore[assignment]
    SynergyExporter = _SynergyExporter  # type: ignore[assignment]
    migrate_json_to_db = _migrate_json_to_db  # type: ignore[assignment]
    insert_entry = _insert_entry  # type: ignore[assignment]
    connect_locked = _connect_locked  # type: ignore[assignment]
    shd = _shd  # type: ignore[assignment]
    start_metrics_server = _start_metrics_server  # type: ignore[assignment]
    roi_threshold_gauge = _roi_threshold_gauge  # type: ignore[assignment]
    synergy_threshold_gauge = _synergy_threshold_gauge  # type: ignore[assignment]
    roi_forecast_gauge = _roi_forecast_gauge  # type: ignore[assignment]
    synergy_forecast_gauge = _synergy_forecast_gauge  # type: ignore[assignment]
    synergy_adaptation_actions_total = _synergy_adaptation_actions_total  # type: ignore[assignment]
    metrics_exporter = _metrics_exporter  # type: ignore[assignment]
    ExporterMonitor = synergy_monitor_module.ExporterMonitor  # type: ignore[assignment]
    AutoTrainerMonitor = synergy_monitor_module.AutoTrainerMonitor  # type: ignore[assignment]
    SandboxRecoveryManager = _SandboxRecoveryManager  # type: ignore[assignment]
    full_autonomous_run = _full_autonomous_run  # type: ignore[assignment]
    ThresholdLogger = _ThresholdLogger  # type: ignore[assignment]
    ForecastLogger = _ForecastLogger  # type: ignore[assignment]
    PresetLogger = _PresetLogger  # type: ignore[assignment]
    RelevancyRadarService = _RelevancyRadarService  # type: ignore[assignment]

    if not hasattr(sandbox_runner, "_sandbox_main"):
        spec = importlib.util.spec_from_file_location(
            "sandbox_runner", path_for_prompt("sandbox_runner.py")
        )
        sr_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sr_mod)
        sandbox_runner = sys.modules["sandbox_runner"] = sr_mod

    logger = get_logger(__name__)
    _RUNTIME_IMPORTS_LOADED = True


logger = logging.getLogger(__name__)

GPT_MEMORY_MANAGER: GPTMemoryManager | None = None
GPT_KNOWLEDGE_SERVICE: GPTKnowledgeService | None = None
LOCAL_KNOWLEDGE_MODULE: LocalKnowledgeModule | None = None


def _port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the TCP ``port`` is free on ``host``."""
    with contextlib.closing(socket.socket()) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _free_port() -> int:
    """Return an available TCP port."""
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _start_local_knowledge_refresh(cleanup_funcs: List[Callable[[], None]]) -> None:
    """Start background thread to periodically refresh local knowledge."""

    _console("initialising local knowledge refresh loop")

    def _loop() -> None:
        _console("local knowledge refresh thread entered loop")
        run = 0
        while not _LKM_REFRESH_STOP.wait(LOCAL_KNOWLEDGE_REFRESH_INTERVAL):
            run += 1
            if LOCAL_KNOWLEDGE_MODULE is not None:
                try:
                    LOCAL_KNOWLEDGE_MODULE.refresh()
                    LOCAL_KNOWLEDGE_MODULE.memory.conn.commit()
                    _console(f"local knowledge refresh cycle {run} completed")
                except Exception:
                    logger.exception(
                        "failed to refresh local knowledge module",
                        **_log_extra(run=run),
                    )
                    _console(f"local knowledge refresh cycle {run} failed; see logs")

    global _LKM_REFRESH_THREAD
    if _LKM_REFRESH_THREAD is None:
        _LKM_REFRESH_THREAD = threading.Thread(target=_loop, daemon=True)
        _LKM_REFRESH_THREAD.start()
        _console("local knowledge refresh thread started")

        def _stop() -> None:
            global _LKM_REFRESH_THREAD
            _console("stopping local knowledge refresh thread")
            _LKM_REFRESH_STOP.set()
            if _LKM_REFRESH_THREAD is not None:
                _LKM_REFRESH_THREAD.join(timeout=1.0)
                if _LKM_REFRESH_THREAD.is_alive():
                    logger.warning(
                        "local knowledge refresh thread did not exit within timeout",
                        **_log_extra(timeout=1.0),
                    )
                _LKM_REFRESH_THREAD = None
            _LKM_REFRESH_STOP.clear()
            _console("local knowledge refresh thread stopped")

        cleanup_funcs.append(_stop)


class PresetModel(BaseModel):
    """Schema for environment presets."""

    model_config = ConfigDict(extra="allow")

    CPU_LIMIT: str
    MEMORY_LIMIT: str
    DISK_LIMIT: str | None = None
    NETWORK_LATENCY_MS: float | None = None
    NETWORK_JITTER_MS: float | None = None
    MIN_BANDWIDTH: str | None = None
    MAX_BANDWIDTH: str | None = None
    BANDWIDTH_LIMIT: str | None = None
    PACKET_LOSS: float | None = None
    PACKET_DUPLICATION: float | None = None
    SECURITY_LEVEL: int | None = None
    THREAT_INTENSITY: int | None = None
    GPU_LIMIT: int | None = None
    OS_TYPE: str | None = None
    CONTAINER_IMAGE: str | None = None
    VM_SETTINGS: dict | None = None
    FAILURE_MODES: list[str] | str | None = None

    _EXTRA_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Z0-9_]+$")

    @field_validator("CPU_LIMIT", mode="before")
    @classmethod
    def _cpu_numeric(cls, v):
        try:
            float(v)
        except Exception as e:
            raise ValueError("CPU_LIMIT must be numeric") from e
        return str(v)

    @field_validator("MEMORY_LIMIT", mode="before")
    @classmethod
    def _mem_numeric(cls, v):
        val = str(v)
        digits = "".join(ch for ch in val if ch.isdigit() or ch == ".")
        if not digits:
            raise ValueError("MEMORY_LIMIT must contain a numeric value")
        try:
            float(digits)
        except Exception as e:
            raise ValueError("MEMORY_LIMIT must contain a numeric value") from e
        return val

    @field_validator(
        "NETWORK_LATENCY_MS",
        "NETWORK_JITTER_MS",
        "PACKET_LOSS",
        "PACKET_DUPLICATION",
        mode="before",
    )
    @classmethod
    def _float_fields(cls, v):
        if v is None:
            return v
        try:
            return float(v)
        except Exception as e:
            raise ValueError("value must be numeric") from e

    @field_validator(
        "SECURITY_LEVEL", "THREAT_INTENSITY", "GPU_LIMIT", mode="before"
    )
    @classmethod
    def _int_fields(cls, v):
        if v is None:
            return v
        try:
            return int(v)
        except Exception as e:
            raise ValueError("value must be an integer") from e

    @field_validator("FAILURE_MODES", mode="before")
    @classmethod
    def _fm_list(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        if isinstance(v, list):
            return [str(m) for m in v]
        raise ValueError("FAILURE_MODES must be a list or comma separated string")

    @model_validator(mode="before")
    @classmethod
    def _normalise_extra_fields(cls, data):
        """Coerce recognised preset metadata and validate extra keys."""

        if not isinstance(data, Mapping):
            return data

        normalised: dict[str, object] = dict(data)
        for key in list(normalised.keys()):
            if not isinstance(key, str):
                raise TypeError("preset keys must be strings")
            if key not in cls.model_fields and not cls._EXTRA_KEY_PATTERN.fullmatch(key):
                raise ValueError(f"Unsupported preset attribute '{key}'")

        def _coerce(value: object) -> object:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return value
                lowered = stripped.lower()
                if lowered in {"true", "false"}:
                    return lowered == "true"
                if lowered in {"null", "none"}:
                    return None
                try:
                    if any(ch in stripped for ch in ".eE"):
                        return float(stripped)
                    return int(stripped)
                except ValueError:
                    return value
            if isinstance(value, Mapping):
                return {k: _coerce(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_coerce(item) for item in value]
            return value

        for key, value in list(normalised.items()):
            if key not in cls.model_fields:
                normalised[key] = _coerce(value)

        return normalised


class SynergyEntry(RootModel[dict[str, float]]):
    """Schema for synergy history entries."""

    @model_validator(mode="before")
    @classmethod
    def _check_values(cls, v):
        if not isinstance(v, dict):
            raise ValueError("entry must be a dict")
        out: dict[str, float] = {}
        for k, val in v.items():
            try:
                out[str(k)] = float(val)
            except Exception as e:
                raise ValueError("synergy values must be numeric") from e
        return out


def validate_presets(presets: list[dict]) -> list[dict]:
    """Validate preset dictionaries using :class:`PresetModel`."""
    validated: list[dict] = []
    errors: list[dict] = []
    for idx, p in enumerate(presets):
        try:
            model = PresetModel.model_validate(p)
            validated.append(model.model_dump(exclude_none=True))
        except ValidationError as exc:
            for err in exc.errors():
                new_err = err.copy()
                new_err["loc"] = ("preset", idx) + tuple(err.get("loc", ()))
                errors.append(new_err)
    if errors:
        raise ValidationError.from_exception_data("PresetModel", errors)
    return validated


def validate_synergy_history(hist: list[dict]) -> list[dict[str, float]]:
    """Validate synergy history entries using :class:`SynergyEntry`."""
    validated: list[dict[str, float]] = []
    for idx, entry in enumerate(hist):
        try:
            validated.append(SynergyEntry.model_validate(entry).root)
        except ValidationError as exc:
            sys.exit(f"Invalid synergy history entry at index {idx}: {exc}")
    return validated


# Resolve the setup marker path via the dynamic router to avoid
# reliance on relative paths when the module is imported from different
# working directories.
_SETUP_MARKER = resolve_path(".autonomous_setup_complete")


def _normalise_repo_path(
    candidate: str | os.PathLike[str] | Path,
    *,
    source: str,
) -> str:
    """Return a canonical repository path for ``candidate``."""

    log = logging.getLogger(__name__)

    if isinstance(candidate, Path):
        resolved_candidate = candidate
    else:
        try:
            resolved_candidate = _expand_path(candidate)
        except Exception:
            log.debug(
                "failed to expand repo path %r from %s; falling back to Path()",
                candidate,
                source,
                exc_info=True,
            )
            resolved_candidate = Path(str(candidate)).expanduser()

    if not isinstance(resolved_candidate, Path):
        resolved_candidate = Path(os.fspath(resolved_candidate))

    if not resolved_candidate.is_absolute():
        try:
            resolved_candidate = (REPO_ROOT / resolved_candidate).resolve()
        except Exception:
            resolved_candidate = (REPO_ROOT / resolved_candidate).absolute()

    return (
        resolved_candidate.as_posix()
        if os.name != "nt"
        else str(resolved_candidate)
    )


def _ensure_repo_path_environment(*, apply_defaults: bool = True) -> None:
    """Normalise ``SANDBOX_REPO_PATH`` and capture hints when it is unset.

    When ``apply_defaults`` is ``False`` the function refrains from mutating
    :data:`os.environ` when the variable is absent.  This keeps the initial
    validation stage honest – unit tests and first-run setups expect
    ``check_env`` to raise when ``SANDBOX_REPO_PATH`` has not been provided yet.
    Downstream bootstrap calls invoke the helper again with defaults enabled so
    the relaxed behaviour remains available when the sandbox actually starts.
    """

    global _REPO_PATH_HINT

    log = logging.getLogger(__name__)
    _REPO_PATH_HINT = None

    current = os.environ.get("SANDBOX_REPO_PATH")
    if current and str(current).strip():
        repo_path = _normalise_repo_path(current, source="environment")
        os.environ["SANDBOX_REPO_PATH"] = repo_path
        _REPO_PATH_HINT = repo_path
        log.debug("normalised SANDBOX_REPO_PATH from environment to %s", repo_path)
        return

    candidate = getattr(settings, "sandbox_repo_path", None)
    resolved_source = "settings"
    repo_path: str | None = None
    if candidate:
        try:
            repo_path = _normalise_repo_path(candidate, source=resolved_source)
        except Exception:
            log.debug(
                "SANDBOX_REPO_PATH missing; unable to resolve default %r from settings",
                candidate,
                exc_info=True,
            )
            repo_path = None
    else:
        log.debug(
            "SANDBOX_REPO_PATH missing and no sandbox_repo_path configured in settings",
        )

    if repo_path is None:
        resolved_source = "computed default"
        fallback = getattr(settings, "sandbox_repo_path", None) or REPO_ROOT
        try:
            repo_path = _normalise_repo_path(fallback, source=resolved_source)
        except Exception:
            log.debug(
                "SANDBOX_REPO_PATH fallback normalisation failed for %r",
                fallback,
                exc_info=True,
            )
            try:
                repo_path = _normalise_repo_path(Path.cwd(), source="working directory")
            except Exception:
                repo_path = None

    if repo_path:
        _REPO_PATH_HINT = repo_path
        if apply_defaults:
            os.environ["SANDBOX_REPO_PATH"] = repo_path
            try:
                if not getattr(settings, "sandbox_repo_path", None):
                    settings.sandbox_repo_path = repo_path  # type: ignore[assignment]
            except Exception:
                log.debug(
                    "unable to update settings.sandbox_repo_path to %s",
                    repo_path,
                    exc_info=True,
                )
            log.info(
                "SANDBOX_REPO_PATH defaulted to %s (%s)",
                repo_path,
                resolved_source,
            )
        else:
            log.info(
                "SANDBOX_REPO_PATH candidate %s available from %s but not applied", 
                repo_path,
                resolved_source,
            )
    else:
        log.debug(
            "SANDBOX_REPO_PATH not set; environment validation will require explicit configuration",
        )



def _check_dependencies(settings: SandboxSettings) -> bool:
    """Return ``True`` and warn if the setup script has not been executed."""
    if not _SETUP_MARKER.exists():
        logger.warning(
            "Dependencies may be missing. Run 'python setup_dependencies.py' first"
        )
    return True


def check_env() -> None:
    """Exit if critical environment variables are unset."""
    required = (
        ("SANDBOX_REPO_PATH", settings.sandbox_repo_path),
    )
    missing: list[str] = []
    for env_name, value in required:
        raw = os.environ.get(env_name)
        if raw is None or not str(raw).strip():
            missing.append(env_name)
            continue
        if not value:
            missing.append(env_name)
    if missing:
        message = "Missing required environment variables: " + ", ".join(missing)
        hints: list[str] = []
        if "SANDBOX_REPO_PATH" in missing:
            hint_value = _REPO_PATH_HINT
            if hint_value is None:
                candidate = getattr(settings, "sandbox_repo_path", None)
                if candidate:
                    try:
                        hint_value = _normalise_repo_path(candidate, source="settings")
                    except Exception:
                        hint_value = None
            if hint_value:
                hints.append(
                    "Set SANDBOX_REPO_PATH to the repository root (for example "
                    f"{hint_value})."
                )
            if os.name == "nt":
                hints.append(
                    "Windows PowerShell example: setx SANDBOX_REPO_PATH \"C:\\path\\to\\repo\""
                )
            else:
                hints.append(
                    "Shell example: export SANDBOX_REPO_PATH=\"/path/to/repo\""
                )
        if hints:
            message += " " + " ".join(hints)
        raise SystemExit(message)


def _get_env_override(name: str, current, settings: SandboxSettings):
    """Return parsed environment variable when ``current`` is ``None``."""
    env_val = getattr(settings, name.lower())
    if current is not None or env_val is None:
        return current

    result = None
    try:
        if isinstance(current, int):
            result = int(env_val)
        elif isinstance(current, float):
            result = float(env_val)
    except Exception:
        result = None

    if result is None:
        for cast in (int, float):
            try:
                result = cast(env_val)
                break
            except Exception:
                continue

    extra_kwargs: dict[str, object] = {}
    if callable(log_record):
        try:
            extra_kwargs["extra"] = log_record(variable=name, value=result)
        except Exception:
            extra_kwargs = {}
    logger.debug(
        "environment variable %s overrides CLI value: %s",
        name,
        result,
        **extra_kwargs,
    )

    return result


def load_previous_synergy(
    data_dir: str | Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return synergy history and moving averages from ``synergy_history.db``."""

    data_dir = Path(resolve_path(data_dir))
    path = data_dir / "synergy_history.db"
    if not path.exists():
        return [], []
    history: list[dict[str, float]] = []
    try:
        logger.warning("🔥 SQLite audit still active despite AUDIT_FILE_MODE")
        with connect_locked(path) as conn:
            rows = conn.execute(
                "SELECT entry FROM synergy_history ORDER BY id"
            ).fetchall()
        for (text,) in rows:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("invalid synergy history entry ignored: %s", exc)
                continue
            if isinstance(data, dict):
                history.append({str(k): float(v) for k, v in data.items()})
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning(
            "failed to load synergy history %s: %s",
            path_for_prompt(path),
            exc,
        )
        history = []

    ma_history: list[dict[str, float]] = []
    for idx, entry in enumerate(history):
        ma_entry: dict[str, float] = {}
        for k in entry:
            vals = [h.get(k, 0.0) for h in history[: idx + 1]]
            ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
            ma_entry[k] = ema
        ma_history.append(ma_entry)

    return history, ma_history


def prepare_presets(
    run_idx: int,
    args: argparse.Namespace,
    settings: SandboxSettings,
    preset_log: PresetLogger | None = None,
) -> tuple[list[dict], str]:
    """Return presets for ``run_idx`` and their source."""

    preset_source = "static file"
    actions: list[str] = []
    if args.preset_files:
        pf = Path(args.preset_files[(run_idx - 1) % len(args.preset_files)])
        try:
            data = json.loads(pf.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("preset file %s is corrupted: %s", pf, exc)
            data = generate_presets(args.preset_count)
            pf.write_text(json.dumps(data))
        presets_raw = [data] if isinstance(data, dict) else list(data)
        presets = validate_presets(presets_raw)
        logger.info(
            "loaded presets from file",
            **_log_extra(run=run_idx, presets=presets, preset_file=str(pf)),
        )
    else:
        if getattr(args, "disable_preset_evolution", False):
            presets = validate_presets(generate_presets(args.preset_count))
        else:
            gen_func = getattr(
                environment_generator,
                "generate_presets_from_history",
                generate_presets,
            )
            if gen_func is generate_presets:
                presets = validate_presets(generate_presets(args.preset_count))
            else:
                data_dir = resolve_path(
                    args.sandbox_data_dir or settings.sandbox_data_dir
                )
                presets = validate_presets(gen_func(str(data_dir), args.preset_count))
                preset_source = "history adaptation"
                if getattr(
                    getattr(environment_generator, "adapt_presets", object),
                    "_rl_agent",
                    None,
                ):
                    preset_source = "RL agent"
        logger.info(
            "generated presets", **_log_extra(run=run_idx, presets=presets)
        )
        actions = _normalise_actions(
            getattr(environment_generator.adapt_presets, "last_actions", [])
        )
        for act in actions:
            try:
                synergy_adaptation_actions_total.labels(action=act).inc()
            except Exception:
                logger.exception(
                    "failed to update adaptation actions gauge",
                    **_log_extra(action=act),
                )
        logger.debug(
            "preset source=%s last_actions=%s",
            preset_source,
            actions,
            **_log_extra(
                run=run_idx,
                preset_source=preset_source,
                last_actions=actions,
            ),
        )
    os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
    prepare_presets.last_source = preset_source  # type: ignore[attr-defined]
    if preset_log is not None:
        try:
            preset_log.log(run_idx, preset_source, actions)
        except Exception:
            logger.exception(
                "failed to log preset details",
                **_log_extra(run=run_idx, preset_source=preset_source),
            )
    return presets, preset_source


def execute_iteration(
    args: argparse.Namespace,
    settings: SandboxSettings,
    presets: list[dict],
    synergy_history: list[dict[str, float]],
    synergy_ma_history: list[dict[str, float]],
) -> tuple[ROITracker | None, ForesightTracker | None]:
    """Run one autonomous iteration using ``presets`` and return trackers."""

    recovery = SandboxRecoveryManager(
        sandbox_runner._sandbox_main,
        settings=settings,
    )
    sandbox_runner._sandbox_main = recovery.run
    volatility_threshold = settings.sandbox_volatility_threshold
    foresight_tracker = ForesightTracker(
        max_cycles=10, volatility_threshold=volatility_threshold
    )
    setattr(args, "foresight_tracker", foresight_tracker)
    try:
        full_autonomous_run(
            args,
            synergy_history=synergy_history,
            synergy_ma_history=synergy_ma_history,
        )
    finally:
        sandbox_runner._sandbox_main = recovery.sandbox_main
        if hasattr(args, "foresight_tracker"):
            delattr(args, "foresight_tracker")

    data_dir = Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
    hist_file = data_dir / "roi_history.json"
    tracker = ROITracker()
    try:
        tracker.load_history(str(hist_file))
    except Exception:
        logger.exception("failed to load tracker history: %s", hist_file)
        return None, foresight_tracker
    return tracker, foresight_tracker


def update_metrics(
    tracker: ROITracker,
    args: argparse.Namespace,
    run_idx: int,
    module_history: dict[str, list[float]],
    entropy_history: dict[str, list[float]],
    flagged: set[str],
    synergy_history: list[dict[str, float]],
    synergy_ma_history: list[dict[str, float]],
    roi_ma_history: list[float],
    history_conn: sqlite3.Connection | None,
    roi_threshold: float | None,
    roi_confidence: float | None,
    entropy_threshold: float | None,
    entropy_consecutive: int | None,
    synergy_threshold_window: int,
    synergy_threshold_weight: float,
    synergy_confidence: float | None,
    threshold_log: ThresholdLogger,
    forecast_log: ForecastLogger | None = None,
) -> tuple[bool, float, float]:
    """Update histories and return convergence status and EMA."""

    for mod, vals in tracker.module_deltas.items():
        module_history.setdefault(mod, []).extend(vals)
    for mod, vals in tracker.module_entropy_deltas.items():
        entropy_history.setdefault(mod, []).extend(vals)

    syn_vals = {
        k: v[-1]
        for k, v in tracker.metrics_history.items()
        if k.startswith("synergy_") and v
    }
    synergy_ema = None
    if syn_vals:
        synergy_history.append(syn_vals)
        if args.save_synergy_history and history_conn is not None:
            try:
                insert_entry(history_conn, syn_vals)
            except Exception:
                logger.exception(
                    "failed to save synergy history", **_log_extra(run=run_idx)
                )
        ma_entry: dict[str, float] = {}
        for k in syn_vals:
            vals = [h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]]
            ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
            ma_entry[k] = ema
        synergy_ma_history.append(ma_entry)
        synergy_ema = ma_entry

    history = getattr(tracker, "roi_history", [])
    roi_ema = None
    if history:
        roi_ema, _ = cli._ema(history[-args.roi_cycles :])
        roi_ma_history.append(roi_ema)

    roi_pred = None
    ci_lo = None
    ci_hi = None
    try:
        pred, (lo, hi) = tracker.forecast()
        roi_pred = float(pred)
        ci_lo = float(lo)
        ci_hi = float(hi)
        roi_forecast_gauge.set(roi_pred)
        logger.debug(
            "roi forecast=%.3f CI=(%.3f, %.3f)",
            roi_pred,
            ci_lo,
            ci_hi,
            **_log_extra(
                run=run_idx,
                roi_prediction=roi_pred,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
            ),
        )
    except Exception:
        logger.exception("ROI forecast failed")
        metrics_exporter.roi_forecast_failures_total.inc()

    try:
        syn_pred = tracker.predict_synergy()
        synergy_forecast_gauge.set(float(syn_pred))
        logger.debug(
            "synergy forecast=%.3f",
            syn_pred,
            **_log_extra(run=run_idx, synergy_prediction=syn_pred),
        )
    except Exception:
        logger.exception("synergy forecast failed")
        metrics_exporter.synergy_forecast_failures_total.inc()

    if getattr(args, "auto_thresholds", False):
        roi_threshold = cli._adaptive_threshold(tracker.roi_history, args.roi_cycles)
        thr_method = "adaptive"
    elif roi_threshold is None:
        roi_threshold = tracker.diminishing()
        thr_method = "diminishing"
    else:
        thr_method = "fixed"
    roi_threshold_gauge.set(float(roi_threshold))
    logger.debug(
        "roi threshold=%.3f method=%s",
        roi_threshold,
        thr_method,
        **_log_extra(
            run=run_idx,
            roi_threshold=roi_threshold,
            method=thr_method,
        ),
    )
    new_flags, _ = cli._diminishing_modules(
        module_history,
        flagged,
        roi_threshold,
        consecutive=args.roi_cycles,
        confidence=roi_confidence,
        entropy_history=entropy_history,
        entropy_threshold=entropy_threshold,
        entropy_consecutive=entropy_consecutive,
    )
    flagged.update(new_flags)

    thr = args.synergy_threshold
    if getattr(args, "auto_thresholds", False) or thr is None:
        thr = None
    syn_thr_val = (
        thr
        if thr is not None
        else cli._adaptive_synergy_threshold(
            synergy_history, synergy_threshold_window, weight=synergy_threshold_weight
        )
    )
    synergy_threshold_gauge.set(float(syn_thr_val))
    logger.debug(
        "synergy threshold=%.3f fixed=%s",
        syn_thr_val,
        thr is not None,
        **_log_extra(
            run=run_idx,
            synergy_threshold=syn_thr_val,
            fixed=thr is not None,
        ),
    )
    converged, ema_val, conf = cli.adaptive_synergy_convergence(
        synergy_history,
        args.synergy_cycles,
        threshold=thr,
        threshold_window=synergy_threshold_window,
        weight=synergy_threshold_weight,
        confidence=synergy_confidence,
    )
    threshold_log.log(run_idx, roi_threshold, syn_thr_val, converged)
    logger.debug(
        "synergy convergence=%s max|ema|=%.3f conf=%.3f thr=%.3f",
        converged,
        ema_val,
        conf,
        syn_thr_val,
        **_log_extra(
            run=run_idx,
            converged=converged,
            ema_value=ema_val,
            confidence=conf,
            threshold=syn_thr_val,
        ),
    )
    logger.debug(
        "forecast %.3f CI=(%.3f, %.3f) roi_thr=%.3f(%s) syn_thr=%.3f",
        roi_pred if roi_pred is not None else float("nan"),
        ci_lo if ci_lo is not None else float("nan"),
        ci_hi if ci_hi is not None else float("nan"),
        roi_threshold,
        thr_method,
        syn_thr_val,
        **_log_extra(
            run=run_idx,
            roi_prediction=roi_pred,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            roi_threshold=roi_threshold,
            threshold_method=thr_method,
            synergy_threshold=syn_thr_val,
        ),
    )
    # log threshold calculation details for debugging
    roi_vals = tracker.roi_history[-args.roi_cycles :] if tracker.roi_history else []
    roi_ema_val, roi_std_val = cli._ema(roi_vals) if roi_vals else (0.0, 0.0)

    synergy_details: dict[str, dict[str, float]] = {}
    if synergy_history:
        metrics: dict[str, list[float]] = {}
        for entry in synergy_history[-args.synergy_cycles :]:
            for k, v in entry.items():
                if k.startswith("synergy_"):
                    metrics.setdefault(k, []).append(float(v))
        for k, vals in metrics.items():
            ema_m, std_m = cli._ema(vals)
            n = len(vals)
            if n < 2 or std_m == 0:
                conf_m = 1.0 if abs(ema_m) <= syn_thr_val else 0.0
            else:
                se = std_m / math.sqrt(n)
                t_stat = abs(ema_m) / se
                p = 2 * (1 - _student_t_cdf(t_stat, n - 1))
                conf_m = 1 - p
            synergy_details[k] = {
                "ema": ema_m,
                "std": std_m,
                "confidence": conf_m,
            }

    if synergy_details:
        logger.debug(
            "synergy metric stats: %s",
            synergy_details,
            **_log_extra(run=run_idx, synergy_metric_stats=synergy_details),
        )

    logger.debug(
        "metrics window sizes roi=%d synergy=%d sy_win=%d w=%.3f",
        args.roi_cycles,
        args.synergy_cycles,
        synergy_threshold_window,
        synergy_threshold_weight,
        **_log_extra(
            run=run_idx,
            roi_ema=roi_ema_val,
            roi_std=roi_std_val,
            synergy_metrics=synergy_details,
            roi_window=args.roi_cycles,
            synergy_window=args.synergy_cycles,
            synergy_threshold_window=synergy_threshold_window,
            synergy_threshold_weight=synergy_threshold_weight,
        ),
    )
    if forecast_log is not None:
        forecast_log.log(
            {
                "run": run_idx,
                "roi_forecast": roi_pred,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "roi_threshold": roi_threshold,
                "threshold_method": thr_method,
                "synergy_threshold": syn_thr_val,
                "synergy_converged": converged,
                "synergy_confidence": conf,
                "synergy_metrics": synergy_details,
                "roi_ema": roi_ema_val,
                "roi_std": roi_std_val,
            }
        )
    return converged, ema_val, roi_threshold


def main(argv: List[str] | None = None) -> None:
    """Entry point for the autonomous runner."""

    global settings, LOCAL_KNOWLEDGE_REFRESH_INTERVAL, _PREPARSED_ARGS, _PREPARSED_ARGV

    if argv is None:
        if _PREPARSED_ARGV is not None:
            argv_list = list(_PREPARSED_ARGV)
        else:
            argv_list = list(sys.argv[1:])
    else:
        argv_list = list(argv)

    _console("main() reached - preparing sandbox environment")
    _prepare_sandbox_data_dir_environment(argv_list)
    parser = _build_argument_parser(settings)

    if argv is None and _PREPARSED_ARGS is not None and _PREPARSED_ARGV is not None:
        args = _PREPARSED_ARGS
        _PREPARSED_ARGS = None
        _PREPARSED_ARGV = None
    else:
        args = parser.parse_args(argv_list)
    _console("arguments parsed; configuring logging")

    log_level = "DEBUG" if args.verbose else args.log_level

    configure_logging = (
        setup_logging if callable(setup_logging) else _basic_setup_logging
    )
    configure_logging(level=log_level)

    logger.info("validating environment variables")
    _console("validating environment variables")
    _ensure_repo_path_environment(apply_defaults=False)

    # ``SANDBOX_REPO_PATH`` is required by a number of services spawned from the
    # runner, however new installations (and the Windows developer experience in
    # particular) benefit from the script selecting a sensible default instead
    # of aborting immediately.  When the variable is absent after the strict
    # check we re-run the helper with defaults enabled so the detected repository
    # root is exported before environment validation.
    repo_env = os.environ.get("SANDBOX_REPO_PATH")
    if repo_env is None or not str(repo_env).strip():
        _console("SANDBOX_REPO_PATH missing; selecting default repository root")
        _ensure_repo_path_environment()

    check_env()

    if (
        args.runs == 0
        and not args.check_settings
        and not args.foresight_trend
        and not args.foresight_stable
        and not getattr(args, "recover", False)
    ):
        logger.info("no sandbox runs requested; exiting before bootstrap")
        _console("no sandbox runs requested; skipping autonomous bootstrap")
        return

    if args.smoke_test:
        logger.info("smoke test requested; exiting before sandbox launch")
        _console("smoke test requested; bootstrap sequence validated; exiting")
        return

    _ensure_runtime_imports()
    set_correlation_id(str(uuid.uuid4()))
    _console("correlation id initialised; runtime imports loaded")

    if configure_logging is not setup_logging and callable(setup_logging):
        setup_logging(level=log_level)

    class _SuppressAuditPersistenceFilter(logging.Filter):
        _TARGET = (
            "AUDIT_FILE_MODE enabled; "
            "skipping shared_db_audit SQLite persistence"
        )

        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            if record.name.startswith("audit") and record.getMessage() == self._TARGET:
                return False
            return True

    logging.getLogger().addFilter(_SuppressAuditPersistenceFilter())
    _console(
        "logging configured at %s level"
        % ("DEBUG" if args.verbose else args.log_level)
    )

    if args.foresight_trend:
        _console("foresight trend mode selected; executing and exiting")
        file, workflow_id = args.foresight_trend
        cli.foresight_trend(file, workflow_id)
        return
    if args.foresight_stable:
        _console("foresight stability mode selected; executing and exiting")
        file, workflow_id = args.foresight_stable
        cli.foresight_stability(file, workflow_id)
        return

    mem_db = args.memory_db or settings.gpt_memory_db
    _console(f"initialising local knowledge module with memory db {mem_db}")
    global LOCAL_KNOWLEDGE_MODULE, GPT_MEMORY_MANAGER, GPT_KNOWLEDGE_SERVICE
    LOCAL_KNOWLEDGE_MODULE = init_local_knowledge(mem_db)
    GPT_MEMORY_MANAGER = LOCAL_KNOWLEDGE_MODULE.memory
    GPT_KNOWLEDGE_SERVICE = LOCAL_KNOWLEDGE_MODULE.knowledge
    _console("local knowledge module ready")

    if args.preset_debug:
        _console("preset debug enabled; configuring debug file handler")
        os.environ["PRESET_DEBUG"] = "1"
        log_path = args.debug_log_file
        if not log_path:
            data_dir = Path(
                resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir)
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = data_dir / "preset_debug.log"
        else:
            log_path = resolve_path(log_path)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(fh)

    port = args.metrics_port
    if port is None:
        env_val = settings.metrics_port
        if env_val is not None:
            port = env_val
    if port is not None:
        _console(f"attempting to start metrics server on port {port}")
        try:
            logger.info("starting metrics server on port %d", port)
            start_metrics_server(int(port))
            logger.info("metrics server running on port %d", port)
            _console(f"metrics server running on port {port}")
        except Exception:
            logger.exception("failed to start metrics server")
            _console(f"failed to start metrics server on port {port}")

    logger.info("validating environment variables")
    _console("validating environment variables")
    _ensure_repo_path_environment()
    check_env()
    logger.info("environment validation complete")
    _console("environment validation complete")

    try:
        refreshed_settings = SandboxSettings()
    except ValidationError as exc:
        if args.check_settings:
            logger.warning("%s", exc)
            _console("settings validation failed during check-settings; exiting")
            return
        logger.warning("settings validation failed; continuing with existing configuration: %s", exc)
        _console("settings validation failed; continuing with existing configuration")
        refreshed_settings = settings
    else:
        settings = refreshed_settings
    settings = refreshed_settings

    env_file = resolve_path(refreshed_settings.menace_env_file)
    try:
        env_path = Path(env_file)
    except TypeError:
        env_path = Path(str(env_file))
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if not env_path.exists():
            env_path.touch()
            _console(f"ensured environment file exists at {env_path}")
    except Exception:
        logger.debug("unable to create environment file at %s", env_path, exc_info=True)

    cwd_env = Path.cwd() / ".env"
    if not cwd_env.exists():
        try:
            cwd_env.touch()
            _console(f"ensured environment file exists at {cwd_env}")
        except Exception:
            logger.debug(
                "unable to create working directory environment file at %s",
                cwd_env,
                exc_info=True,
            )

    previous_interval = LOCAL_KNOWLEDGE_REFRESH_INTERVAL
    new_interval = _normalise_refresh_interval(refreshed_settings.local_knowledge_refresh_interval)
    if not math.isclose(previous_interval, new_interval, rel_tol=0.0, abs_tol=1e-9):
        _console(
            "local knowledge refresh interval updated from %.3fs to %.3fs"
            % (previous_interval, new_interval)
        )
    LOCAL_KNOWLEDGE_REFRESH_INTERVAL = new_interval
    _console("runtime settings loaded")

    auto_include_isolated = bool(
        getattr(refreshed_settings, "auto_include_isolated", True)
        or getattr(args, "auto_include_isolated", False)
    )
    recursive_orphans = getattr(settings, "recursive_orphan_scan", True)
    if args.recursive_orphans is not None:
        recursive_orphans = args.recursive_orphans
    recursive_isolated = getattr(settings, "recursive_isolated", True)
    if args.recursive_isolated is not None:
        recursive_isolated = args.recursive_isolated
    # ``auto_include_isolated`` forces recursive isolated scans only when the
    # user didn't explicitly request otherwise via ``--recursive-isolated``.
    if auto_include_isolated and args.recursive_isolated is None:
        recursive_isolated = True

    args.auto_include_isolated = auto_include_isolated
    args.recursive_orphans = recursive_orphans
    args.recursive_isolated = recursive_isolated

    _console(
        "calculated sandbox toggles: auto_include_isolated=%s recursive_orphans=%s recursive_isolated=%s"
        % (auto_include_isolated, recursive_orphans, recursive_isolated)
    )
    os.environ["SANDBOX_AUTO_INCLUDE_ISOLATED"] = "1" if auto_include_isolated else "0"
    os.environ["SELF_TEST_AUTO_INCLUDE_ISOLATED"] = (
        "1" if auto_include_isolated else "0"
    )
    val = "1" if recursive_orphans else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    val_iso = "1" if recursive_isolated else "0"
    os.environ["SANDBOX_RECURSIVE_ISOLATED"] = val_iso
    os.environ["SELF_TEST_RECURSIVE_ISOLATED"] = val_iso
    os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1"
    os.environ["SELF_TEST_DISCOVER_ISOLATED"] = "1"
    include_orphans = True
    if getattr(args, "include_orphans") is False:
        include_orphans = False
    args.include_orphans = include_orphans
    os.environ["SANDBOX_INCLUDE_ORPHANS"] = "1" if include_orphans else "0"
    os.environ["SELF_TEST_INCLUDE_ORPHANS"] = "1" if include_orphans else "0"
    if not include_orphans:
        os.environ["SANDBOX_DISABLE_ORPHANS"] = "1"
    discover_orphans = True
    if getattr(args, "discover_orphans") is False:
        discover_orphans = False
    args.discover_orphans = discover_orphans
    os.environ["SANDBOX_DISABLE_ORPHAN_SCAN"] = "1" if not discover_orphans else "0"
    os.environ["SELF_TEST_DISCOVER_ORPHANS"] = "1" if discover_orphans else "0"
    if getattr(args, "discover_isolated") is not None:
        val_di = "1" if args.discover_isolated else "0"
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = val_di
        os.environ["SELF_TEST_DISCOVER_ISOLATED"] = val_di

    logger.info(
        "run_autonomous starting with data_dir=%s runs=%s metrics_port=%s",
        resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir),
        args.runs,
        port,
    )

    data_dir = Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
    legacy_json = data_dir / "synergy_history.json"
    db_file = data_dir / "synergy_history.db"
    if not db_file.exists() and legacy_json.exists():
        logger.info("migrating %s to SQLite", legacy_json)
        logger.warning("🔥 SQLite audit still active despite AUDIT_FILE_MODE")
        migrate_json_to_db(legacy_json, db_file)

    if args.check_settings:
        logger.info("Environment settings valid")
        return

    if settings.roi_cycles is not None:
        args.roi_cycles = settings.roi_cycles
    if settings.synergy_cycles is not None:
        args.synergy_cycles = settings.synergy_cycles
    if settings.save_synergy_history is not None:
        args.save_synergy_history = settings.save_synergy_history
    elif args.save_synergy_history is None:
        args.save_synergy_history = True

    synergy_history: list[dict[str, float]] = []
    synergy_ma_prev: list[dict[str, float]] = []
    exit_stack = contextlib.ExitStack()
    history_conn: sqlite3.Connection | None = None
    if args.save_synergy_history or args.recover:
        _console("loading previous synergy history from disk")
        synergy_history, synergy_ma_prev = load_previous_synergy(data_dir)
        logger.warning("🔥 SQLite audit still active despite AUDIT_FILE_MODE")
        history_conn = exit_stack.enter_context(
            shd.connect_locked(data_dir / "synergy_history.db")
        )
    if args.synergy_cycles is None:
        args.synergy_cycles = max(3, len(synergy_history))
    _console(
        "synergy history prepared; cycles=%s entries=%d"
        % (args.synergy_cycles, len(synergy_history))
    )

    if args.preset_files is None:
        _console("preparing preset files")
        data_dir = Path(
            resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir)
        )
        preset_file = data_dir / "presets.json"
        created_preset = False
        env_val = settings.sandbox_env_presets
        if env_val:
            try:
                presets_raw = json.loads(env_val)
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except ValidationError as exc:
                sys.exit(f"Invalid preset from SANDBOX_ENV_PRESETS: {exc}")
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
                os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
            logger.info(
                "generated presets from environment",
                **_log_extra(presets=presets, source="environment"),
            )
        elif preset_file.exists():
            try:
                presets_raw = json.loads(preset_file.read_text())
                if isinstance(presets_raw, dict):
                    presets_raw = [presets_raw]
                presets = validate_presets(presets_raw)
            except ValidationError as exc:
                sys.exit(f"Invalid preset file {preset_file}: {exc}")
            except json.JSONDecodeError as exc:
                logger.warning("preset file %s is corrupted: %s", preset_file, exc)
                presets = validate_presets(generate_presets(args.preset_count))
                preset_file.write_text(json.dumps(presets))
            except Exception:
                presets = validate_presets(generate_presets(args.preset_count))
            logger.info(
                "loaded presets from file",
                **_log_extra(presets=presets, source=str(preset_file)),
            )
        else:
            if getattr(args, "disable_preset_evolution", False):
                presets = validate_presets(generate_presets(args.preset_count))
            else:
                gen_func = getattr(
                    environment_generator,
                    "generate_presets_from_history",
                    generate_presets,
                )
                if gen_func is generate_presets:
                    presets = validate_presets(generate_presets(args.preset_count))
                else:
                    presets = validate_presets(
                        gen_func(str(data_dir), args.preset_count)
                    )
                    actions = _normalise_actions(
                        getattr(
                            environment_generator.adapt_presets, "last_actions", []
                        )
                    )
                    for act in actions:
                        try:
                            synergy_adaptation_actions_total.labels(action=act).inc()
                        except Exception:
                            logger.exception(
                                "failed to update adaptation actions gauge",
                                **_log_extra(action=act),
                            )
            logger.info(
                "generated presets", **_log_extra(presets=presets, source="auto")
            )
        os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(presets)
        if not preset_file.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            preset_file.write_text(json.dumps(presets))
            created_preset = True
        args.preset_files = [str(preset_file)]
        if created_preset:
            logger.info("created preset file at %s", preset_file)
        _console(
            "preset preparation complete using source %s"
            % ("environment" if env_val else str(preset_file))
        )

    logger.info("performing dependency check")
    _console("performing dependency check")
    _check_dependencies(settings)
    logger.info("dependency check complete")
    _console("dependency check complete")

    dash_port = args.dashboard_port
    dash_env = settings.auto_dashboard_port
    if dash_port is None and dash_env is not None:
        dash_port = dash_env

    synergy_dash_port = None
    if args.save_synergy_history and dash_env is not None:
        synergy_dash_port = dash_env + 1

    cleanup_funcs: list[Callable[[], None]] = []
    cleanup_funcs.append(exit_stack.close)
    _console("starting local knowledge refresh thread")
    _start_local_knowledge_refresh(cleanup_funcs)

    shutdown_requested = threading.Event()

    def _cleanup() -> None:
        _console("running registered cleanup handlers")
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)

    def _handle_shutdown_signal(sig: int | None, _frame: object) -> None:
        """Request cooperative shutdown when external signals are received."""

        first_request = not shutdown_requested.is_set()
        shutdown_requested.set()
        if sig is not None:
            if first_request:
                logger.info("received signal %s; requesting shutdown", sig)
                _console(f"shutdown signal {sig} received; interrupting main loop")
            else:
                logger.debug("additional shutdown signal %s ignored", sig)
        if not first_request:
            return
        try:
            _thread.interrupt_main()
        except RuntimeError:  # pragma: no cover - defensive when no main thread
            logger.debug("interrupt_main unavailable; relying on signal propagation")

    cleanup_signals = [
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGBREAK", None) if os.name == "nt" else None,
    ]
    for sig in cleanup_signals:
        if sig is None:
            continue
        try:
            signal.signal(sig, _handle_shutdown_signal)
        except (AttributeError, ValueError):  # pragma: no cover - platform specific
            logger.debug("signal %s unavailable; skipping handler", sig)

    mem_maint = None
    if GPT_MEMORY_MANAGER is not None:
        retention_rules = _load_retention_rules()
        if args.memory_retention:
            os.environ["GPT_MEMORY_RETENTION"] = args.memory_retention
            retention_rules = _load_retention_rules()
        interval = args.memory_compact_interval
        if interval is None and settings.gpt_memory_compact_interval is not None:
            interval = settings.gpt_memory_compact_interval
        mem_maint = MemoryMaintenance(
            GPT_MEMORY_MANAGER,
            interval=interval,
            retention=retention_rules,
            knowledge_service=GPT_KNOWLEDGE_SERVICE,
        )
        _console("starting memory maintenance thread")
        mem_maint.start()
        cleanup_funcs.append(mem_maint.stop)
    if GPT_KNOWLEDGE_SERVICE is not None:
        cleanup_funcs.append(getattr(GPT_KNOWLEDGE_SERVICE, "stop", lambda: None))

    meta_log_path = (
        Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "sandbox_meta.log"
    )
    exporter_log = AuditTrail(str(meta_log_path))
    threshold_log_path = (
        Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "threshold_log.jsonl"
    )
    threshold_log = ThresholdLogger(str(threshold_log_path))
    cleanup_funcs.append(threshold_log.close)
    preset_log_path = (
        Path(resolve_path(args.preset_log_file))
        if args.preset_log_file
        else Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
        / "preset_log.jsonl"
    )
    preset_log = PresetLogger(str(preset_log_path))
    cleanup_funcs.append(preset_log.close)
    forecast_log = None
    if args.forecast_log:
        _console(f"forecast logging enabled at {args.forecast_log}")
        forecast_log = ForecastLogger(str(Path(resolve_path(args.forecast_log))))
        cleanup_funcs.append(forecast_log.close)

    synergy_exporter: SynergyExporter | None = None
    exporter_monitor: ExporterMonitor | None = None
    trainer_monitor: AutoTrainerMonitor | None = None
    if settings.export_synergy_metrics:
        port = settings.synergy_metrics_port
        if not _port_available(port):
            logger.error("synergy exporter port %d in use", port)
            port = _free_port()
            logger.info("using port %d for synergy exporter", port)
        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_history.db"
        )
        synergy_exporter = SynergyExporter(
            history_file=str(history_file),
            port=port,
        )
        try:
            logger.info(
                "starting synergy exporter on port %d using history %s",
                port,
                history_file,
            )
            synergy_exporter.start()
            logger.info(
                "synergy exporter running on port %d serving %s",
                port,
                history_file,
            )
            _console(
                f"synergy exporter running on port {port} serving {history_file}"
            )
            exporter_log.record(
                {"timestamp": int(time.time()), "event": "exporter_started"}
            )
            exporter_monitor = ExporterMonitor(synergy_exporter, exporter_log)
            exporter_monitor.start()
            cleanup_funcs.append(exporter_monitor.stop)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy exporter: %s", exc)
            _console("failed to start synergy exporter; continuing without exporter")
            exporter_log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_start_failed",
                    "error": str(exc),
                }
            )

    auto_trainer = None
    if settings.auto_train_synergy:
        from menace.synergy_auto_trainer import SynergyAutoTrainer

        interval = settings.auto_train_interval
        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_history.db"
        )
        weights_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "synergy_weights.json"
        )
        auto_trainer = SynergyAutoTrainer(
            history_file=str(history_file),
            weights_file=str(weights_file),
            interval=interval,
        )
        try:
            logger.info(
                "starting synergy auto trainer with history %s weights %s interval %.1fs",
                history_file,
                weights_file,
                interval,
            )
            auto_trainer.start()
            logger.info(
                "synergy auto trainer running with history %s weights %s",
                history_file,
                weights_file,
            )
            _console("synergy auto trainer active")
            trainer_monitor = AutoTrainerMonitor(auto_trainer, exporter_log)
            trainer_monitor.start()
            cleanup_funcs.append(trainer_monitor.stop)
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("failed to start synergy auto trainer: %s", exc)
            _console("failed to start synergy auto trainer; continuing without trainer")

    dash_thread = None
    if dash_port:
        if not _port_available(dash_port):
            logger.error("metrics dashboard port %d in use", dash_port)
            dash_port = _free_port()
            logger.info("using port %d for MetricsDashboard", dash_port)
        _console(f"initialising MetricsDashboard on port {dash_port}")
        from threading import Thread

        from menace.metrics_dashboard import MetricsDashboard

        history_file = (
            Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
            / "roi_history.json"
        )
        _console(f"MetricsDashboard will use history file {history_file}")
        dash = MetricsDashboard(str(history_file))
        dash_thread = Thread(
            target=dash.run,
            kwargs={"port": dash_port},
            daemon=True,
        )
        logger.info("starting MetricsDashboard on port %d", dash_port)
        dash_thread.start()
        logger.info("MetricsDashboard running on port %d", dash_port)
        _console(f"MetricsDashboard active on port {dash_port}")
        cleanup_funcs.append(
            lambda: dash_thread and dash_thread.is_alive() and dash_thread.join(0.1)
        )

    s_dash = None
    if synergy_dash_port:
        if not _port_available(synergy_dash_port):
            logger.error("synergy dashboard port %d in use", synergy_dash_port)
            synergy_dash_port = _free_port()
            logger.info("using port %d for SynergyDashboard", synergy_dash_port)
        from threading import Thread

        try:
            from menace.self_improvement.engine import SynergyDashboard
        except RuntimeError as exc:
            logger.warning("SynergyDashboard unavailable: %s", exc)
            _console("SynergyDashboard unavailable; continuing without it")
        else:
            synergy_file = (
                Path(resolve_path(args.sandbox_data_dir or settings.sandbox_data_dir))
                / "synergy_history.db"
            )
            _console(
                f"initialising SynergyDashboard on port {synergy_dash_port} with {synergy_file}"
            )
            s_dash = SynergyDashboard(str(synergy_file))
            dash_t = Thread(
                target=s_dash.run,
                kwargs={"port": synergy_dash_port},
                daemon=True,
            )
            logger.info("starting SynergyDashboard on port %d", synergy_dash_port)
            dash_t.start()
            logger.info("SynergyDashboard running on port %d", synergy_dash_port)
            _console(f"SynergyDashboard active on port {synergy_dash_port}")
            cleanup_funcs.append(s_dash.stop)
            cleanup_funcs.append(lambda: dash_t.is_alive() and dash_t.join(0.1))

    relevancy_radar = None
    if (
        settings.enable_relevancy_radar
        and settings.relevancy_radar_interval is not None
    ):
        _console("starting relevancy radar service")
        relevancy_radar = RelevancyRadarService(
            REPO_ROOT, float(settings.relevancy_radar_interval)
        )
        relevancy_radar.start()
        atexit.register(relevancy_radar.stop)
        cleanup_funcs.append(relevancy_radar.stop)
        _console("relevancy radar service started")

    module_history: dict[str, list[float]] = {}
    entropy_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = list(synergy_ma_prev)
    roi_threshold = _get_env_override("ROI_THRESHOLD", args.roi_threshold, settings)
    synergy_threshold = _get_env_override(
        "SYNERGY_THRESHOLD", args.synergy_threshold, settings
    )
    roi_confidence = _get_env_override("ROI_CONFIDENCE", args.roi_confidence, settings)
    synergy_confidence = _get_env_override(
        "SYNERGY_CONFIDENCE", args.synergy_confidence, settings
    )
    if roi_confidence is None:
        roi_confidence = settings.roi.confidence or 0.95
    if synergy_confidence is None:
        synergy_confidence = settings.synergy.confidence or 0.95
    entropy_threshold = _get_env_override(
        "ENTROPY_PLATEAU_THRESHOLD", args.entropy_plateau_threshold, settings
    )
    entropy_consecutive = _get_env_override(
        "ENTROPY_PLATEAU_CONSECUTIVE", args.entropy_plateau_consecutive, settings
    )
    synergy_threshold_window = _get_env_override(
        "SYNERGY_THRESHOLD_WINDOW", args.synergy_threshold_window, settings
    )
    synergy_threshold_weight = _get_env_override(
        "SYNERGY_THRESHOLD_WEIGHT", args.synergy_threshold_weight, settings
    )
    synergy_ma_window = _get_env_override(
        "SYNERGY_MA_WINDOW", args.synergy_ma_window, settings
    )
    synergy_stationarity_confidence = _get_env_override(
        "SYNERGY_STATIONARITY_CONFIDENCE",
        args.synergy_stationarity_confidence,
        settings,
    )
    synergy_std_threshold = _get_env_override(
        "SYNERGY_STD_THRESHOLD", args.synergy_std_threshold, settings
    )
    synergy_variance_confidence = _get_env_override(
        "SYNERGY_VARIANCE_CONFIDENCE", args.synergy_variance_confidence, settings
    )
    if synergy_threshold_window is None:
        synergy_threshold_window = args.synergy_cycles
    if synergy_threshold_weight is None:
        synergy_threshold_weight = 1.0
    if synergy_ma_window is None:
        synergy_ma_window = args.synergy_cycles
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = (
            settings.synergy.stationarity_confidence or synergy_confidence or 0.95
        )
    if synergy_std_threshold is None:
        synergy_std_threshold = 1e-3
    if synergy_variance_confidence is None:
        synergy_variance_confidence = (
            settings.synergy.variance_confidence or synergy_confidence or 0.95
        )

    if args.recover:
        _console("recover flag detected; attempting to restore last tracker state")
        tracker = SandboxRecoveryManager.load_last_tracker(data_dir)
        if tracker:
            _console("previous tracker state found; merging histories")
            last_tracker = tracker
            for mod, vals in tracker.module_deltas.items():
                module_history.setdefault(mod, []).extend(vals)
            for mod, vals in tracker.module_entropy_deltas.items():
                entropy_history.setdefault(mod, []).extend(vals)
            missing = tracker.synergy_history[len(synergy_history) :]
            for entry in missing:
                synergy_history.append(entry)
                ma_entry: dict[str, float] = {}
                for k in entry:
                    vals = [
                        h.get(k, 0.0) for h in synergy_history[-args.synergy_cycles :]
                    ]
                    ema, _ = cli._ema(vals) if vals else (0.0, 0.0)
                    ma_entry[k] = ema
                synergy_ma_history.append(ma_entry)
            if missing:
                synergy_ma_prev = synergy_ma_history
            if missing and args.save_synergy_history and history_conn is not None:
                try:
                    for entry in missing:
                        insert_entry(history_conn, entry)
                except Exception:
                    logger.exception(
                        "failed to save synergy history",
                        **_log_extra(run=run_idx, recovered_entries=len(missing)),
                    )
                    _console("failed to persist recovered synergy history; see logs")
            if tracker.roi_history:
                ema, _ = cli._ema(tracker.roi_history[-args.roi_cycles :])
                roi_ma_history.append(ema)
    else:
        last_tracker = None
        _console("no recovery requested; starting fresh tracker state")

    run_idx = 0
    interrupted = False
    try:
        while (args.runs is None or run_idx < args.runs) and not shutdown_requested.is_set():
            run_idx += 1
            set_correlation_id(f"run-{run_idx}")
            logger.info(
                "Starting autonomous run %d/%s",
                run_idx,
                args.runs if args.runs is not None else "?",
            )
            _console(
                "starting autonomous run %d of %s"
                % (run_idx, args.runs if args.runs is not None else "∞")
            )
            presets, preset_source = prepare_presets(run_idx, args, settings, preset_log)
            logger.info(
                "using presets from %s",
                preset_source,
                **_log_extra(run=run_idx, preset_source=preset_source),
            )
            _console(f"presets ready for run {run_idx} from {preset_source}")
            logger.debug(
                "loaded presets from %s: %s",
                preset_source,
                presets,
                **_log_extra(
                    run=run_idx, preset_source=preset_source, presets=presets
                ),
            )

            tracker, foresight_tracker = execute_iteration(
                args,
                settings,
                presets,
                synergy_history,
                synergy_ma_history,
            )
            _console(f"iteration execution returned tracker={tracker is not None}")
            _console(f"run {run_idx} iteration complete")
            if tracker is None:
                logger.info("completed autonomous run %d", run_idx)
                set_correlation_id(None)
                _console(f"run {run_idx} completed with no tracker state")
                continue
            last_tracker = tracker

            if args.save_synergy_history and history_conn is not None:
                try:
                    rows = history_conn.execute(
                        "SELECT entry FROM synergy_history ORDER BY id"
                    ).fetchall()
                    _console("reloaded persisted synergy history from database")
                    synergy_history = [
                        {str(k): float(v) for k, v in json.loads(text).items()}
                        for (text,) in rows
                        if isinstance(json.loads(text), dict)
                    ]
                except Exception as exc:  # pragma: no cover - unexpected errors
                    logger.warning(
                        "failed to load synergy history %s: %s",
                        data_dir / "synergy_history.db",
                        exc,
                    )
                    synergy_history = []
                    _console("failed to reload synergy history; continuing with empty list")

            _console("updating metrics with latest tracker data")
            converged, ema_val, roi_threshold = update_metrics(
                tracker,
                args,
                run_idx,
                module_history,
                entropy_history,
                flagged,
                synergy_history,
                synergy_ma_history,
                roi_ma_history,
                history_conn,
                roi_threshold,
                roi_confidence,
                entropy_threshold,
                entropy_consecutive,
                synergy_threshold_window,
                synergy_threshold_weight,
                synergy_confidence,
                threshold_log,
                forecast_log,
            )
            _console(
                "metrics updated for run %d (ema=%.3f threshold=%.3f converged=%s)"
                % (run_idx, ema_val, roi_threshold, converged)
            )

            if foresight_tracker is not None:
                try:
                    slope, second_derivative, avg_stability = (
                        foresight_tracker.get_trend_curve("_global")
                    )
                    logger.info(
                        "foresight trend: slope=%.3f curve=%.3f avg_stability=%.3f",
                        slope,
                        second_derivative,
                        avg_stability,
                        **_log_extra(
                            run=run_idx,
                            foresight_slope=slope,
                            foresight_curve=second_derivative,
                            foresight_avg_stability=avg_stability,
                        ),
                    )
                except Exception:
                    logger.exception("failed to compute foresight trend")

            logger.info(
                "run %d summary: roi_threshold=%.3f ema=%.3f converged=%s flagged_modules=%d",
                run_idx,
                roi_threshold,
                ema_val,
                converged,
                len(flagged),
                **_log_extra(
                    run=run_idx,
                    roi_threshold=roi_threshold,
                    ema_value=ema_val,
                    converged=converged,
                    flagged_count=len(flagged),
                ),
            )

            logger.info("completed autonomous run %d", run_idx)
            set_correlation_id(None)
            _console(f"run {run_idx} fully complete")

            all_mods = set(module_history) | set(entropy_history)
            if all_mods and all_mods <= flagged and converged:
                logger.info(
                    "convergence reached",
                    **_log_extra(
                        run=run_idx,
                        ema_value=ema_val,
                        flagged_modules=sorted(flagged),
                    ),
                )
                _console(
                    "all modules converged; stopping after run %d" % run_idx
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        shutdown_requested.set()
        logger.info("keyboard interrupt received; aborting remaining runs")
        _console("keyboard interrupt received; beginning shutdown sequence")
    finally:
        shutdown_requested.set()
        try:
            atexit.unregister(_cleanup)
        except Exception:
            pass
        _cleanup()
        cleanup_funcs.clear()
        if interrupted:
            _console("cleanup complete after keyboard interrupt; finalising run_autonomous")
        else:
            _console("cleanup complete; finalising run_autonomous")

        if LOCAL_KNOWLEDGE_MODULE is not None:
            try:
                LOCAL_KNOWLEDGE_MODULE.refresh()
                LOCAL_KNOWLEDGE_MODULE.memory.conn.commit()
            except Exception:
                logger.exception("failed to refresh local knowledge module")
            else:
                _console("local knowledge module refreshed")

        if exporter_monitor is not None:
            try:
                logger.info(
                    "synergy exporter stopped after %d restarts",
                    exporter_monitor.restart_count,
                )
                exporter_log.record(
                    {
                        "timestamp": int(time.time()),
                        "event": "exporter_stopped",
                        "restart_count": exporter_monitor.restart_count,
                    }
                )
            except Exception:
                logger.exception("failed to stop synergy exporter")

        if trainer_monitor is not None:
            try:
                logger.info(
                    "synergy auto trainer stopped after %d restarts",
                    trainer_monitor.restart_count,
                )
                exporter_log.record(
                    {
                        "timestamp": int(time.time()),
                        "event": "auto_trainer_stopped",
                        "restart_count": trainer_monitor.restart_count,
                    }
                )
            except Exception:
                logger.exception("failed to stop synergy auto trainer")

        if GPT_MEMORY_MANAGER is not None:
            try:
                GPT_MEMORY_MANAGER.close()
            except Exception:
                logger.exception("failed to close GPT memory")

        if interrupted:
            logger.info("run_autonomous exiting after interrupt")
        else:
            logger.info("run_autonomous exiting")


def bootstrap(
    config_path: str | Path = get_project_root() / "config" / "bootstrap.yaml",
) -> None:
    """Bootstrap the autonomous sandbox using configuration from ``config_path``.

    The helper loads :class:`SandboxSettings`, initialises core databases and the
    optional event bus, starts the self improvement cycle in a background
    daemon thread and finally launches the sandbox runner.
    """
    from pydantic import ValidationError
    from sandbox_settings import load_sandbox_settings
    from self_improvement.api import init_self_improvement
    from self_improvement.orchestration import (
        start_self_improvement_cycle,
        stop_self_improvement_cycle,
    )
    from unified_event_bus import UnifiedEventBus
    from roi_results_db import ROIResultsDB
    from workflow_stability_db import WorkflowStabilityDB
    from sandbox_runner import launch_sandbox
    from self_learning_service import run_background as run_learning_background
    from self_test_service import SelfTestService
    import asyncio
    import threading

    try:
        bootstrap_settings = load_sandbox_settings(resolve_path(config_path))
    except ValidationError as exc:
        raise SystemExit(f"Invalid bootstrap configuration: {exc}") from exc

    bootstrap_environment, _verify_required_dependencies = _get_bootstrap_helpers()
    bootstrap_environment(bootstrap_settings, _verify_required_dependencies)

    global settings
    previous_settings = settings
    try:
        settings = bootstrap_settings
        _ensure_repo_path_environment()
        check_env()
    finally:
        settings = previous_settings

    os.environ.setdefault(
        "SANDBOX_DATA_DIR",
        os.fspath(resolve_path(bootstrap_settings.sandbox_data_dir)),
    )

    init_self_improvement(bootstrap_settings)

    try:
        ROIResultsDB()
        WorkflowStabilityDB()
    except Exception:
        logger.warning("database initialisation failed", exc_info=True)

    try:
        bus = UnifiedEventBus()
    except Exception:
        bus = None
        logger.warning("UnifiedEventBus unavailable")

    cleanup_funcs: list[Callable[[], None]] = []

    class _ExceptionThread(threading.Thread):
        """Thread subclass that stores exceptions from its target."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exception: BaseException | None = None

        def run(self) -> None:  # pragma: no cover - defensive
            try:
                super().run()
            except BaseException as exc:  # pragma: no cover - runtime safety
                self.exception = exc
                raise

    learn_start, learn_stop = run_learning_background(bootstrap_settings)

    _orig_thread = threading.Thread
    learn_thread: _ExceptionThread | None = None
    try:
        threading.Thread = _ExceptionThread  # type: ignore[assignment]
        learn_start()
        if learn_start.__closure__ is not None:
            for cell in learn_start.__closure__:
                obj = cell.cell_contents
                if isinstance(obj, _ExceptionThread):
                    learn_thread = obj
                    break
    finally:
        threading.Thread = _orig_thread

    monitor_stop = threading.Event()

    def _monitor_learning() -> None:
        if learn_thread is None:
            return
        while not monitor_stop.wait(5.0):
            if learn_thread.exception is not None:
                logger.critical(
                    "self-learning service failed", exc_info=learn_thread.exception
                )
                _thread.interrupt_main()
                return
            if not learn_thread.is_alive():
                logger.critical("self-learning service exited unexpectedly")
                _thread.interrupt_main()
                return

    monitor_thread = threading.Thread(target=_monitor_learning, daemon=True)
    monitor_thread.start()

    def _stop_monitor() -> None:
        monitor_stop.set()
        monitor_thread.join(timeout=1.0)

    cleanup_funcs.append(_stop_monitor)
    cleanup_funcs.append(learn_stop)

    from vector_service.context_builder import ContextBuilder
    from context_builder_util import ensure_fresh_weights

    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    try:
        ensure_fresh_weights(builder)
    except Exception:  # pragma: no cover - log and skip self-test when init fails
        logger.exception(
            "ContextBuilder initialisation failed; self-test loop disabled"
        )
    else:
        tester = SelfTestService(context_builder=builder)
        test_loop = asyncio.new_event_loop()

        def _tester_thread() -> None:
            asyncio.set_event_loop(test_loop)
            tester.run_continuous(loop=test_loop)
            test_loop.run_forever()

        t = threading.Thread(target=_tester_thread, daemon=True)
        t.start()

        def _stop_tests() -> None:
            test_loop.call_soon_threadsafe(lambda: asyncio.create_task(tester.stop()))
            test_loop.call_soon_threadsafe(test_loop.stop)
            t.join(timeout=1.0)

        cleanup_funcs.append(_stop_tests)

    def _noop():
        return None

    cycle_thread = start_self_improvement_cycle({"bootstrap": _noop}, event_bus=bus)
    cycle_thread.start()
    cleanup_funcs.append(stop_self_improvement_cycle)

    def _cleanup() -> None:
        if learn_thread is not None and getattr(learn_thread, "exception", None):
            logger.critical(
                "self-learning service failed", exc_info=learn_thread.exception
            )
        for func in cleanup_funcs:
            try:
                func()
            except Exception:
                logger.exception("cleanup failed")

    atexit.register(_cleanup)

    print(
        "[RUN_AUTONOMOUS] bootstrap complete, launching sandbox",
        flush=True,
    )
    logger.info(
        "autonomous bootstrap complete; handing off to sandbox runner",
        **_log_extra(stage="launch"),
    )

    try:
        launch_sandbox(settings=bootstrap_settings)
        logger.info(
            "autonomous sandbox run exited cleanly",
            **_log_extra(stage="shutdown"),
        )
    finally:
        _cleanup()
        logger.info(
            "autonomous cleanup complete",
            **_log_extra(stage="cleanup"),
        )


if __name__ == "__main__":
    main()
