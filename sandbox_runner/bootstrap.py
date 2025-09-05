from __future__ import annotations

import importlib
import importlib.util
import os
import sqlite3
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from logging_utils import get_logger, set_correlation_id, log_record

from packaging.version import Version
from dynamic_path_router import resolve_path, repo_root, path_for_prompt

from menace.auto_env_setup import ensure_env
from menace.default_config_manager import DefaultConfigManager
from sandbox_settings import SandboxSettings, load_sandbox_settings
try:  # pragma: no cover - allow flat import
    from metrics_exporter import (
        sandbox_restart_total,
        environment_failure_total,
        sandbox_crashes_total,
    )
except Exception:  # pragma: no cover - fallback for package execution
    from .metrics_exporter import (  # type: ignore
        sandbox_restart_total,
        environment_failure_total,
        sandbox_crashes_total,
    )

from .cli import main as _cli_main
from .cycle import ensure_vector_service


_SELF_IMPROVEMENT_THREAD: Any | None = None
_INITIALISED = False


logger = get_logger(__name__)


def _ensure_sqlite_db(path: Path) -> None:
    """Ensure an SQLite database exists at ``path``."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _start_optional_services(modules: Iterable[str]) -> None:
    """Best-effort launch of auxiliary service modules.

    For each entry in ``modules`` we attempt to import ``<module>_service`` and
    invoke a ``start`` or ``main`` function if present.  Failures are logged but
    otherwise ignored to keep bootstrap resilient.
    """

    for mod in modules:
        svc_name = f"{mod}_service"
        try:
            svc_mod = importlib.import_module(svc_name)
        except ModuleNotFoundError:
            logger.info("%s service not installed", svc_name)
            continue
        start_fn = getattr(svc_mod, "start", None) or getattr(svc_mod, "main", None)
        if callable(start_fn):
            try:
                start_fn()  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - best effort
                logger.warning("failed to launch %s", svc_name, exc_info=True)


def _verify_optional_modules(
    modules: Iterable[str], versions: dict[str, str]
) -> set[str]:
    """Attempt to import optional modules and warn when missing.

    Returns a set of modules that could not be imported.  The caller may use
    this information to skip further checks for those modules and avoid
    duplicate warnings.
    """

    missing: set[str] = set()
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            missing.add(mod)
            min_ver = versions.get(mod, "")
            ver_hint = f">={min_ver}" if min_ver else ""
            logger.warning(
                "%s module not found; install with 'pip install %s%s' to enable it",
                mod,
                mod,
                ver_hint,
            )
    return missing


def _self_improvement_warmup() -> None:
    """Perform basic warm-up steps prior to launching the optimisation loop.

    The warm-up applies default configuration values and attempts to reload
    sandbox settings.  Any failure is logged and re-raised to ensure that
    irrecoverable issues prevent the self-improvement cycle from starting."""

    try:
        DefaultConfigManager().apply_defaults()
        load_sandbox_settings()
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("self-improvement warm-up failed", exc_info=True)
        raise RuntimeError("self-improvement warm-up failed") from exc


def _default_env_value(name: str, settings: SandboxSettings) -> str:
    """Return a safe fallback value for ``name``.

    The helper favours innocuous defaults to keep the sandbox functional during
    initial setup while making it obvious that real credentials should be
    supplied for production use.
    """

    if name == "DATABASE_URL":
        data_dir = resolve_path(settings.sandbox_data_dir)
        return f"sqlite:///{data_dir / 'sandbox.db'}"
    if name == "MODELS":
        return "micro_models"
    if name.endswith("_KEY"):
        return f"{name.lower()}-placeholder"
    return ""


def auto_configure_env(settings: SandboxSettings) -> None:
    """Populate missing environment variables and ensure a model path exists.

    The configuration is persisted to ``settings.menace_env_file`` so repeated
    runs do not prompt again.  When ``MODELS`` points to a non-existent
    directory the helper falls back to the built-in ``micro_models`` demo
    bundle.
    """

    env_file = Path(getattr(settings, "menace_env_file", ".env"))
    if not env_file.is_absolute():
        env_file = repo_root() / env_file
    ensure_env(str(env_file))
    DefaultConfigManager(str(env_file)).apply_defaults()

    # load existing env file into a dictionary for easy updates
    existing: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    changed = False
    for name in settings.required_env_vars:
        if os.getenv(name):
            continue
        value = _default_env_value(name, settings)
        if not value and sys.stdin.isatty():  # pragma: no cover - interactive fallback
            try:
                entered = input(f"Enter value for {name} [{value}]: ").strip()
                if entered:
                    value = entered
            except EOFError:
                pass
        os.environ[name] = value
        if existing.get(name) != value:
            existing[name] = value
            changed = True

    models_spec = os.getenv("MODELS", "").strip()
    if models_spec in {"", "demo"}:
        model_path = resolve_path("micro_models")
    else:
        model_path = Path(models_spec)
        if not model_path.is_absolute():
            model_path = repo_root() / model_path
    if not model_path.exists():
        try:  # pragma: no cover - best effort download
            from vector_service.download_model import ensure_model as _ensure_model

            _ensure_model()
        except Exception:
            logger.warning("failed to ensure demo model", exc_info=True)
            model_path.mkdir(parents=True, exist_ok=True)
    os.environ["MODELS"] = str(model_path)
    if existing.get("MODELS") != str(model_path):
        existing["MODELS"] = str(model_path)
        changed = True

    if changed:
        env_file.write_text("\n".join(f"{k}={v}" for k, v in sorted(existing.items())))


def initialize_autonomous_sandbox(
    settings: SandboxSettings | None = None,
) -> SandboxSettings:
    cid = f"bootstrap-init-{uuid.uuid4()}"
    set_correlation_id(cid)
    sandbox_restart_total.labels(service="bootstrap", reason="init").inc()
    logger.info("initialize sandbox start", extra=log_record(event="start"))
    try:
        result = _initialize_autonomous_sandbox(settings)
        logger.info("initialize sandbox complete", extra=log_record(event="shutdown"))
        return result
    except Exception:
        environment_failure_total.labels(reason="init").inc()
        sandbox_crashes_total.inc()
        logger.exception(
            "initialize sandbox failure", extra=log_record(event="failure")
        )
        raise
    finally:
        set_correlation_id(None)


def _initialize_autonomous_sandbox(
    settings: SandboxSettings | None = None,
) -> SandboxSettings:
    """Prepare data directories, baseline metrics and optional services.

    The helper creates ``sandbox_data`` and baseline metric files when missing,
    initialises a couple of lightweight SQLite databases and verifies that the
    optional ``relevancy_radar`` and ``quick_fix_engine`` modules are available.
    A :class:`SandboxSettings` instance is returned for convenience.
    """

    settings = settings or load_sandbox_settings()
    global _INITIALISED, _SELF_IMPROVEMENT_THREAD
    if _INITIALISED:
        return settings

    try:
        auto_configure_env(settings)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("environment bootstrap failed", exc_info=True)
        raise RuntimeError("environment configuration incomplete") from exc

    # Ensure the mandatory vector_service dependency is available before
    # proceeding with further sandbox initialisation.
    ensure_vector_service()

    # Verify optional modules referenced in the docstring.  Missing modules are
    # collected so the version check below can skip them and avoid duplicate
    # warnings.
    try:
        missing_optional = _verify_optional_modules(
            ("relevancy_radar", "quick_fix_engine"),
            settings.optional_service_versions,
        )
    except Exception:  # pragma: no cover - best effort
        logger.warning("optional module verification failed", exc_info=True)
        missing_optional = set()

    data_dir = resolve_path(settings.sandbox_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe = data_dir / ".write-test"
        probe.touch()
        probe.unlink()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("sandbox data directory %s is not writable", data_dir, exc_info=True)
        raise RuntimeError(
            f"sandbox data directory '{data_dir}' is not writable; "
            "adjust permissions or update sandbox_data_dir"
        ) from exc

    # Ensure baseline metrics file exists; fall back to minimal snapshot when
    # metrics collection fails.
    baseline_str = getattr(settings, "alignment_baseline_metrics_path", "")
    baseline_path = Path()
    if baseline_str:
        try:
            baseline_path = resolve_path(str(baseline_str))
        except FileNotFoundError:
            baseline_path = Path(str(baseline_str))
            if not baseline_path.is_absolute():
                baseline_path = repo_root() / baseline_path
        if not baseline_path.exists():
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            try:  # compute baseline if possible
                from self_improvement.metrics import _update_alignment_baseline

                _update_alignment_baseline(settings)
            except Exception:  # pragma: no cover - best effort
                logger.warning("failed to populate baseline metrics", exc_info=True)
                baseline_path.write_text("{}\n", encoding="utf-8")

    # Create expected SQLite databases
    for name in settings.sandbox_required_db_files:
        _ensure_sqlite_db(data_dir / name)

    # Verify optional services are importable and meet version requirements
    from importlib import metadata

    for mod, min_version in settings.optional_service_versions.items():
        if mod in missing_optional:
            # Already warned about the missing module above.
            continue
        try:
            module = importlib.import_module(mod)
        except ModuleNotFoundError:
            logger.warning(
                "%s service not found; install with 'pip install %s>=%s' to enable it",
                mod,
                mod,
                min_version,
            )
            continue
        version: str | None
        try:
            version = metadata.version(mod)
        except Exception:
            version = getattr(module, "__version__", None)
        if version is None:
            logger.warning(
                "unable to determine %s version; reinstall with 'pip install --upgrade %s'",
                mod,
                mod,
            )
            continue
        if Version(version) < Version(min_version):
            logger.warning(
                "%s service version %s is too old; install %s>=%s with 'pip install --upgrade %s'",
                mod,
                version,
                mod,
                min_version,
                mod,
            )

    _start_optional_services(settings.optional_service_versions.keys())
    _INITIALISED = True

    try:
        from self_improvement.api import (
            init_self_improvement,
            start_self_improvement_cycle,
        )

        init_self_improvement(settings)
        thread = start_self_improvement_cycle({"bootstrap": _self_improvement_warmup})
        thread.start()
        try:
            thread.join(0)
        except Exception as exc:
            logger.error("self-improvement thread raised during startup", exc_info=True)
            raise RuntimeError("self-improvement thread failed to start") from exc
        inner = getattr(thread, "_thread", thread)
        if hasattr(inner, "is_alive") and not inner.is_alive():
            raise RuntimeError("self-improvement thread terminated unexpectedly")
        _SELF_IMPROVEMENT_THREAD = thread
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("self-improvement startup failed", exc_info=True)
        raise RuntimeError("self-improvement startup failed") from exc

    return settings


def shutdown_autonomous_sandbox(timeout: float | None = None) -> None:
    """Stop background self-improvement thread and reset globals."""

    cid = f"bootstrap-shutdown-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger.info("shutdown sandbox start", extra=log_record(event="start"))
    try:
        global _SELF_IMPROVEMENT_THREAD, _INITIALISED
        thread = _SELF_IMPROVEMENT_THREAD
        if thread is None:
            _INITIALISED = False
            logger.info(
                "shutdown sandbox complete", extra=log_record(event="shutdown")
            )
            return
        from self_improvement.api import stop_self_improvement_cycle

        stop_self_improvement_cycle()
        if hasattr(thread, "join"):
            thread.join(timeout)
        inner = getattr(thread, "_thread", thread)
        if hasattr(inner, "is_alive") and inner.is_alive():
            raise RuntimeError("self-improvement thread failed to shut down")
        try:
            from sandbox_runner import generative_stub_provider as _gsp

            _gsp.flush_caches()
            _gsp.cleanup_cache_files()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("stub cache cleanup failed", exc_info=True)
        try:
            from self_improvement import utils as _si_utils

            _si_utils.clear_import_cache()
            _si_utils.remove_import_cache_files()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("self-improvement cache cleanup failed", exc_info=True)
        _SELF_IMPROVEMENT_THREAD = None
        _INITIALISED = False
        logger.info(
            "shutdown sandbox complete", extra=log_record(event="shutdown")
        )
    except Exception:
        sandbox_crashes_total.inc()
        logger.exception(
            "shutdown sandbox failure", extra=log_record(event="failure")
        )
        raise
    finally:
        set_correlation_id(None)


def sandbox_health() -> dict[str, bool | dict[str, str]]:
    """Return basic health indicators for the sandbox environment."""

    thread = _SELF_IMPROVEMENT_THREAD
    inner = getattr(thread, "_thread", thread) if thread is not None else None
    alive = bool(getattr(inner, "is_alive", lambda: False)())

    settings = load_sandbox_settings()
    data_dir = resolve_path(
        os.getenv("SANDBOX_DATA_DIR", settings.sandbox_data_dir)
    )
    db_errors: dict[str, str] = {}
    for name in settings.sandbox_required_db_files:
        db_path = data_dir / name
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # noqa: SQL001
            conn.execute("PRAGMA schema_version")
            conn.close()
        except Exception as exc:
            logger.error(
                "failed to access database %s: %s", path_for_prompt(db_path), exc
            )
            db_errors[name] = str(exc)

    db_ok = not db_errors

    try:
        from sandbox_runner import generative_stub_provider as _gsp

        stub_init = _gsp._GENERATOR is not None  # type: ignore[attr-defined]
    except Exception:
        stub_init = False

    return {
        "self_improvement_thread_alive": alive,
        "databases_accessible": db_ok,
        "database_errors": db_errors,
        "stub_generator_initialized": stub_init,
    }


def _verify_required_dependencies(settings: SandboxSettings) -> dict[str, list[str]]:
    """Validate required dependencies and return missing categories.

    The function performs a pure validation pass without attempting to install
    any packages.  A dictionary mapping dependency categories to lists of missing
    packages is returned.  The categories are ``system``, ``python`` and
    ``optional``.  Callers may surface user-facing messages or decide how to
    handle the missing dependencies.
    """

    def _have_spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return name in sys.modules

    def _clean_list(name: str, items: list[str]) -> list[str]:
        valid: list[str] = []
        invalid: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                valid.append(item.strip())
            else:
                invalid.append(str(item))
        if invalid:
            logger.warning(
                "Ignoring unrecognised %s entries: %s", name, ", ".join(invalid)
            )
        return valid

    req_tools = _clean_list("system tool", settings.required_system_tools)
    req_pkgs = _clean_list("python package", settings.required_python_packages)
    opt_pkgs = _clean_list("optional python package", settings.optional_python_packages)

    missing_sys = [t for t in req_tools if shutil.which(t) is None]
    missing_req = [p for p in req_pkgs if not _have_spec(p)]
    missing_opt = [p for p in opt_pkgs if not _have_spec(p)]

    mode = settings.menace_mode.lower()

    errors: dict[str, list[str]] = {}
    if missing_sys:
        errors["system"] = missing_sys
    if missing_req:
        errors["python"] = missing_req
    if missing_opt and mode == "production":
        errors["optional"] = missing_opt
    elif missing_opt:
        logger.warning(
            "Missing optional Python packages: %s", ", ".join(missing_opt)
        )

    return errors


def bootstrap_environment(
    settings: SandboxSettings | None = None,
    verifier: Callable[..., dict[str, list[str]]] | None = None,
    *,
    auto_install: bool = True,
) -> SandboxSettings:
    cid = f"bootstrap-env-{uuid.uuid4()}"
    set_correlation_id(cid)
    sandbox_restart_total.labels(service="bootstrap", reason="bootstrap").inc()
    logger.info("bootstrap environment start", extra=log_record(event="start"))
    try:
        result = _bootstrap_environment(
            settings, verifier, auto_install=auto_install
        )
        logger.info(
            "bootstrap environment complete", extra=log_record(event="shutdown")
        )
        return result
    except Exception:
        environment_failure_total.labels(reason="bootstrap").inc()
        sandbox_crashes_total.inc()
        logger.exception(
            "bootstrap environment failure", extra=log_record(event="failure")
        )
        raise
    finally:
        set_correlation_id(None)


def _bootstrap_environment(
    settings: SandboxSettings | None = None,
    verifier: Callable[..., dict[str, list[str]]] | None = None,
    *,
    auto_install: bool = True,
) -> SandboxSettings:
    """Fully prepare the autonomous sandbox environment.

    This convenience routine ensures the environment file exists, verifies
    required dependencies, creates core SQLite databases and checks for
    optional service modules.
    """

    settings = settings or load_sandbox_settings()
    env_file = Path(settings.menace_env_file)
    created_env = not env_file.exists()
    ensure_env(str(env_file))
    # populate defaults for any missing configuration values without prompting
    DefaultConfigManager(str(env_file)).apply_defaults()
    if created_env:
        logger.info("created env file at %s", env_file)
    if verifier:
        try:
            errors = verifier(settings, auto_install)  # type: ignore[misc]
        except TypeError:
            errors = verifier(settings)  # type: ignore[misc]
    else:
        errors = _verify_required_dependencies(settings)
    if errors:
        messages: list[str] = []
        if errors.get("system"):
            messages.append(
                "Missing system packages: "
                + ", ".join(errors["system"])
                + ". Install them using your package manager."
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
        raise SystemExit("\n".join(messages))
    return initialize_autonomous_sandbox(settings)


def launch_sandbox(
    settings: SandboxSettings | None = None,
    verifier: Callable[..., None] | None = None,
) -> None:
    """Run the sandbox runner using ``settings`` for configuration."""
    cid = f"bootstrap-launch-{uuid.uuid4()}"
    set_correlation_id(cid)
    sandbox_restart_total.labels(service="bootstrap", reason="launch").inc()
    logger.info("launch sandbox start", extra=log_record(event="start"))
    try:
        settings = bootstrap_environment(settings, verifier)
        os.environ.setdefault(
            "SANDBOX_REPO_PATH", str(resolve_path(settings.sandbox_repo_path))
        )
        data_dir = resolve_path(settings.sandbox_data_dir)
        os.environ.setdefault("SANDBOX_DATA_DIR", str(data_dir))
        _cli_main([])
        logger.info("launch sandbox shutdown", extra=log_record(event="shutdown"))
    except Exception:
        sandbox_crashes_total.inc()
        logger.exception(
            "launch sandbox failure", extra=log_record(event="failure")
        )
        raise
    finally:
        set_correlation_id(None)
