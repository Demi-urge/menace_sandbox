"""Bootstrap dependency policy management.

This module centralises the configuration that drives environment bootstrapping
and startup validation.  Historically the bootstrap process always enforced the
full production dependency surface.  That behaviour is problematic for
resource-constrained developer environments (including the execution sandboxes
used for automated grading) where the majority of heavy dependencies are
unavailable.  ``DependencyPolicy`` captures a configurable view of what should
be enforced for a particular run so callers can downshift checks without
hacking in ad-hoc environment variables sprinkled across the codebase.

Policies can be declared programmatically or via ``configs/bootstrap_policy.toml``.
When a custom policy is requested the loader merges it with built-in defaults,
allowing installations to override a subset of fields without having to copy
hundreds of dependency names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import logging
import os
import platform
import tomllib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default dependency declarations -------------------------------------------------
DEFAULT_OPTIONAL_PYTHON_MODULES: tuple[str, ...] = (
    "moviepy",
    "pytube",
    "selenium",
    "pyautogui",
    "requests",
    "opencv-python",
    "numpy",
    "filelock",
    "undetected-chromedriver",
    "selenium-stealth",
    "scikit-learn",
    "beautifulsoup4",
    "scipy",
    "PyPDF2",
    "gensim",
    "SpeechRecognition",
    "gTTS",
    "SQLAlchemy",
    "sqlparse",
    "alembic",
    "psycopg2-binary",
    "pandas",
    "fuzzywuzzy[speedup]",
    "marshmallow",
    "celery",
    "pyzmq",
    "pika",
    "psutil",
    "risky",
    "networkx",
    "pulp",
    "Flask",
    "PyYAML",
    "GitPython",
    "pymongo",
    "elasticsearch",
    "Faker",
    "docker",
    "boto3",
    "prometheus-client",
    "deap",
    "simpy",
    "matplotlib",
    "Jinja2",
    "playwright",
    "fastapi",
    "pydantic",
    "pydantic-settings",
    "radon",
    "redis",
    "sentence-transformers",
    "hdbscan",
    "annoy",
    "faiss-cpu",
    "libcst",
    "kafka-python",
    "pyspark",
    "stable-baselines3",
    "torch",
    "sentry-sdk",
    "cryptography",
    "tiktoken",
    "vaderSentiment",
    "stripe>=12.5.0",
    "datasets",
)

DEFAULT_SANDBOX_MODULES: tuple[str, ...] = ("quick_fix_engine",)
DEFAULT_STRIPE_OPTIONALS: tuple[str, ...] = ("quick_fix_engine",)
DEFAULT_CRITICAL_DEPENDENCIES: Mapping[str, str] = {
    "pandas": "",
    "torch": "",
    "networkx": "",
}

# ---------------------------------------------------------------------------


def _as_tuple(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalise_dependency_mapping(data: Mapping[str, str] | None) -> Mapping[str, str]:
    if data is None:
        return {}
    return {str(key).strip(): str(value).strip() for key, value in data.items()}


@dataclass(frozen=True)
class DependencyPolicy:
    """Describe which bootstrap checks should be enforced."""

    name: str
    description: str
    project_dependencies: tuple[str, ...] | None = None
    optional_dependencies: tuple[str, ...] | None = None
    sandbox_modules: tuple[str, ...] | None = None
    critical_dependencies: Mapping[str, str] | None = None
    enforce_remote_checks: bool = True
    enforce_systemd: bool = True
    enforce_os_package_checks: bool = True
    ensure_apscheduler: bool = True
    run_database_migrations: bool = True
    provision_vector_assets: bool = True
    additional_python_dependencies: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ("git", "curl", "python3")
    windows_package_managers: tuple[str, ...] = ("winget", "choco")
    linux_package_managers: tuple[str, ...] = ("dpkg", "rpm")

    def derive(self, **overrides: object) -> "DependencyPolicy":
        """Return a copy of the policy with ``overrides`` applied."""

        return replace(self, **overrides)

    # ------------------------------------------------------------------
    def resolved_project_dependencies(self, dependencies: Sequence[str]) -> tuple[str, ...]:
        """Return project dependencies that should be enforced."""

        if self.project_dependencies is None:
            return tuple(dependencies)
        allow = {dep.lower() for dep in self.project_dependencies}
        if not allow:
            return tuple()
        return tuple(dep for dep in dependencies if dep.lower() in allow)

    # ------------------------------------------------------------------
    def resolved_optional_python_modules(self, defaults: Sequence[str]) -> tuple[str, ...]:
        """Return optional modules that should be probed."""

        if self.optional_dependencies is None:
            return tuple(defaults)
        return self.optional_dependencies

    # ------------------------------------------------------------------
    def resolved_sandbox_modules(
        self,
        defaults: Sequence[str],
        *,
        skip_stripe: bool,
        stripe_sensitive: Iterable[str],
    ) -> tuple[str, ...]:
        """Return sandbox modules to probe during startup."""

        modules = (
            defaults
            if self.sandbox_modules is None
            else self.sandbox_modules
        )
        if not modules:
            return tuple()
        stripe_sensitive_lower = {name.lower() for name in stripe_sensitive}
        if skip_stripe:
            return tuple(
                module
                for module in modules
                if module.lower() not in stripe_sensitive_lower
            )
        return tuple(modules)

    # ------------------------------------------------------------------
    def resolved_critical_dependencies(
        self, defaults: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Return critical dependency mapping to verify."""

        if self.critical_dependencies is None:
            return dict(defaults)
        if not self.critical_dependencies:
            return {}
        return dict(self.critical_dependencies)


# ---------------------------------------------------------------------------


class PolicyLoader:
    """Load :class:`DependencyPolicy` objects from configuration."""

    DEFAULT_CONFIG_PATH = Path("configs/bootstrap_policy.toml")

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------
    def resolve(
        self,
        *,
        requested: str | None = None,
        auto_install_enabled: bool | None = None,
        platform_name: str | None = None,
    ) -> DependencyPolicy:
        """Return the policy that should be used for this process."""

        requested = (requested or os.getenv("MENACE_BOOTSTRAP_PROFILE")) or None
        requested_key = requested.lower() if requested else None
        auto_install = (
            auto_install_enabled
            if auto_install_enabled is not None
            else self._auto_install_requested()
        )
        platform_name = (platform_name or platform.system()).lower()

        policies = {policy.name: policy for policy in self._builtin_policies()}
        policies.update(self._load_custom_policies())

        if requested_key:
            policy = policies.get(requested_key)
            if not policy:
                raise KeyError(f"Unknown bootstrap profile '{requested}'")
            return policy

        fallback = policies.get("full") or policies.get("minimal")
        if platform_name.startswith("win"):
            return policies.get("windows", fallback)
        if auto_install:
            return policies.get("full", fallback)
        return fallback

    # ------------------------------------------------------------------
    def available_policies(self) -> Mapping[str, DependencyPolicy]:
        """Return a mapping of policy names to definitions."""

        policies = {policy.name: policy for policy in self._builtin_policies()}
        policies.update(self._load_custom_policies())
        return policies

    # ------------------------------------------------------------------
    def _builtin_policies(self) -> Sequence[DependencyPolicy]:
        minimal = DependencyPolicy(
            name="minimal",
            description=(
                "Lightweight bootstrap that skips heavyweight dependency enforcement. "
                "Intended for sandboxes and CI environments without full access to "
                "infrastructure services."
            ),
            project_dependencies=tuple(),
            optional_dependencies=tuple(),
            sandbox_modules=tuple(),
            critical_dependencies={},
            enforce_remote_checks=False,
            enforce_systemd=False,
            enforce_os_package_checks=False,
            ensure_apscheduler=False,
            run_database_migrations=False,
            provision_vector_assets=False,
            required_commands=("git", "curl", "python3"),
        )
        full = DependencyPolicy(
            name="full",
            description="Production parity bootstrap enforcing every dependency.",
            project_dependencies=None,
            optional_dependencies=DEFAULT_OPTIONAL_PYTHON_MODULES,
            sandbox_modules=DEFAULT_SANDBOX_MODULES,
            critical_dependencies=DEFAULT_CRITICAL_DEPENDENCIES,
            enforce_remote_checks=True,
            enforce_systemd=True,
            enforce_os_package_checks=True,
            ensure_apscheduler=True,
            run_database_migrations=True,
            provision_vector_assets=True,
            required_commands=("git", "curl", "python3"),
        )
        windows = DependencyPolicy(
            name="windows",
            description=(
                "Developer workstations running Windows.  Skips Linux specific checks "
                "and uses Windows package managers when available."
            ),
            project_dependencies=tuple(),
            optional_dependencies=tuple(),
            sandbox_modules=tuple(),
            critical_dependencies={},
            enforce_remote_checks=False,
            enforce_systemd=False,
            enforce_os_package_checks=False,
            ensure_apscheduler=False,
            run_database_migrations=False,
            provision_vector_assets=False,
            required_commands=("git", "curl", "python"),
            windows_package_managers=("winget", "choco"),
        )
        return (minimal, full, windows)

    # ------------------------------------------------------------------
    def _auto_install_requested(self) -> bool:
        value = os.getenv("MENACE_AUTO_INSTALL")
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    # ------------------------------------------------------------------
    def _load_custom_policies(self) -> Mapping[str, DependencyPolicy]:
        path = self.config_path
        if not path.exists():
            return {}
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except Exception as exc:  # pragma: no cover - configuration errors
            logger.warning("Failed loading bootstrap policy configuration: %s", exc)
            return {}
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            return {}

        builtin = {policy.name: policy for policy in self._builtin_policies()}
        resolved: dict[str, DependencyPolicy] = {}

        for name, payload in profiles.items():
            if not isinstance(payload, dict):
                continue
            key = str(name).lower()
            base_name = str(payload.get("extends", "")) or None
            base_policy = None
            if base_name:
                base_policy = resolved.get(base_name.lower()) or builtin.get(
                    base_name.lower()
                )
                if base_policy is None:
                    logger.warning(
                        "bootstrap policy '%s' references unknown base '%s'", key, base_name
                    )
            description = payload.get(
                "description",
                base_policy.description if base_policy else f"Custom policy '{key}'",
            )
            overrides: dict[str, object] = {
                "name": key,
                "description": description,
            }
            if "project" in payload:
                overrides["project_dependencies"] = _as_tuple(payload.get("project"))
            if "optional" in payload:
                overrides["optional_dependencies"] = _as_tuple(payload.get("optional"))
            if "sandbox" in payload:
                overrides["sandbox_modules"] = _as_tuple(payload.get("sandbox"))
            if "critical" in payload:
                overrides["critical_dependencies"] = _normalise_dependency_mapping(
                    payload.get("critical") or {}
                )
            if "enforce_remote_checks" in payload:
                overrides["enforce_remote_checks"] = bool(payload.get("enforce_remote_checks"))
            if "enforce_systemd" in payload:
                overrides["enforce_systemd"] = bool(payload.get("enforce_systemd"))
            if "enforce_os_package_checks" in payload:
                overrides["enforce_os_package_checks"] = bool(
                    payload.get("enforce_os_package_checks")
                )
            if "ensure_apscheduler" in payload:
                overrides["ensure_apscheduler"] = bool(payload.get("ensure_apscheduler"))
            if "run_database_migrations" in payload:
                overrides["run_database_migrations"] = bool(
                    payload.get("run_database_migrations")
                )
            if "provision_vector_assets" in payload:
                overrides["provision_vector_assets"] = bool(
                    payload.get("provision_vector_assets")
                )
            if "additional_python_dependencies" in payload:
                overrides["additional_python_dependencies"] = _as_tuple(
                    payload.get("additional_python_dependencies")
                )
            if "required_commands" in payload:
                overrides["required_commands"] = _as_tuple(payload.get("required_commands"))
            if "windows_package_managers" in payload:
                overrides["windows_package_managers"] = _as_tuple(
                    payload.get("windows_package_managers")
                )
            if "linux_package_managers" in payload:
                overrides["linux_package_managers"] = _as_tuple(
                    payload.get("linux_package_managers")
                )

            if base_policy is not None:
                policy = base_policy.derive(**overrides)
            else:
                policy = DependencyPolicy(**overrides)
            resolved[key] = policy
        return resolved


__all__ = [
    "DependencyPolicy",
    "PolicyLoader",
    "DEFAULT_OPTIONAL_PYTHON_MODULES",
    "DEFAULT_SANDBOX_MODULES",
    "DEFAULT_STRIPE_OPTIONALS",
    "DEFAULT_CRITICAL_DEPENDENCIES",
]
