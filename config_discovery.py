from __future__ import annotations

"""Helpers for discovering and generating configuration values."""

import os
import secrets
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable
import threading
import logging

import yaml

try:  # pragma: no cover - allow lightweight test stubs
    from . import RAISE_ERRORS
except Exception:  # pragma: no cover - fallback when package stubbed
    RAISE_ERRORS = False  # type: ignore[assignment]

logger = logging.getLogger(__name__)

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class ConfigDiscovery:
    """Discover local configuration such as Terraform and host info."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_count = 0

    TERRAFORM_PATHS = [
        "terraform",
        "infra/terraform",
        "infrastructure/terraform",
        "deployment/terraform",
    ]
    HOST_FILES = ["hosts", "hosts.txt", "cluster_hosts", "/etc/menace/hosts"]
    STACK_ENV_KEYS = (
        "STACK_STREAMING",
        "STACK_HF_TOKEN",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
    )
    STACK_CONFIG_HINTS = (
        ("stack_dataset", "index_path", "STACK_INDEX_PATH"),
        ("stack_dataset", "metadata_path", "STACK_METADATA_PATH"),
        ("context_builder", "stack", "index_path", "STACK_INDEX_PATH"),
        ("context_builder", "stack", "metadata_path", "STACK_METADATA_PATH"),
    )
    STACK_ENV_FILES = (
        Path(".env"),
        Path(".env.local"),
        Path(".env.auto"),
        Path("config/.env"),
        Path("config/.env.local"),
        Path("config/stack.env"),
    )

    def _find_terraform_dir(self) -> str | None:
        for name in self.TERRAFORM_PATHS:
            path = Path(name)
            if path.is_dir() and list(path.glob("*.tf")):
                return str(path.resolve())
        return None

    def _read_hosts(self, path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def _find_hosts(self) -> list[str]:
        for name in self.HOST_FILES:
            path = Path(name)
            if path.exists():
                hosts = self._read_hosts(path)
                if hosts:
                    return hosts
        return []

    # ------------------------------------------------------------------
    def discover(self) -> None:
        """Populate environment variables from common config locations."""

        if not os.getenv("TERRAFORM_DIR"):
            tf_dir = self._find_terraform_dir()
            if tf_dir:
                os.environ["TERRAFORM_DIR"] = tf_dir

        hosts = []
        if not os.getenv("CLUSTER_HOSTS") or not os.getenv("REMOTE_HOSTS"):
            hosts = self._find_hosts()
        if hosts:
            os.environ.setdefault("CLUSTER_HOSTS", ",".join(hosts))
            os.environ.setdefault("REMOTE_HOSTS", ",".join(hosts))

        stack_hints = self._collect_stack_hints()
        for key, value in stack_hints.items():
            if key in self.STACK_ENV_KEYS and key not in os.environ and value is not None:
                os.environ[key] = value
        token_hint = stack_hints.get("STACK_HF_TOKEN")
        if token_hint:
            os.environ.setdefault("HUGGINGFACE_TOKEN", token_hint)

        self._ensure_stack_env(hints=stack_hints)
        self._detect_hardware()
        self._detect_cloud()

    # ------------------------------------------------------------------
    def _parse_env_file(self, path: Path) -> dict[str, str]:
        data: dict[str, str] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith(("'", '"')) and value.endswith(value[0]):
                    value = value[1:-1]
                data[key] = value
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("failed reading %s: %s", path, exc)
        return data

    def _collect_stack_hints(self) -> dict[str, str]:
        hints: dict[str, str] = {}
        for path in self.STACK_ENV_FILES:
            if not path.exists():
                continue
            entries = self._parse_env_file(path)
            for key in self.STACK_ENV_KEYS:
                if key in entries:
                    hints[key] = entries[key]

        context_path = Path("config/stack_context.yaml")
        if context_path.exists():
            try:
                data = yaml.safe_load(context_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("failed reading %s: %s", context_path, exc)
            else:
                for hint in self.STACK_CONFIG_HINTS:
                    *keys, env_name = hint
                    section = data
                    try:
                        for key in keys:
                            if section is None:
                                break
                            section = section.get(key)
                    except AttributeError:
                        section = None
                    if isinstance(section, dict):
                        continue
                    if section in {None, "", "~"}:
                        continue
                    value = section
                    if isinstance(value, (str, Path)):
                        text = str(value).strip()
                        if text:
                            hints.setdefault(env_name, text)

        return hints

    def _ensure_stack_env(
        self,
        save_path: str | Path = ".env.auto",
        *,
        hints: dict[str, str] | None = None,
    ) -> None:
        """Ensure Stack-related environment variables exist.

        ``ConfigDiscovery`` historically focussed on infrastructure hints such
        as Terraform directories.  Stack ingestion and retrieval now depend on
        a small set of environment variables.  This helper mirrors the
        behaviour of :func:`ensure_config` by exporting sensible defaults to the
        process environment and persisting them to ``.env.auto`` so other
        services pick them up automatically.
        """

        env_path = Path(save_path)
        existing: dict[str, str] = {}
        if env_path.exists():
            existing = self._parse_env_file(env_path)

        hints = hints or {}
        combined: dict[str, str] = dict(existing)
        for key, value in hints.items():
            if value is not None:
                combined[key] = value

        env_updates: dict[str, str] = {}
        generated_defaults: set[str] = set()
        file_values: dict[str, str] = dict(existing)
        needs_rewrite = False

        placeholders = {
            "STACK_STREAMING": "0",
            "STACK_HF_TOKEN": "",
            "STACK_INDEX_PATH": "",
            "STACK_METADATA_PATH": "",
        }

        def _truthy(value: str | None) -> bool:
            if value is None:
                return False
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}

        for key, placeholder in placeholders.items():
            value = os.environ.get(key)
            if value is None:
                value = combined.get(key)
            if value is None:
                value = placeholder
                generated_defaults.add(key)
            if key not in existing:
                env_updates[key] = value
            elif existing.get(key) != value:
                needs_rewrite = True
            file_values[key] = value
            os.environ.setdefault(key, value)

        token_sources = (
            os.environ.get("STACK_HF_TOKEN"),
            os.environ.get("HUGGINGFACE_TOKEN"),
            os.environ.get("HUGGINGFACE_API_TOKEN"),
            os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            os.environ.get("HF_TOKEN"),
            combined.get("STACK_HF_TOKEN"),
            combined.get("HUGGINGFACE_TOKEN"),
        )
        token_value = next((val for val in token_sources if val), None)
        if token_value is None:
            token_value = combined.get("HUGGINGFACE_TOKEN")
        if token_value is None:
            token_value = ""
            if "HUGGINGFACE_TOKEN" not in existing:
                env_updates["HUGGINGFACE_TOKEN"] = token_value
            else:
                if existing.get("HUGGINGFACE_TOKEN") != token_value:
                    needs_rewrite = True
            os.environ.setdefault("HUGGINGFACE_TOKEN", token_value)
        else:
            os.environ["HUGGINGFACE_TOKEN"] = token_value
            if "HUGGINGFACE_TOKEN" not in existing:
                env_updates["HUGGINGFACE_TOKEN"] = token_value
            elif existing.get("HUGGINGFACE_TOKEN") != token_value:
                needs_rewrite = True
        file_values["HUGGINGFACE_TOKEN"] = token_value

        missing_credentials = []
        for alias in ("STACK_HF_TOKEN", "HUGGINGFACE_TOKEN"):
            if not os.environ.get(alias):
                missing_credentials.append(alias)
        if not token_value:
            missing_credentials.append("HUGGINGFACE_TOKEN")

        stack_enabled = _truthy(os.environ.get("STACK_STREAMING"))
        if not stack_enabled and "STACK_STREAMING" not in generated_defaults:
            self.logger.warning(
                "STACK_STREAMING disabled via environment; Stack ingestion will not run"
            )

        if needs_rewrite:
            file_values.update({k: v for k, v in env_updates.items()})
            try:
                with env_path.open("w", encoding="utf-8") as fh:
                    for key, value in file_values.items():
                        fh.write(f"{key}={value}\n")
            except Exception as exc:
                logger.exception("Failed to save config to %s: %s", env_path, exc)
                if RAISE_ERRORS:
                    raise
        elif env_updates:
            try:
                with env_path.open("a", encoding="utf-8") as fh:
                    for key, value in env_updates.items():
                        fh.write(f"{key}={value}\n")
            except Exception as exc:
                logger.exception("Failed to save config to %s: %s", env_path, exc)
                if RAISE_ERRORS:
                    raise

        if missing_credentials:
            joined = ", ".join(sorted(set(missing_credentials)))
            self.logger.warning(
                "Stack processing disabled or degraded; set %s in %s",
                joined,
                env_path,
            )

    # ------------------------------------------------------------------
    def _detect_hardware(self) -> None:
        """Set GPU related environment variables when possible."""
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
                gpus = len([line for line in out.splitlines() if line.strip()])
            except Exception:
                gpus = 0
        else:
            gpus = 0
        os.environ.setdefault("NUM_GPUS", str(gpus))
        os.environ.setdefault("GPU_AVAILABLE", "1" if gpus > 0 else "0")

    # ------------------------------------------------------------------
    def _detect_cloud(self) -> None:
        """Identify cloud provider and export CPU and memory info."""

        def _probe(url: str, headers: dict[str, str] | None = None) -> bool:
            try:
                req = urllib.request.Request(url, headers=headers or {})
                with urllib.request.urlopen(req, timeout=0.2):
                    return True
            except Exception:
                return False

        provider = os.getenv("CLOUD_PROVIDER")
        if not provider:
            if os.getenv("AWS_EXECUTION_ENV") or _probe("http://169.254.169.254/latest/meta-data/"):
                provider = "AWS"
            elif os.getenv("GOOGLE_CLOUD_PROJECT") or _probe("http://metadata.google.internal"):
                provider = "GCP"
            elif os.getenv("MSI_ENDPOINT") or _probe(
                "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                {"Metadata": "true"},
            ):
                provider = "AZURE"

        if provider:
            os.environ.setdefault("CLOUD_PROVIDER", provider)

        os.environ.setdefault("TOTAL_CPUS", str(os.cpu_count() or 0))
        if psutil:
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
            os.environ.setdefault("TOTAL_MEMORY_GB", str(round(mem_gb, 2)))

    # ------------------------------------------------------------------
    def run_continuous(
        self, interval: float = 3600.0, *, stop_event: threading.Event | None = None
    ) -> threading.Thread:
        """Run :meth:`discover` repeatedly in a thread."""

        if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
            return self._thread
        self._stop = stop_event or threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.discover()
                except Exception as exc:
                    self.logger.exception("config discovery failed: %s", exc)
                    self.failure_count += 1
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self._thread


_STACK_PLACEHOLDERS = {
    "STACK_STREAMING": "0",
    "STACK_HF_TOKEN": "",
    "STACK_INDEX_PATH": "",
    "STACK_METADATA_PATH": "",
}

_DEFAULT_VARS = [
    # Stripe keys are intentionally excluded; stripe_billing_router loads them
    # directly from the environment to avoid storing secrets in config files.
    "DATABASE_URL",
    "OPENAI_API_KEY",
    *list(_STACK_PLACEHOLDERS.keys()),
]


def _generate_value(name: str) -> str:
    if name in _STACK_PLACEHOLDERS:
        return _STACK_PLACEHOLDERS[name]
    if name.endswith("_URL"):
        return f"sqlite:///{name.lower()}.db"
    return secrets.token_hex(16)


def ensure_config(vars: Iterable[str] | None = None, *, save_path: str = ".env.auto") -> None:
    required = vars or _DEFAULT_VARS
    missing = {v: _generate_value(v) for v in required if not os.getenv(v)}
    if not missing:
        return
    for k, v in missing.items():
        os.environ[k] = v
    try:
        with open(save_path, "a", encoding="utf-8") as fh:
            for k, v in missing.items():
                fh.write(f"{k}={v}\n")
    except Exception as exc:
        logger.exception("Failed to save config to %s: %s", save_path, exc)
        if RAISE_ERRORS:
            raise


__all__ = ["ensure_config", "ConfigDiscovery"]
