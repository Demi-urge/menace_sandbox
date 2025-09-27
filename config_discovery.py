from __future__ import annotations

"""Helpers for discovering and generating configuration values."""

import importlib
import os
import secrets
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable
import threading
import logging
from collections.abc import Iterator

_PARENT_PACKAGE = (__package__ or "").split(".", 1)[0] or "menace"
RAISE_ERRORS = bool(getattr(importlib.import_module(_PARENT_PACKAGE), "RAISE_ERRORS", False))

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
        self._stack_env_values: dict[str, str] = {}
        self._stack_explicit_overrides: set[str] = {
            key for key in ("STACK_STREAMING", "STACK_DATA_DIR") if key in os.environ
        }
        self._token_missing_logged = False

    TERRAFORM_PATHS = [
        "terraform",
        "infra/terraform",
        "infrastructure/terraform",
        "deployment/terraform",
    ]
    HOST_FILES = ["hosts", "hosts.txt", "cluster_hosts", "/etc/menace/hosts"]
    DEFAULT_STACK_DIR = Path("~/.cache/menace/stack")
    STACK_ENV_FILENAMES = (".stack_env", "stack.env")

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

        self.reload_stack_settings()
        self.reload_tokens()
        self._detect_hardware()
        self._detect_cloud()

    # ------------------------------------------------------------------
    def reload_tokens(self) -> None:
        """Ensure Hugging Face tokens are exposed via ``HUGGINGFACE_TOKEN``."""

        token = self._discover_huggingface_token()
        if token:
            os.environ["HUGGINGFACE_TOKEN"] = token
            self._token_missing_logged = False
            return

        self.failure_count += 1
        if not self._token_missing_logged:
            self.logger.warning(
                "huggingface token not found in environment or cache; stack ingestion may stall"
            )
            self._token_missing_logged = True
        else:
            self.logger.debug("huggingface token still missing; failure_count=%s", self.failure_count)

    # ------------------------------------------------------------------
    def reload_stack_settings(self) -> None:
        """Reload stack dataset configuration from optional env files."""

        values = self._load_stack_env()
        if values:
            # data dir should be applied first so dependent defaults use overrides
            if "STACK_DATA_DIR" in values:
                self._apply_stack_env_value("STACK_DATA_DIR", values.pop("STACK_DATA_DIR"))
            if "STACK_STREAMING" in values:
                self._apply_stack_env_value("STACK_STREAMING", values.pop("STACK_STREAMING"))
            for key, value in values.items():
                if key.startswith("STACK_"):
                    self._apply_stack_env_value(key, value)

        self._apply_stack_defaults()

    # ------------------------------------------------------------------
    def _discover_huggingface_token(self) -> str | None:
        """Return the first available Hugging Face token, if any."""

        env_candidates = (
            os.environ.get("HUGGINGFACE_TOKEN"),
            os.environ.get("HUGGINGFACE_API_TOKEN"),
            os.environ.get("HF_TOKEN"),
            os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        )
        for candidate in env_candidates:
            if candidate:
                token = candidate.strip()
                if token:
                    return token

        for path in self._huggingface_token_paths():
            try:
                data = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if data:
                return data
        return None

    # ------------------------------------------------------------------
    def _huggingface_token_paths(self) -> Iterator[Path]:
        """Yield potential Hugging Face token file locations."""

        home = Path.home()
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            yield Path(hf_home).expanduser() / "token"
        yield home / ".huggingface" / "token"
        yield home / ".cache" / "huggingface" / "token"
        yield home / ".cache" / "huggingface" / "token.txt"

    # ------------------------------------------------------------------
    def _load_stack_env(self) -> dict[str, str]:
        """Read stack-specific environment overrides from optional files."""

        result: dict[str, str] = {}
        stack_env = os.getenv("STACK_ENV_FILE")
        candidates: list[Path] = []
        if stack_env:
            candidates.append(Path(stack_env).expanduser())
        cwd = Path.cwd()
        for name in self.STACK_ENV_FILENAMES:
            candidates.append(cwd / name)
        home = Path.home()
        for name in self.STACK_ENV_FILENAMES:
            candidates.append(home / name)

        for path in candidates:
            try:
                if not path.exists() or not path.is_file():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.startswith("export "):
                        stripped = stripped[len("export ") :]
                    if "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue
                    result[key] = value.strip().strip('"')
            except OSError as exc:
                self.logger.debug("unable to read stack env file %s: %s", path, exc)
        return result

    # ------------------------------------------------------------------
    def _apply_stack_defaults(self) -> None:
        base_dir = os.environ.get("STACK_DATA_DIR")
        if not base_dir:
            base_path = self.DEFAULT_STACK_DIR.expanduser()
            self._apply_stack_env_value("STACK_DATA_DIR", str(base_path))
            base_dir = str(base_path)
        base_path = Path(base_dir).expanduser()
        if not os.environ.get("STACK_STREAMING"):
            self._apply_stack_env_value("STACK_STREAMING", "0")
        defaults = {
            "STACK_METADATA_DB": str(base_path / "stack_metadata.db"),
            "STACK_METADATA_PATH": str(base_path / "stack_metadata.db"),
            "STACK_CACHE_DIR": str(base_path / "cache"),
            "STACK_VECTOR_PATH": str(base_path / "stack_vectors"),
        }
        for key, value in defaults.items():
            if not os.environ.get(key):
                self._apply_stack_env_value(key, value)

    # ------------------------------------------------------------------
    def _apply_stack_env_value(self, key: str, value: str) -> None:
        if key in self._stack_explicit_overrides:
            return

        current = os.environ.get(key)
        previous_default = self._stack_env_values.get(key)
        if key in {"STACK_STREAMING", "STACK_DATA_DIR"}:
            if current is not None and previous_default is None and current != value:
                self._stack_explicit_overrides.add(key)
                return
            if (
                current is not None
                and previous_default is not None
                and current != previous_default
                and current != value
            ):
                self._stack_explicit_overrides.add(key)
                return

        if current != value:
            os.environ[key] = value
        self._stack_env_values[key] = value

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


_DEFAULT_VARS = [
    # Stripe keys are intentionally excluded; stripe_billing_router loads them
    # directly from the environment to avoid storing secrets in config files.
    "DATABASE_URL",
    "OPENAI_API_KEY",
]


def _stack_auto_defaults() -> dict[str, str]:
    base_dir = os.getenv("STACK_DATA_DIR")
    if not base_dir:
        base_dir = str(ConfigDiscovery.DEFAULT_STACK_DIR.expanduser())
    base_path = Path(base_dir).expanduser()
    defaults = {
        "STACK_STREAMING": "0",
        "STACK_DATA_DIR": str(base_path),
        "STACK_METADATA_DB": str(base_path / "stack_metadata.db"),
        "STACK_METADATA_PATH": str(base_path / "stack_metadata.db"),
        "STACK_CACHE_DIR": str(base_path / "cache"),
        "STACK_VECTOR_PATH": str(base_path / "stack_vectors"),
    }
    return defaults


def _generate_value(name: str) -> str:
    if name.endswith("_URL"):
        return f"sqlite:///{name.lower()}.db"
    return secrets.token_hex(16)


def ensure_config(vars: Iterable[str] | None = None, *, save_path: str = ".env.auto") -> None:
    required = vars or _DEFAULT_VARS
    missing = {v: _generate_value(v) for v in required if not os.getenv(v)}
    for key, value in _stack_auto_defaults().items():
        if not os.getenv(key):
            missing.setdefault(key, value)
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
