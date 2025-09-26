"""Application configuration loader.

Loads base settings from ``config/settings.yaml`` and overlays profile-specific
values from ``config/<mode>.yaml``. Environment variables defined in a ``.env``
file are loaded beforehand so they can override values from those files. When a
``VaultSecretProvider`` is available, any secrets it supplies take precedence
over both environment variables and YAML/JSON configuration.

Precedence: ``Vault`` > ``Environment variables`` > ``YAML/JSON``.

The configuration schema is validated using Pydantic models to provide helpful
error messages for missing or invalid fields.

When an optional :class:`~unified_event_bus.EventBus` is configured, a
``config.reload`` event is published after reloads or overrides. Listeners may
use this to adjust logger levels, risk thresholds and other runtime settings.
"""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Set, Optional
from types import SimpleNamespace
import sys
import site
import copy

from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from dynamic_path_router import get_project_root

try:  # pragma: no cover - optional dependencies
    from .unified_config_store import UnifiedConfigStore
except Exception:  # pragma: no cover - fallback when module missing
    UnifiedConfigStore = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency
    from .vault_secret_provider import VaultSecretProvider
except Exception:  # pragma: no cover - fallback when module missing
    VaultSecretProvider = None  # type: ignore[misc,assignment]

# Watchdog is an optional dependency and a local module named ``watchdog.py``
# exists in this package. To import the third-party library we temporarily
# adjust ``sys.path`` so that site-packages are searched first and remove any
# previously loaded module named ``watchdog``.
_orig_sys_path = list(sys.path)
try:  # pragma: no cover - optional dependency
    sys.modules.pop("watchdog", None)
    sys.path = site.getsitepackages() + sys.path
    from watchdog.events import FileSystemEventHandler  # type: ignore
    from watchdog.observers import Observer  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    Observer = None  # type: ignore[misc,assignment]
    FileSystemEventHandler = object  # type: ignore[misc,assignment]
finally:  # ensure path restoration
    sys.path = _orig_sys_path

import yaml  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

try:  # noqa: E402 - prefer attribute detection over import side effects
    import pydantic as _pydantic
except Exception as exc:  # pragma: no cover - pydantic should always be present
    raise ImportError("pydantic is required for configuration models") from exc

if all(
    hasattr(_pydantic, attr)
    for attr in ("ConfigDict", "field_validator", "model_validator")
):
    ConfigDict = getattr(_pydantic, "ConfigDict")  # type: ignore[assignment]
    field_validator = getattr(_pydantic, "field_validator")  # type: ignore[assignment]
    model_validator = getattr(_pydantic, "model_validator")  # type: ignore[assignment]
    _PYDANTIC_V2 = True
else:  # pragma: no cover - executed when running under pydantic v1
    from pydantic import root_validator, validator  # type: ignore

    _PYDANTIC_V2 = False

    def field_validator(*fields, **kwargs):  # type: ignore[override]
        """Compatibility shim for :func:`pydantic.field_validator` on v1."""

        decorator = validator(*fields, allow_reuse=True, **kwargs)

        def wrapper(func):
            method = func.__func__ if isinstance(func, classmethod) else func

            def _validator(cls, value, values, config, field):  # type: ignore[override]
                info = SimpleNamespace(field_name=getattr(field, "name", None))
                return method(cls, value, info)

            _validator.__name__ = getattr(method, "__name__", "validator")
            _validator.__qualname__ = getattr(method, "__qualname__", _validator.__qualname__)

            return decorator(_validator)

        return wrapper

    def model_validator(*, mode):  # type: ignore[override]
        """Compatibility shim for :func:`pydantic.model_validator` on v1."""

        if mode != "after":  # pragma: no cover - other modes unused
            raise ValueError("pydantic v1 fallback only supports mode='after'")

        def decorator(func):
            method = func.__func__ if isinstance(func, classmethod) else func

            def _validator(cls, values):
                instance = cls.construct(**values)  # type: ignore[attr-defined]
                result = method(instance)
                if isinstance(result, cls):
                    return result.dict()
                return values if result is None else result

            _validator.__name__ = getattr(method, "__name__", "validator")
            _validator.__qualname__ = getattr(method, "__qualname__", _validator.__qualname__)
            _validator.__doc__ = getattr(method, "__doc__", None)

            return root_validator(skip_on_failure=True, allow_reuse=True)(_validator)

        return decorator

    class ConfigDict(dict):  # type: ignore[misc,override]
        """Minimal stand-in so attribute access succeeds under v1."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .unified_event_bus import EventBus


logger = logging.getLogger(__name__)


class _StrictBaseModel(BaseModel):
    """Base model enforcing ``extra = forbid`` across Pydantic versions."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - exercised only on pydantic v1

        class Config:
            extra = "forbid"


# ---------------------------------------------------------------------------
# Pydantic models defining the configuration schema
# ---------------------------------------------------------------------------


class Paths(_StrictBaseModel):
    """File system locations used by the application."""

    data_dir: str
    log_dir: str

    @field_validator("data_dir", "log_dir")
    @classmethod
    def _not_empty(cls, value: str, info) -> str:  # type: ignore[override]
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class Thresholds(_StrictBaseModel):
    """Operational thresholds expressed as floats between 0 and 1."""

    error: float = Field(ge=0.0, le=1.0)
    alert: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> "Thresholds":
        if self.alert <= self.error:
            raise ValueError("alert must be greater than error")
        return self


class APIKeys(_StrictBaseModel):
    """External service authentication keys."""

    openai: str
    serp: str

    @field_validator("openai", "serp")
    @classmethod
    def _non_blank(cls, value: str, info) -> str:  # type: ignore[override]
        if not value or "REPLACE" in value:
            raise ValueError(f"{info.field_name} API key must be provided")
        return value


class Logging(_StrictBaseModel):
    """Logging configuration."""

    verbosity: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class Vector(_StrictBaseModel):
    """Vector search parameters."""

    dimensions: int = Field(gt=0)
    distance_metric: str


class VectorStoreConfig(_StrictBaseModel):
    """Backend configuration for vector storage."""

    backend: str = Field(default="faiss")
    path: str = Field(default="vectors.index")


class BotConfig(_StrictBaseModel):
    """Bot tuning parameters."""

    learning_rate: float = Field(gt=0)
    epsilon: float = Field(ge=0.0, le=1.0)


class StackDatasetConfig(_StrictBaseModel):
    """Configuration shared by Stack ingestion and retrieval."""

    enabled: bool = False
    languages: Set[str] = Field(
        default_factory=lambda: {"python"},
        description="Whitelisted programming languages from The Stack dataset",
    )
    max_lines: int = Field(
        200,
        ge=0,
        description="Maximum number of lines retained per Stack snippet",
    )
    max_bytes: int | None = Field(
        262_144,
        ge=0,
        description="Maximum number of bytes preserved for each Stack file",
    )
    retrieval_top_k: int = Field(
        3,
        ge=0,
        description="Maximum Stack records to retrieve per query",
    )
    index_path: str | None = Field(
        None,
        description="Optional override for the Stack vector index location",
    )
    metadata_path: str | None = Field(
        None,
        description="Optional override for the Stack metadata SQLite database",
    )
    cache_dir: str | None = Field(
        None,
        description="Directory used for Stack snippet caches and progress files",
    )
    progress_path: str | None = Field(
        None,
        description="Explicit checkpoint file tracking Stack ingestion progress",
    )
    chunk_lines: int = Field(
        512,
        ge=1,
        description="Number of lines per Stack chunk during embedding",
    )

    @field_validator("languages")
    @classmethod
    def _normalise_languages(
        cls, value: Set[str], _info
    ) -> Set[str]:  # type: ignore[override]
        normalised = {
            str(language).strip().lower()
            for language in value
            if isinstance(language, str) and language.strip()
        }
        return normalised


class ContextBuilderConfig(_StrictBaseModel):
    """Context builder tuning parameters."""

    max_tokens: int = 800
    db_weights: Dict[str, float] = Field(
        default_factory=dict, description="Score multipliers per origin database"
    )
    ranking_weight: float = Field(
        1.0, description="Default multiplier when ranking model absent or fails"
    )
    roi_weight: float = Field(
        1.0, description="Default multiplier when ROI tracker lacks bias"
    )
    recency_weight: float = Field(
        1.0, description="Multiplier applied to recency when ranking patches"
    )
    safety_weight: float = Field(
        1.0,
        description=(
            "Weight applied to safety signals such as "
            "win/regret rate and alignment severity"
        ),
    )
    regret_penalty: float = Field(
        1.0, description="Penalty multiplier for regret rate when ranking results"
    )
    alignment_penalty: float = Field(
        1.0, description="Penalty multiplier for alignment severity"
    )
    alert_penalty: float = Field(
        1.0, description="Penalty applied per semantic alert"
    )
    risk_penalty: float = Field(
        1.0,
        description="Penalty multiplier applied to risk scores from patch safety checks",
    )
    roi_tag_penalties: Dict[str, float] = Field(
        default_factory=dict,
        description="Score adjustments applied based on ROI tags",
    )
    enhancement_weight: float = Field(
        1.0,
        description="Multiplier applied to enhancement_score when ranking patches",
    )
    max_alignment_severity: float = Field(
        1.0, description="Skip vectors with alignment severity above this value",
    )
    max_alerts: int = Field(
        5, description="Skip vectors with more than this number of semantic alerts",
    )
    license_denylist: Set[str] = Field(
        default_factory=lambda: set(_LICENSE_DENYLIST.values()),
        description="Skip vectors carrying these licenses",
    )
    precise_token_count: bool = Field(
        True,
        description=(
            "Use a tokenizer such as tiktoken for exact token counting when available;"
            " if set to False or the dependency is missing, a regex approximation is used."
        ),
    )
    embedding_check_interval: float = Field(
        0,
        description="Minutes between background embedding freshness checks",
    )
    max_diff_lines: int = Field(
        200,
        description="Truncate diffs to this many lines before summarisation",
    )
    similarity_metric: str = Field(
        "cosine",
        description="Similarity metric for patch examples. Options: cosine or inner_product",
    )
    prompt_max_tokens: int = Field(
        800,
        description="Token budget for prompts generated via ContextBuilder.build_prompt",
    )
    prompt_score_weight: float = Field(
        1.0,
        description="Multiplier applied to similarity scores when ranking prompt examples",
    )
    stack: StackDatasetConfig = Field(
        default_factory=StackDatasetConfig,
        description="Stack retrieval defaults applied when embeddings are available",
    )
    stack_prompt_enabled: bool = Field(
        True,
        description="Include Stack snippets when assembling prompts by default",
    )
    stack_prompt_limit: int = Field(
        2,
        ge=0,
        description="Maximum number of Stack snippets surfaced in prompts (0 disables)",
    )
    stack_enabled: bool | None = Field(
        None,
        description="Toggle Stack retrieval when assembling prompts",
    )
    stack_languages: Set[str] | None = Field(
        default=None,
        description="Preferred programming languages for Stack retrieval",
    )
    stack_max_lines: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of lines retained per Stack snippet",
    )
    stack_max_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of bytes retained per Stack snippet (None keeps all)",
    )
    stack_top_k: int | None = Field(
        default=None,
        ge=0,
        description="Number of Stack snippets retrieved per query",
    )
    stack_index_path: Optional[str] = Field(
        default=None,
        description="Explicit path to the Stack vector index",
    )
    stack_metadata_path: Optional[str] = Field(
        default=None,
        description="Explicit path to the Stack metadata database",
    )
    stack_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory used for Stack snippet caches",
    )
    stack_progress_path: Optional[str] = Field(
        default=None,
        description="File path used to persist Stack ingestion progress",
    )
    stack_requests_per_minute: int | None = Field(
        default=None,
        ge=0,
        description="Optional rate limit for Stack requests per minute",
    )
    stack_tokens_per_minute: int | None = Field(
        default=None,
        ge=0,
        description="Optional rate limit for Stack tokens per minute",
    )

    @model_validator(mode="after")
    def _sync_stack_fields(self) -> "ContextBuilderConfig":
        stack_cfg = getattr(self, "stack", None)
        if stack_cfg is None:
            stack_cfg = StackDatasetConfig()
        elif not isinstance(stack_cfg, StackDatasetConfig):
            try:
                stack_cfg = StackDatasetConfig.model_validate(stack_cfg)
            except Exception:
                stack_cfg = StackDatasetConfig()

        if self.stack_enabled is None:
            self.stack_enabled = bool(getattr(stack_cfg, "enabled", False))
        else:
            stack_cfg.enabled = bool(self.stack_enabled)

        languages = self.stack_languages
        if languages is None:
            languages = set(getattr(stack_cfg, "languages", set()))
        normalised_languages = {
            str(language).strip().lower()
            for language in (languages or set())
            if isinstance(language, str) and language.strip()
        }
        self.stack_languages = normalised_languages
        stack_cfg.languages = normalised_languages

        if self.stack_top_k is None:
            self.stack_top_k = getattr(stack_cfg, "retrieval_top_k", None)
        else:
            try:
                stack_cfg.retrieval_top_k = int(self.stack_top_k)
            except Exception:
                stack_cfg.retrieval_top_k = getattr(stack_cfg, "retrieval_top_k", 0)

        if self.stack_max_lines is None:
            self.stack_max_lines = getattr(stack_cfg, "max_lines", None)
        else:
            try:
                stack_cfg.max_lines = int(self.stack_max_lines)
            except Exception:
                stack_cfg.max_lines = getattr(stack_cfg, "max_lines", 0)

        if self.stack_max_bytes is None:
            self.stack_max_bytes = getattr(stack_cfg, "max_bytes", None)
        else:
            try:
                stack_cfg.max_bytes = None if self.stack_max_bytes is None else int(self.stack_max_bytes)
            except Exception:
                stack_cfg.max_bytes = getattr(stack_cfg, "max_bytes", None)

        if self.stack_index_path is None:
            self.stack_index_path = getattr(stack_cfg, "index_path", None)
        else:
            stack_cfg.index_path = self.stack_index_path

        if self.stack_metadata_path is None:
            self.stack_metadata_path = getattr(stack_cfg, "metadata_path", None)
        else:
            stack_cfg.metadata_path = self.stack_metadata_path

        if self.stack_cache_dir is None:
            self.stack_cache_dir = getattr(stack_cfg, "cache_dir", None)
        else:
            stack_cfg.cache_dir = self.stack_cache_dir

        if self.stack_progress_path is None:
            self.stack_progress_path = getattr(stack_cfg, "progress_path", None)
        else:
            stack_cfg.progress_path = self.stack_progress_path

        self.stack = stack_cfg
        return self


class Config(_StrictBaseModel):
    """Top-level application configuration."""

    paths: Paths
    thresholds: Thresholds
    api_keys: APIKeys
    logging: Logging
    vector: Vector
    vector_store: VectorStoreConfig = VectorStoreConfig()
    bot: BotConfig
    context_builder: ContextBuilderConfig = ContextBuilderConfig()
    stack_dataset: StackDatasetConfig = StackDatasetConfig()
    watch_config: bool = True

    # ------------------------------------------------------------------
    # Runtime modification helpers
    # ------------------------------------------------------------------

    def apply_overrides(self, mapping: Dict[str, Any]) -> "Config":
        """Return a new ``Config`` with *mapping* merged into the current data."""

        data = self.model_dump()
        old = copy.deepcopy(data)
        _merge_dict(data, mapping)
        cfg = Config.model_validate(data)
        if _EVENT_BUS is not None:
            try:  # pragma: no cover - best effort only
                payload = {"config": data, "diff": _dict_diff(old, data)}
                _EVENT_BUS.publish("config.reload", payload)
            except Exception:
                logger.exception("failed publishing config reload event")
        return cfg

    @classmethod
    def from_overrides(
        cls,
        overrides: Dict[str, Any],
        mode: str | None = None,
        event_bus: "EventBus" | None = None,
    ) -> "Config":
        """Load configuration and apply *overrides* for testing.

        This helper mirrors ``load_config`` but allows tests to provide a
        dictionary of values that should be merged on top of the loaded
        configuration without going through CLI parsing.
        """

        return load_config(mode=mode, overrides=overrides, event_bus=event_bus)

    # ------------------------------------------------------------------
    @staticmethod
    def get_secret(name: str) -> str | None:
        """Return secret *name* from loaded env or vault fallback."""

        env_name = name.upper()
        secret = os.getenv(env_name)
        if secret:
            return secret

        provider = None
        if _CONFIG_STORE is not None:
            provider = getattr(_CONFIG_STORE, "vault", None)
        elif VaultSecretProvider is not None:
            try:  # pragma: no cover - best effort only
                provider = VaultSecretProvider()
            except Exception:
                provider = None

        if provider is not None:
            try:  # pragma: no cover - best effort only
                secret = provider.get(env_name)
                if secret:
                    os.environ[env_name] = secret
                return secret
            except Exception:
                logger.exception("failed fetching secret %s from vault", env_name)
        return None


# ---------------------------------------------------------------------------
# Configuration loading utilities
# ---------------------------------------------------------------------------

BASE_DIR = get_project_root()
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
STACK_CONTEXT_FILE = CONFIG_DIR / "stack_context.yaml"
STACK_THRESHOLD_HINTS = CONFIG_DIR / "self_coding_thresholds.yaml"

_MODE: str | None = None
_CONFIG_PATH: Path | None = None
_OVERRIDES: Dict[str, Any] = {}
_CONFIG_STORE: "UnifiedConfigStore" | None = None
CONFIG: Config | None = None
_EVENT_BUS: "EventBus" | None = None
_WATCHER: "Observer" | None = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*."""
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _stack_context_overrides() -> Dict[str, Any]:
    """Extract Stack context overrides from ``self_coding_thresholds.yaml``."""

    if not STACK_THRESHOLD_HINTS.exists():
        return {}
    try:
        raw = _load_yaml(STACK_THRESHOLD_HINTS)
    except Exception:  # pragma: no cover - diagnostics handled elsewhere
        logger.exception("failed loading %s", STACK_THRESHOLD_HINTS)
        return {}

    if not isinstance(raw, dict):
        return {}

    stack_section = raw.get("stack")
    if not isinstance(stack_section, dict):
        return {}

    overrides: Dict[str, Any] = {}

    context_defaults = stack_section.get("context_builder")
    dataset_defaults = stack_section.get("dataset")

    def _is_blank(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    context_builder_updates: Dict[str, Any] = {}
    stack_updates: Dict[str, Any] = {}

    if isinstance(context_defaults, dict):
        mapping = {
            "enabled": ("stack_enabled", "enabled"),
            "languages": ("stack_languages", "languages"),
            "max_lines": ("stack_max_lines", "max_lines"),
            "max_bytes": ("stack_max_bytes", "max_bytes"),
            "top_k": ("stack_top_k", "retrieval_top_k"),
            "index_path": ("stack_index_path", "index_path"),
            "metadata_path": ("stack_metadata_path", "metadata_path"),
            "cache_dir": ("stack_cache_dir", "cache_dir"),
            "progress_path": ("stack_progress_path", "progress_path"),
            "requests_per_minute": ("stack_requests_per_minute", None),
            "tokens_per_minute": ("stack_tokens_per_minute", None),
        }
        for source, (context_key, stack_key) in mapping.items():
            if source not in context_defaults:
                continue
            value = context_defaults[source]
            if _is_blank(value):
                continue
            context_builder_updates[context_key] = value
            if stack_key:
                stack_updates[stack_key] = value

    if context_builder_updates or stack_updates:
        cb_section: Dict[str, Any] = dict(context_builder_updates)
        if stack_updates:
            cb_section.setdefault("stack", {}).update(stack_updates)
        overrides["context_builder"] = cb_section

    if isinstance(dataset_defaults, dict):
        dataset_mapping = {
            "enabled": "enabled",
            "languages": "languages",
            "max_lines": "max_lines",
            "max_bytes": "max_bytes",
            "top_k": "retrieval_top_k",
            "index_path": "index_path",
            "metadata_path": "metadata_path",
            "cache_dir": "cache_dir",
            "progress_path": "progress_path",
            "chunk_lines": "chunk_lines",
        }
        dataset_overrides = {}
        for source, dest in dataset_mapping.items():
            if source not in dataset_defaults:
                continue
            value = dataset_defaults[source]
            if _is_blank(value):
                continue
            dataset_overrides[dest] = value
        if dataset_overrides:
            overrides.setdefault("stack_dataset", {}).update(dataset_overrides)

    return overrides


def _dict_diff(old: Dict[str, Any] | None, new: Dict[str, Any]) -> Dict[str, Any]:
    """Return a nested dict of keys whose values changed from *old* to *new*."""

    if old is None:
        return new
    diff: Dict[str, Any] = {}
    for key, new_val in new.items():
        old_val = old.get(key) if isinstance(old, dict) else None
        if isinstance(new_val, dict) and isinstance(old_val, dict):
            nested = _dict_diff(old_val, new_val)
            if nested:
                diff[key] = nested
        elif old_val != new_val:
            diff[key] = new_val
    return diff


def _env_flag(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


_SECRET_TOKEN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def _looks_like_secret_token(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    core = stripped.rstrip("=")
    if len(core) not in {43, 44}:
        return False
    return all(char in _SECRET_TOKEN_CHARS for char in core)


def _stack_env_text(name: str) -> str | None:
    value = _env_text(name)
    if value and _looks_like_secret_token(value):
        return None
    return value


class _ConfigChangeHandler(FileSystemEventHandler):
    """Watchdog handler that reloads configuration on file changes."""

    def on_modified(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if _EVENT_BUS is not None:
            try:  # pragma: no cover - best effort only
                _EVENT_BUS.publish("config:file_change", {"path": event.src_path})
            except Exception:
                logger.exception("failed publishing config change event")
        try:  # pragma: no cover - best effort only
            reload()
        except Exception:
            logger.exception("failed reloading config after change")

    def on_moved(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        self.on_modified(event)


def _start_watcher() -> None:
    """Start a watchdog observer on the configuration directory."""

    global _WATCHER
    if _WATCHER is not None or Observer is None:
        return
    handler = _ConfigChangeHandler()
    observer = Observer()
    observer.schedule(handler, str(CONFIG_DIR.resolve()), recursive=False)
    observer.daemon = True
    observer.start()
    _WATCHER = observer


def load_config(
    mode: str | None = None,
    config_file: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
    event_bus: "EventBus" | None = None,
    watch: bool = False,
) -> Config:
    """Load the configuration for the given *mode*.

    Parameters
    ----------
    mode:
        Optional profile name such as ``"dev"`` or ``"prod"``. When ``None`` the
        value is read from the ``MENACE_MODE`` environment variable or falls back to
        ``"dev"``.
    config_file:
        Optional path to an additional configuration file that will be merged
        after the profile-specific settings.
    overrides:
        Mapping of configuration values to override using dotted keys.
    event_bus:
        Optional event bus for broadcasting configuration reload events.
    watch:
        When ``True`` a watchdog observer is started to monitor the
        configuration directory for changes.
    """

    if event_bus is not None:
        set_event_bus(event_bus)

    if UnifiedConfigStore is not None:  # load .env and vault secrets first
        global _CONFIG_STORE
        try:
            _CONFIG_STORE = UnifiedConfigStore()
            _CONFIG_STORE.load()
        except Exception:  # pragma: no cover - best effort only
            logger.exception("failed loading unified config store")

    active_mode = mode or os.getenv("MENACE_MODE", "dev")

    data = _load_yaml(DEFAULT_SETTINGS_FILE)

    profile_file = CONFIG_DIR / f"{active_mode}.yaml"
    if profile_file.exists():
        data = _merge_dict(data, _load_yaml(profile_file))
    else:
        raise FileNotFoundError(
            f"Config profile '{active_mode}' not found at {profile_file}"
        )

    if STACK_CONTEXT_FILE.exists():
        data = _merge_dict(data, _load_yaml(STACK_CONTEXT_FILE))

    stack_threshold_overrides = _stack_context_overrides()
    if stack_threshold_overrides:
        data = _merge_dict(data, stack_threshold_overrides)

    if config_file:
        data = _merge_dict(data, _load_yaml(Path(config_file)))

    # Environment variable overrides
    env_overrides: Dict[str, Any] = {}
    openai_env = os.getenv("OPENAI_API_KEY")
    serp_env = os.getenv("SERP_API_KEY")
    if openai_env or serp_env:
        env_overrides["api_keys"] = {}
        if openai_env:
            env_overrides["api_keys"]["openai"] = openai_env
        if serp_env:
            env_overrides["api_keys"]["serp"] = serp_env
        data = _merge_dict(data, env_overrides)

    stack_env_overrides: Dict[str, Any] = {}
    streaming_override = _env_flag("STACK_STREAMING")
    if streaming_override is not None:
        stack_env_overrides["enabled"] = streaming_override
    for env_name, attr in (
        ("STACK_INDEX_PATH", "index_path"),
        ("STACK_METADATA_PATH", "metadata_path"),
        ("STACK_CACHE_DIR", "cache_dir"),
        ("STACK_PROGRESS_PATH", "progress_path"),
    ):
        value = _stack_env_text(env_name)
        if value:
            stack_env_overrides[attr] = value

    if stack_env_overrides:
        stack_dataset_cfg = data.setdefault("stack_dataset", {})
        if not isinstance(stack_dataset_cfg, dict):
            stack_dataset_cfg = {}
            data["stack_dataset"] = stack_dataset_cfg
        stack_dataset_cfg.update(stack_env_overrides)

        context_builder_cfg = data.setdefault("context_builder", {})
        if not isinstance(context_builder_cfg, dict):
            context_builder_cfg = {}
            data["context_builder"] = context_builder_cfg
        stack_section = context_builder_cfg.setdefault("stack", {})
        if not isinstance(stack_section, dict):
            stack_section = {}
            context_builder_cfg["stack"] = stack_section
        stack_section.update(stack_env_overrides)

        cb_field_map = {
            "enabled": "stack_enabled",
            "index_path": "stack_index_path",
            "metadata_path": "stack_metadata_path",
            "cache_dir": "stack_cache_dir",
            "progress_path": "stack_progress_path",
        }
        for source_key, target_key in cb_field_map.items():
            if source_key in stack_env_overrides:
                context_builder_cfg[target_key] = stack_env_overrides[source_key]

    cfg = Config.model_validate(data)
    if overrides:
        cfg = cfg.apply_overrides(overrides)

    if watch and Observer is not None:
        _start_watcher()

    return cfg


def get_config(
    event_bus: "EventBus" | None = None, *, watch: bool = False
) -> Config:
    """Return the canonical :class:`Config` instance.

    The configuration is loaded lazily on first access so that callers may set
    ``_MODE``, ``_CONFIG_PATH`` or ``_OVERRIDES`` prior to loading. Subsequent
    calls return the same object. When *event_bus* is provided it is stored for
    future reload notifications. When ``watch`` is ``True`` a watcher is started
    to monitor the configuration directory for changes.
    """

    if event_bus is not None:
        set_event_bus(event_bus)

    global CONFIG
    if CONFIG is None:
        CONFIG = load_config(_MODE, _CONFIG_PATH, _OVERRIDES, watch=watch)
    elif watch and _WATCHER is None:
        _start_watcher()
    return CONFIG


def set_event_bus(bus: "EventBus" | None) -> None:
    """Configure the event bus used for reload notifications."""

    global _EVENT_BUS
    _EVENT_BUS = bus


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _build_overrides(pairs: list[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}', expected key=value")
        key, value = pair.split("=", 1)
        value_data = yaml.safe_load(value)
        current = result
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value_data
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load configuration")
    parser.add_argument("--mode", help="Configuration mode (e.g. dev or prod)")
    parser.add_argument(
        "--config", help="Additional configuration YAML file to merge", dest="config_file"
    )
    parser.add_argument(
        "--config-override",
        action="append",
        dest="config_override",
        metavar="KEY=VALUE",
        help="Override configuration values (can be specified multiple times)",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable config file watcher",
    )
    return parser.parse_args(argv)


def reload(event_bus: "EventBus" | None = None) -> Config:
    """Reload configuration from disk using stored parameters."""
    global CONFIG

    if event_bus is not None:
        set_event_bus(event_bus)

    old_data = CONFIG.model_dump() if CONFIG else None
    CONFIG = load_config(_MODE, _CONFIG_PATH, _OVERRIDES)
    new_data = CONFIG.model_dump()
    if _EVENT_BUS is not None:
        try:
            payload = {"config": new_data, "diff": _dict_diff(old_data, new_data)}
            _EVENT_BUS.publish("config.reload", payload)
        except Exception:  # pragma: no cover - best effort only
            logger.exception("failed publishing config reload event")
    return CONFIG


def shutdown() -> None:
    """Stop the configuration watcher if it is running."""

    global _WATCHER
    if _WATCHER is None:
        return
    try:  # pragma: no cover - best effort only
        _WATCHER.stop()
        _WATCHER.join()
    finally:
        _WATCHER = None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    global _MODE, _CONFIG_PATH, _OVERRIDES
    _MODE = args.mode
    _CONFIG_PATH = Path(args.config_file) if args.config_file else None
    _OVERRIDES = _build_overrides(args.config_override or [])
    cfg = get_config(watch=not args.no_watch)
    print(cfg.model_dump())


if __name__ == "__main__":  # pragma: no cover
    main()
