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
from typing import Any, Dict, TYPE_CHECKING, Set, List
from types import SimpleNamespace
import sys
import site
import copy

from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from dynamic_path_router import get_project_root
from stack_dataset_defaults import (
    STACK_LANGUAGE_ALLOWLIST,
    normalise_stack_languages,
)

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


def _model_dump_any(value: Any) -> Dict[str, Any]:
    """Return a dictionary representation for Pydantic models and dataclasses."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                data = method()  # type: ignore[misc]
            except TypeError:
                data = method(exclude_none=False)  # type: ignore[misc]
            return dict(data)
    return dict(getattr(value, "__dict__", {}))


def _model_field_names(model: Any) -> Set[str]:
    """Return the declared field names for a Pydantic model class."""

    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields.keys())
    legacy = getattr(model, "__fields__", None)
    if legacy is not None:
        return set(legacy.keys())
    return set()


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


class StackTokenConfig(_StrictBaseModel):
    """Hugging Face authentication references for Stack ingestion."""

    env_vars: List[str] = Field(
        default_factory=lambda: [
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        ]
    )
    required: bool = Field(
        default=False,
        description="Whether ingestion should fail when no token is exported",
    )

    @field_validator("env_vars", mode="before")
    @classmethod
    def _coerce_env_vars(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(part).strip() for part in value if str(part).strip()]

    @field_validator("env_vars")
    @classmethod
    def _validate_env_vars(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("env_vars must include at least one environment variable")
        return value


class StackCacheConfig(_StrictBaseModel):
    """Cache and index locations used by Stack ingestion and retrieval."""

    data_dir: str = Field(default="~/.cache/menace/stack")
    index_path: str = Field(
        default="vector_service/stack_vectors",
        description="Path to the Stack vector index",
    )
    metadata_path: str = Field(
        default="vector_service/stack_metadata.db",
        description="Path to the Stack metadata catalogue",
    )
    document_cache: str = Field(
        default="chunk_summary_cache/stack_documents",
        description="Local cache for fetched Stack documents",
    )

    @field_validator("data_dir", "index_path", "metadata_path", "document_cache")
    @classmethod
    def _not_blank(cls, value: str, info) -> str:  # type: ignore[override]
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class StackIngestionConfig(_StrictBaseModel):
    """Filters and limits applied when ingesting Stack documents."""

    streaming: bool = False
    languages: List[str] = Field(default_factory=list)
    max_document_lines: int = Field(default=200, gt=0)
    chunk_overlap: int = Field(default=20, ge=0)
    batch_size: int | None = Field(default=None, gt=0)

    @field_validator("languages", mode="before")
    @classmethod
    def _coerce_languages(cls, value: Any) -> List[str]:
        return normalise_stack_languages(value)

    @field_validator("languages")
    @classmethod
    def _validate_languages(cls, value: List[str]) -> List[str]:
        unknown = [lang for lang in value if lang not in STACK_LANGUAGE_ALLOWLIST]
        if unknown:
            allowed = ", ".join(sorted(STACK_LANGUAGE_ALLOWLIST))
            raise ValueError(
                f"Unsupported Stack dataset languages: {', '.join(sorted(set(unknown)))}. "
                f"Allowed values are: {allowed}"
            )
        return value


class StackRetrievalConfig(_StrictBaseModel):
    """Retrieval parameters and context budgets for Stack results."""

    top_k: int = Field(default=50, gt=0)
    weight: float = Field(default=1.0, ge=0.0)
    max_context_documents: int = Field(
        default=8,
        gt=0,
        description="Maximum Stack documents merged into a single context window",
    )
    max_context_lines: int = Field(
        default=400,
        gt=0,
        description="Total line budget for Stack snippets included in prompts",
    )


class StackDatasetConfig(_StrictBaseModel):
    """Configuration for Stack dataset ingestion and retrieval."""

    enabled: bool = False
    dataset_name: str = "bigcode/the-stack-v2-dedup"
    split: str = "train"
    ingestion: StackIngestionConfig = Field(default_factory=StackIngestionConfig)
    retrieval: StackRetrievalConfig = Field(default_factory=StackRetrievalConfig)
    cache: StackCacheConfig = Field(default_factory=StackCacheConfig)
    tokens: StackTokenConfig = Field(default_factory=StackTokenConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        data = dict(values)
        ingestion = dict(data.get("ingestion") or {})
        retrieval = dict(data.get("retrieval") or {})
        cache = dict(data.get("cache") or {})
        tokens = dict(data.get("tokens") or {})

        if "languages" in data and "languages" not in ingestion:
            ingestion["languages"] = data.pop("languages")
        if "max_lines" in data and "max_document_lines" not in ingestion:
            ingestion["max_document_lines"] = data.pop("max_lines")
        if "chunk_overlap" in data and "chunk_overlap" not in ingestion:
            ingestion["chunk_overlap"] = data.pop("chunk_overlap")
        if "streaming" in data and "streaming" not in ingestion:
            ingestion["streaming"] = data.pop("streaming")
        if "batch_size" in data and "batch_size" not in ingestion:
            ingestion["batch_size"] = data.pop("batch_size")

        if "top_k" in data and "top_k" not in retrieval:
            retrieval["top_k"] = data.pop("top_k")
        if "weight" in data and "weight" not in retrieval:
            retrieval["weight"] = data.pop("weight")
        if "max_context_documents" in data and "max_context_documents" not in retrieval:
            retrieval["max_context_documents"] = data.pop("max_context_documents")
        if "max_context_lines" in data and "max_context_lines" not in retrieval:
            retrieval["max_context_lines"] = data.pop("max_context_lines")

        if "data_dir" in data and "data_dir" not in cache:
            cache["data_dir"] = data.pop("data_dir")
        if "cache_dir" in data and "data_dir" not in cache:
            cache["data_dir"] = data.pop("cache_dir")
        if "index_path" in data and "index_path" not in cache:
            cache["index_path"] = data.pop("index_path")
        if "metadata_path" in data and "metadata_path" not in cache:
            cache["metadata_path"] = data.pop("metadata_path")
        if "document_cache" in data and "document_cache" not in cache:
            cache["document_cache"] = data.pop("document_cache")

        token_keys = []
        if "token_env" in data:
            token_keys.append(data.pop("token_env"))
        if "token_env_vars" in data:
            token_keys.extend(data.pop("token_env_vars"))
        if "huggingface_token" in data:
            token_keys.append(data.pop("huggingface_token"))
        if token_keys and "env_vars" not in tokens:
            tokens["env_vars"] = token_keys

        data["ingestion"] = ingestion
        data["retrieval"] = retrieval
        data["cache"] = cache
        data["tokens"] = tokens
        return data

    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> "StackDatasetConfig":
        if self.ingestion.chunk_overlap >= self.ingestion.max_document_lines:
            raise ValueError("chunk_overlap must be smaller than max_document_lines")
        return self


class StackContextConfig(StackDatasetConfig):
    """Extended Stack configuration exposed directly to context builders."""

    summary_tokens: int = Field(default=160, ge=0)
    text_max_tokens: int = Field(default=320, ge=0)
    penalty: float = Field(default=0.0)
    ingestion_enabled: bool = Field(default=True)
    ingestion_throttle_seconds: float = Field(default=600.0, ge=0.0)
    ingestion_batch_limit: int | None = Field(default=None, ge=1)
    ensure_before_search: bool = True

    @property
    def languages(self) -> List[str]:
        return list(self.ingestion.languages)

    @property
    def max_lines(self) -> int:
        return self.ingestion.max_document_lines

    @property
    def chunk_overlap(self) -> int:
        return self.ingestion.chunk_overlap

    @property
    def streaming(self) -> bool:
        return self.ingestion.streaming

    @property
    def batch_size(self) -> int | None:
        return self.ingestion.batch_size

    @property
    def top_k(self) -> int:
        return self.retrieval.top_k

    @property
    def weight(self) -> float:
        return self.retrieval.weight

    @property
    def cache_dir(self) -> str:
        return self.cache.data_dir

    @property
    def index_path(self) -> str:
        return self.cache.index_path

    @property
    def metadata_path(self) -> str:
        return self.cache.metadata_path


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
    stack: StackContextConfig | None = None
    stack_dataset: StackDatasetConfig | None = None

    @model_validator(mode="after")
    def _sync_stack_configs(self) -> "ContextBuilderConfig":
        stack_data = _model_dump_any(self.stack)
        dataset_data = _model_dump_any(self.stack_dataset)
        if not stack_data and not dataset_data:
            return self

        dataset_fields = _model_field_names(StackDatasetConfig)
        merged_dataset = dict(dataset_data)
        for key in dataset_fields:
            if key not in merged_dataset or merged_dataset[key] is None:
                if key in stack_data:
                    merged_dataset[key] = stack_data[key]

        combined_stack = dict(stack_data)
        for key in dataset_fields:
            if key in merged_dataset:
                combined_stack[key] = merged_dataset[key]

        self.stack_dataset = StackDatasetConfig.model_validate(merged_dataset)
        self.stack = StackContextConfig.model_validate(combined_stack)
        return self

    @property
    def stack_top_k(self) -> int:
        """Return the configured Stack retrieval depth.

        Defaults to ``0`` when Stack integration is disabled or misconfigured.
        """

        if self.stack is not None:
            try:
                return int(self.stack.top_k)
            except Exception:
                pass
        if self.stack_dataset is not None:
            try:
                return int(self.stack_dataset.retrieval.top_k)
            except Exception:
                pass
        return 0

    @property
    def stack_weight(self) -> float:
        """Return weighting multiplier applied to Stack results."""

        if self.stack is not None:
            try:
                return float(self.stack.weight)
            except Exception:
                pass
        if self.stack_dataset is not None:
            try:
                return float(self.stack_dataset.retrieval.weight)
            except Exception:
                pass
        return 1.0

    @property
    def stack_penalty(self) -> float:
        """Return penalty offset applied to Stack scores."""

        if self.stack is not None:
            try:
                return float(self.stack.penalty)
            except Exception:
                pass
        return 0.0


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


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str, *, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field} must be an integer") from exc


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

    stack_defaults = CONFIG_DIR / "stack_context.yaml"
    if stack_defaults.exists():
        data = _merge_dict(data, _load_yaml(stack_defaults))
    else:
        legacy_stack = CONFIG_DIR / "stack_retrieval.yaml"
        if legacy_stack.exists():
            data = _merge_dict(data, _load_yaml(legacy_stack))

    profile_file = CONFIG_DIR / f"{active_mode}.yaml"
    if profile_file.exists():
        data = _merge_dict(data, _load_yaml(profile_file))
    else:
        raise FileNotFoundError(
            f"Config profile '{active_mode}' not found at {profile_file}"
        )

    if config_file:
        data = _merge_dict(data, _load_yaml(Path(config_file)))

    # Environment variable overrides
    env_overrides: Dict[str, Any] = {}
    openai_env = os.getenv("OPENAI_API_KEY")
    serp_env = os.getenv("SERP_API_KEY")
    if openai_env or serp_env:
        env_overrides.setdefault("api_keys", {})
        if openai_env:
            env_overrides["api_keys"]["openai"] = openai_env
        if serp_env:
            env_overrides["api_keys"]["serp"] = serp_env

    stack_override: Dict[str, Any] = {}

    def _stack_set(path: list[str], value: Any) -> None:
        current = stack_override
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value

    stack_enabled = os.getenv("STACK_DATA_ENABLED")
    if stack_enabled is not None:
        _stack_set(["enabled"], _parse_bool(stack_enabled))
    stack_dataset = os.getenv("STACK_DATASET")
    if stack_dataset:
        _stack_set(["dataset_name"], stack_dataset)
    stack_split = os.getenv("STACK_SPLIT")
    if stack_split:
        _stack_set(["split"], stack_split)
    stack_languages = os.getenv("STACK_LANGUAGES")
    if stack_languages is not None:
        _stack_set(["ingestion", "languages"], normalise_stack_languages(stack_languages))
    stack_streaming = os.getenv("STACK_STREAMING")
    if stack_streaming is not None:
        _stack_set(["ingestion", "streaming"], _parse_bool(stack_streaming))
    stack_max_lines = os.getenv("STACK_MAX_LINES")
    if stack_max_lines is not None:
        _stack_set(
            ["ingestion", "max_document_lines"],
            _parse_int(stack_max_lines, field="STACK_MAX_LINES"),
        )
    stack_chunk_overlap = os.getenv("STACK_CHUNK_OVERLAP")
    if stack_chunk_overlap is not None:
        _stack_set(
            ["ingestion", "chunk_overlap"],
            _parse_int(stack_chunk_overlap, field="STACK_CHUNK_OVERLAP"),
        )
    stack_batch_size = os.getenv("STACK_BATCH_SIZE")
    if stack_batch_size is not None:
        _stack_set(
            ["ingestion", "batch_size"],
            _parse_int(stack_batch_size, field="STACK_BATCH_SIZE"),
        )
    stack_top_k = os.getenv("STACK_TOP_K")
    if stack_top_k is not None:
        _stack_set(["retrieval", "top_k"], _parse_int(stack_top_k, field="STACK_TOP_K"))
    stack_weight = os.getenv("STACK_WEIGHT")
    if stack_weight is not None:
        try:
            _stack_set(["retrieval", "weight"], float(stack_weight))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("STACK_WEIGHT must be a float") from exc
    stack_context_docs = os.getenv("STACK_CONTEXT_DOCS")
    if stack_context_docs is not None:
        _stack_set(
            ["retrieval", "max_context_documents"],
            _parse_int(stack_context_docs, field="STACK_CONTEXT_DOCS"),
        )
    stack_context_lines = os.getenv("STACK_CONTEXT_LINES")
    if stack_context_lines is not None:
        _stack_set(
            ["retrieval", "max_context_lines"],
            _parse_int(stack_context_lines, field="STACK_CONTEXT_LINES"),
        )
    stack_data_dir = os.getenv("STACK_DATA_DIR") or os.getenv("STACK_CACHE_DIR")
    if stack_data_dir:
        _stack_set(["cache", "data_dir"], stack_data_dir)
    stack_index = (
        os.getenv("STACK_DATA_INDEX")
        or os.getenv("STACK_VECTOR_PATH")
        or os.getenv("STACK_INDEX_PATH")
    )
    if stack_index:
        _stack_set(["cache", "index_path"], stack_index)
    stack_metadata = os.getenv("STACK_METADATA_PATH") or os.getenv("STACK_METADATA_DB")
    if stack_metadata:
        _stack_set(["cache", "metadata_path"], stack_metadata)
    stack_doc_cache = os.getenv("STACK_DOCUMENT_CACHE")
    if stack_doc_cache:
        _stack_set(["cache", "document_cache"], stack_doc_cache)
    if stack_override:
        context_section = env_overrides.setdefault("context_builder", {})
        context_section["stack_dataset"] = copy.deepcopy(stack_override)
        context_section["stack"] = copy.deepcopy(stack_override)

    if env_overrides:
        data = _merge_dict(data, env_overrides)

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
