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
from typing import Any, Dict, TYPE_CHECKING, Set
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
    """Configuration for Stack dataset ingestion and retrieval."""

    enabled: bool = False
    allowed_languages: Set[str] = Field(
        default_factory=lambda: {"python"},
        description="Whitelisted programming languages from The Stack dataset",
    )
    max_lines_per_document: int = Field(
        800,
        ge=0,
        description="Maximum number of lines retained per Stack document",
    )
    chunk_size: int = Field(
        2048,
        ge=1,
        description="Character chunk size used when embedding Stack documents",
    )
    retrieval_top_k: int = Field(
        5,
        ge=0,
        description="Maximum Stack records to retrieve per query",
    )

    @field_validator("allowed_languages")
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
