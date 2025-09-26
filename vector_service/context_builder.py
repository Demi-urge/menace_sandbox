"""Cross-database context builder used by language model prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
# ``ParsedFailure`` previously provided structured failure info.  The new parser
# returns dictionaries so we reference the type indirectly to avoid tight
# coupling.
import logging
import asyncio
import uuid
import time
import threading
import os

from dynamic_path_router import resolve_path

from filelock import FileLock

from redaction_utils import redact_text
from snippet_compressor import compress_snippets
from context_builder import handle_failure, PromptBuildError

from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError
from .retriever import Retriever, PatchRetriever, StackRetriever, FallbackResult
from .vector_store import get_stack_vector_store, get_stack_metadata_path
from config import ContextBuilderConfig, StackDatasetConfig, get_config
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from .patch_logger import _VECTOR_RISK  # type: ignore
from patch_safety import PatchSafety
from .ranking_utils import rank_patches
from .embedding_backfill import (
    ensure_embeddings_fresh,
    StaleEmbeddingsError,
    EmbeddingBackfill,
    schedule_backfill,
)
from prompt_types import Prompt

try:  # pragma: no cover - optional precise tokenizer
    import tiktoken

    _FALLBACK_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - dependency missing or failed
    tiktoken = None  # type: ignore
    _FALLBACK_ENCODER = None

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())

try:  # pragma: no cover - optional dependency
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

_VEC_METRICS = VectorMetricsDB() if VectorMetricsDB is not None else None

# Alias retained for backward compatibility with tests expecting
# ``UniversalRetriever`` to be injectable.
UniversalRetriever = Retriever

logger = logging.getLogger(__name__)

_HF_ENV_KEYS = (
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
)

_STACK_WARNED_FLAGS: set[str] = set()

# ---------------------------------------------------------------------------
# Stack dataset configuration helpers
# ---------------------------------------------------------------------------


def _stack_dataset_config() -> StackDatasetConfig:
    """Return the active Stack dataset configuration."""

    try:
        cfg = get_config()
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("failed to access stack dataset configuration")
        return StackDatasetConfig()

    stack_cfg: StackDatasetConfig | None = None
    try:
        context_cfg = getattr(cfg, "context_builder", None)
        if context_cfg is not None:
            stack_cfg = getattr(context_cfg, "stack", None)
    except Exception:
        stack_cfg = None

    if stack_cfg is None:
        stack_cfg = getattr(cfg, "stack_dataset", None)

    if isinstance(stack_cfg, StackDatasetConfig):
        return stack_cfg
    try:
        return StackDatasetConfig.model_validate(stack_cfg)
    except Exception:
        return StackDatasetConfig()


# ---------------------------------------------------------------------------
# Failed tag tracking
# ---------------------------------------------------------------------------
_FAILED_TAG_CACHE: set[str] = set()
_FAILED_TAG_FILE = resolve_path("vector_service") / "failed_tags.json"
_FAILED_TAG_LOCK = threading.Lock()
_FAILED_TAG_FILE_LOCK = FileLock(str(_FAILED_TAG_FILE) + ".lock")


def load_failed_tags() -> set[str]:
    """Load persisted failed tags into the cache."""

    with _FAILED_TAG_LOCK, _FAILED_TAG_FILE_LOCK:
        try:
            if _FAILED_TAG_FILE.exists():
                data = json.loads(_FAILED_TAG_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _FAILED_TAG_CACHE.update(
                        str(t) for t in data if isinstance(t, str) and t
                    )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to load failed tags")
    return set(_FAILED_TAG_CACHE)


def record_failed_tags(tags: List[str]) -> None:
    """Record strategy or ROI tags that led to failures.

    The tags are stored in a module-level cache so subsequent context builds
    can automatically exclude vectors associated with past failures.  Tags are
    also persisted to disk so failures are remembered across runs.
    """

    valid = [tag for tag in tags if isinstance(tag, str) and tag]
    if not valid:
        return

    with _FAILED_TAG_LOCK, _FAILED_TAG_FILE_LOCK:
        try:
            if _FAILED_TAG_FILE.exists():
                data = json.loads(_FAILED_TAG_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _FAILED_TAG_CACHE.update(
                        str(t) for t in data if isinstance(t, str) and t
                    )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to read failed tag store")

        _FAILED_TAG_CACHE.update(valid)
        try:
            _FAILED_TAG_FILE.write_text(
                json.dumps(sorted(_FAILED_TAG_CACHE)), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to write failed tag store")


def _get_failed_tags() -> set[str]:
    """Return a copy of the cached failed tags.

    Exposed for testability; external modules should rely on
    :func:`record_failed_tags` to mutate the cache.
    """

    return set(_FAILED_TAG_CACHE)


try:  # pragma: no cover - best effort
    load_failed_tags()
except Exception:
    logger.exception("load_failed_tags failed")


def _ensure_vector_service() -> None:
    """Ensure the configured vector service is reachable."""

    base = os.environ.get("VECTOR_SERVICE_URL")
    if not base:
        return

    import urllib.request
    import subprocess
    import sys

    # Configuration hooks ---------------------------------------------------
    ready_tries = int(os.environ.get("VECTOR_SERVICE_RETRY_COUNT", "5"))
    ready_delay = float(os.environ.get("VECTOR_SERVICE_RETRY_DELAY", "0.5"))
    ready_backoff = float(os.environ.get("VECTOR_SERVICE_RETRY_BACKOFF", "2.0"))
    start_tries = int(os.environ.get("VECTOR_SERVICE_START_RETRIES", "3"))
    start_delay = float(os.environ.get("VECTOR_SERVICE_START_DELAY", "1.0"))
    start_backoff = float(os.environ.get("VECTOR_SERVICE_START_BACKOFF", "2.0"))
    verify_embeddings = (
        os.environ.get("VECTOR_SERVICE_VERIFY_EMBEDDINGS", "").lower()
        in {"1", "true", "yes"}
    )

    def _ready() -> bool:
        url = f"{base.rstrip('/')}/health/ready"
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception as exc:
            logger.debug("vector service health check failed: %s", exc)
            return False

    def _wait_ready(tries: int, delay: float, backoff: float) -> bool:
        wait = delay
        for _ in range(tries):
            if _ready():
                return True
            time.sleep(wait)
            wait *= backoff
        return False

    def _verify_embedding_endpoint() -> None:
        if not verify_embeddings:
            return
        url = f"{base.rstrip('/')}/search"
        payload = json.dumps({"query": "ping"}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5):
                return
        except Exception as exc:
            logger.error("embedding endpoint check failed: %s", exc)
            raise VectorServiceError(
                f"embedding endpoint unavailable at {url}") from exc

    if _wait_ready(ready_tries, ready_delay, ready_backoff):
        _verify_embedding_endpoint()
        return

    script = resolve_path("scripts/run_vector_service.py")
    last_error: Exception | None = None
    wait = start_delay
    for attempt in range(1, start_tries + 1):
        logger.info(
            "starting vector service attempt %s/%s", attempt, start_tries
        )
        try:
            subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # pragma: no cover - best effort
            last_error = exc
            logger.error("failed to launch vector service: %s", exc)
        if _wait_ready(ready_tries, ready_delay, ready_backoff):
            _verify_embedding_endpoint()
            return
        logger.error(
            "vector service not ready after attempt %s/%s", attempt, start_tries
        )
        time.sleep(wait)
        wait *= start_backoff

    message = f"vector service unavailable at {base} after {start_tries} attempts"
    logger.error(message)
    if last_error is not None:
        raise VectorServiceError(message) from last_error
    raise VectorServiceError(message)


try:  # pragma: no cover - optional dependency
    from . import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when undefined
    class ErrorResult(Exception):
        """Fallback ErrorResult used when vector service lacks explicit class."""

        pass

try:  # pragma: no cover - heavy dependency
    from menace_memory_manager import MenaceMemoryManager
except Exception:  # pragma: no cover
    MenaceMemoryManager = None  # type: ignore

# Optional patch history ----------------------------------------------------
try:  # pragma: no cover - optional dependency
    from code_database import PatchHistoryDB  # type: ignore
except Exception:  # pragma: no cover
    PatchHistoryDB = None  # type: ignore


@dataclass
class _ScoredEntry:
    entry: Dict[str, Any]
    score: float
    origin: str
    vector_id: str
    metadata: Dict[str, Any]


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        patch_retriever: PatchRetriever | None = None,
        stack_retriever: Any | None = None,
        ranking_model: Any | None = None,
        roi_tracker: Any | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        summariser: Callable[[str], str] | None = None,
        db_weights: Dict[str, float] | None = None,
        ranking_weight: float | None = None,
        roi_weight: float | None = None,
        recency_weight: float | None = None,
        safety_weight: float | None = None,
        max_tokens: int | None = None,
        regret_penalty: float | None = None,
        alignment_penalty: float | None = None,
        alert_penalty: float | None = None,
        risk_penalty: float | None = None,
        roi_tag_penalties: Dict[str, float] | None = None,
        enhancement_weight: float | None = None,
        max_alignment_severity: float | None = None,
        max_alerts: int | None = None,
        license_denylist: set[str] | None = None,
        precise_token_count: bool | None = None,
        max_diff_lines: int | None = None,
        patch_safety: PatchSafety | None = None,
        similarity_metric: str | None = None,
        embedding_check_interval: float | None = None,
        prompt_score_weight: float | None = None,
        prompt_max_tokens: int | None = None,
        config: ContextBuilderConfig | None = None,
        stack_enabled: bool | None = None,
        stack_languages: Iterable[str] | None = None,
        stack_top_k: int | None = None,
        stack_max_lines: int | None = None,
        stack_index_path: str | None = None,
        stack_metadata_path: str | None = None,
        stack_max_bytes: int | None = None,
        stack_cache_dir: str | None = None,
        stack_progress_path: str | None = None,
        stack_prompt_enabled: bool | None = None,
        stack_prompt_limit: int | None = None,
        stack_config: StackDatasetConfig | None = None,
    ) -> None:
        defaults = ContextBuilderConfig()
        global_stack_cfg = _stack_dataset_config()
        try:
            cfg_source = config or getattr(get_config(), "context_builder", defaults)
        except Exception:
            cfg_source = config or defaults
        cfg = cfg_source or defaults

        stack_defaults = getattr(defaults, "stack", StackDatasetConfig())
        cfg_stack = getattr(cfg, "stack", None)
        if cfg_stack is not None and not isinstance(cfg_stack, StackDatasetConfig):
            try:
                cfg_stack = StackDatasetConfig.model_validate(cfg_stack)
            except Exception:
                cfg_stack = None
        if not isinstance(global_stack_cfg, StackDatasetConfig):
            try:
                global_stack_cfg = StackDatasetConfig.model_validate(global_stack_cfg)
            except Exception:
                global_stack_cfg = stack_defaults
        dataset_cfg = stack_config or cfg_stack or global_stack_cfg or stack_defaults

        def _cfg(name: str, fallback: Any) -> Any:
            try:
                return getattr(cfg, name)
            except Exception:
                return fallback

        def _coerce_float(value: Any, fallback: float) -> float:
            try:
                return float(value)
            except Exception:
                return float(fallback)

        def _coerce_int(value: Any, fallback: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(fallback)

        def _coerce_bool(value: Any, fallback: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            try:
                return bool(value)
            except Exception:
                return bool(fallback)

        def _resolve_optional_path(value: Any) -> Path | None:
            if value in {None, "", b""}:
                return None
            try:
                candidate = Path(str(value))
            except Exception:
                return None
            try:
                return resolve_path(candidate)
            except Exception:
                try:
                    return candidate.expanduser().resolve()
                except Exception:
                    return candidate

        ranking_weight = _coerce_float(
            ranking_weight if ranking_weight is not None else _cfg("ranking_weight", defaults.ranking_weight),
            defaults.ranking_weight,
        )
        roi_weight = _coerce_float(
            roi_weight if roi_weight is not None else _cfg("roi_weight", defaults.roi_weight),
            defaults.roi_weight,
        )
        recency_weight = _coerce_float(
            recency_weight if recency_weight is not None else _cfg("recency_weight", defaults.recency_weight),
            defaults.recency_weight,
        )
        safety_weight = _coerce_float(
            safety_weight if safety_weight is not None else _cfg("safety_weight", defaults.safety_weight),
            defaults.safety_weight,
        )
        max_tokens = _coerce_int(
            max_tokens if max_tokens is not None else _cfg("max_tokens", defaults.max_tokens),
            defaults.max_tokens,
        )
        regret_penalty = _coerce_float(
            regret_penalty if regret_penalty is not None else _cfg("regret_penalty", defaults.regret_penalty),
            defaults.regret_penalty,
        )
        alignment_penalty = _coerce_float(
            alignment_penalty if alignment_penalty is not None else _cfg("alignment_penalty", defaults.alignment_penalty),
            defaults.alignment_penalty,
        )
        alert_penalty = _coerce_float(
            alert_penalty if alert_penalty is not None else _cfg("alert_penalty", defaults.alert_penalty),
            defaults.alert_penalty,
        )
        risk_penalty = _coerce_float(
            risk_penalty if risk_penalty is not None else _cfg("risk_penalty", defaults.risk_penalty),
            defaults.risk_penalty,
        )
        enhancement_weight = _coerce_float(
            enhancement_weight if enhancement_weight is not None else _cfg("enhancement_weight", defaults.enhancement_weight),
            defaults.enhancement_weight,
        )
        prompt_score_weight = _coerce_float(
            prompt_score_weight if prompt_score_weight is not None else _cfg("prompt_score_weight", defaults.prompt_score_weight),
            defaults.prompt_score_weight,
        )
        prompt_max_tokens = _coerce_int(
            prompt_max_tokens if prompt_max_tokens is not None else _cfg("prompt_max_tokens", defaults.prompt_max_tokens),
            defaults.prompt_max_tokens,
        )
        max_alignment_severity = _coerce_float(
            max_alignment_severity if max_alignment_severity is not None else _cfg("max_alignment_severity", defaults.max_alignment_severity),
            defaults.max_alignment_severity,
        )
        max_alerts = _coerce_int(
            max_alerts if max_alerts is not None else _cfg("max_alerts", defaults.max_alerts),
            defaults.max_alerts,
        )
        max_diff_lines = _coerce_int(
            max_diff_lines if max_diff_lines is not None else _cfg("max_diff_lines", defaults.max_diff_lines),
            defaults.max_diff_lines,
        )
        embedding_check_interval = _coerce_float(
            embedding_check_interval
            if embedding_check_interval is not None
            else _cfg("embedding_check_interval", defaults.embedding_check_interval),
            defaults.embedding_check_interval,
        )
        similarity_metric = (
            similarity_metric
            if similarity_metric is not None
            else str(_cfg("similarity_metric", defaults.similarity_metric))
        ) or "cosine"
        precise_token_count = _coerce_bool(
            precise_token_count
            if precise_token_count is not None
            else _cfg("precise_token_count", defaults.precise_token_count),
            defaults.precise_token_count,
        )

        if roi_tag_penalties is None:
            try:
                roi_tag_penalties = dict(_cfg("roi_tag_penalties", defaults.roi_tag_penalties))
            except Exception:
                roi_tag_penalties = dict(defaults.roi_tag_penalties)
        else:
            roi_tag_penalties = dict(roi_tag_penalties)

        if db_weights is None:
            try:
                db_weights = dict(_cfg("db_weights", defaults.db_weights))
            except Exception:
                db_weights = dict(defaults.db_weights)
        else:
            db_weights = dict(db_weights)

        if license_denylist is None:
            raw_licenses = _cfg("license_denylist", defaults.license_denylist)
        else:
            raw_licenses = license_denylist
        license_denylist = {
            str(lic).strip()
            for lic in (raw_licenses or set())
            if isinstance(lic, str) and lic.strip()
        }

        dataset_enabled = bool(getattr(dataset_cfg, "enabled", False)) or bool(
            getattr(global_stack_cfg, "enabled", False)
        )

        enabled_candidate = stack_enabled
        if enabled_candidate is None:
            enabled_candidate = _first_non_none(
                getattr(cfg, "stack_enabled", None),
                getattr(cfg_stack, "enabled", None),
                getattr(global_stack_cfg, "enabled", None),
                getattr(stack_defaults, "enabled", None),
            )
        enabled_value = enabled_candidate if enabled_candidate is not None else False
        stack_enabled_bool = _coerce_bool(enabled_value, bool(enabled_value))
        if not stack_enabled_bool and dataset_enabled:
            stack_enabled_bool = True

        language_source = _first_non_none(
            stack_languages,
            getattr(cfg, "stack_languages", None),
            getattr(cfg_stack, "languages", None),
            getattr(dataset_cfg, "languages", None),
            getattr(global_stack_cfg, "languages", None),
            getattr(stack_defaults, "languages", None),
        )
        lang_set = {
            str(language).strip().lower()
            for language in (language_source or [])
            if isinstance(language, str) and language.strip()
        }

        stack_top_k_val = _first_non_none(
            stack_top_k,
            getattr(cfg, "stack_top_k", None),
            getattr(cfg_stack, "retrieval_top_k", None),
            getattr(dataset_cfg, "retrieval_top_k", None),
            getattr(global_stack_cfg, "retrieval_top_k", None),
            getattr(stack_defaults, "retrieval_top_k", None),
        )
        top_k_fallback = getattr(stack_defaults, "retrieval_top_k", 0)
        stack_top_k_int = _coerce_int(
            stack_top_k_val if stack_top_k_val is not None else top_k_fallback,
            top_k_fallback,
        )
        if stack_top_k_int <= 0:
            try:
                stack_top_k_int = max(
                    0,
                    int(getattr(global_stack_cfg, "retrieval_top_k", stack_top_k_int)),
                )
            except Exception:
                stack_top_k_int = max(0, stack_top_k_int)

        stack_max_lines_val = _first_non_none(
            stack_max_lines,
            getattr(cfg, "stack_max_lines", None),
            getattr(cfg_stack, "max_lines", None),
            getattr(dataset_cfg, "max_lines", None),
            getattr(global_stack_cfg, "max_lines", None),
            getattr(stack_defaults, "max_lines", None),
        )
        lines_fallback = getattr(stack_defaults, "max_lines", 0)
        stack_max_lines_int = _coerce_int(
            stack_max_lines_val if stack_max_lines_val is not None else lines_fallback,
            lines_fallback,
        )
        if stack_max_lines_int < 0:
            stack_max_lines_int = 0

        stack_max_bytes_val = _first_non_none(
            stack_max_bytes,
            getattr(cfg, "stack_max_bytes", None),
            getattr(cfg_stack, "max_bytes", None),
            getattr(dataset_cfg, "max_bytes", None),
            getattr(global_stack_cfg, "max_bytes", None),
            getattr(stack_defaults, "max_bytes", None),
        )
        stack_max_bytes_int: int | None
        if stack_max_bytes_val is None:
            stack_max_bytes_int = None
        else:
            stack_max_bytes_int = _coerce_int(stack_max_bytes_val, int(stack_max_bytes_val))
            if stack_max_bytes_int < 0:
                stack_max_bytes_int = 0

        index_source = _first_non_none(
            stack_index_path,
            getattr(cfg, "stack_index_path", None),
            getattr(cfg_stack, "index_path", None),
            getattr(dataset_cfg, "index_path", None),
            getattr(global_stack_cfg, "index_path", None),
            getattr(stack_defaults, "index_path", None),
        )

        metadata_source = _first_non_none(
            stack_metadata_path,
            getattr(cfg, "stack_metadata_path", None),
            getattr(cfg_stack, "metadata_path", None),
            getattr(dataset_cfg, "metadata_path", None),
            getattr(global_stack_cfg, "metadata_path", None),
            getattr(stack_defaults, "metadata_path", None),
        )

        cache_dir_source = _first_non_none(
            stack_cache_dir,
            getattr(cfg, "stack_cache_dir", None),
            getattr(cfg_stack, "cache_dir", None),
            getattr(dataset_cfg, "cache_dir", None),
            getattr(global_stack_cfg, "cache_dir", None),
            getattr(stack_defaults, "cache_dir", None),
        )

        progress_source = _first_non_none(
            stack_progress_path,
            getattr(cfg, "stack_progress_path", None),
            getattr(cfg_stack, "progress_path", None),
            getattr(dataset_cfg, "progress_path", None),
            getattr(global_stack_cfg, "progress_path", None),
            getattr(stack_defaults, "progress_path", None),
        )

        self.stack_enabled = bool(stack_enabled_bool)
        self.stack_languages = set(lang_set)
        self.stack_top_k = max(0, stack_top_k_int)
        self.stack_max_lines = max(0, stack_max_lines_int)
        self.stack_max_bytes = None if stack_max_bytes_int is None else max(0, stack_max_bytes_int)
        self.stack_index_path = _resolve_optional_path(index_source)
        self.stack_metadata_path = _resolve_optional_path(metadata_source)
        self.stack_cache_dir = _resolve_optional_path(cache_dir_source)
        self.stack_progress_path = _resolve_optional_path(progress_source)
        self.stack_config = dataset_cfg
        prompt_enabled_val = (
            stack_prompt_enabled
            if stack_prompt_enabled is not None
            else _cfg("stack_prompt_enabled", getattr(defaults, "stack_prompt_enabled", True))
        )
        self.stack_prompt_enabled = _coerce_bool(prompt_enabled_val, bool(prompt_enabled_val))
        prompt_limit_val = (
            stack_prompt_limit
            if stack_prompt_limit is not None
            else _cfg("stack_prompt_limit", getattr(defaults, "stack_prompt_limit", 0))
        )
        try:
            prompt_limit_int = int(prompt_limit_val)
        except Exception:
            prompt_limit_int = int(getattr(defaults, "stack_prompt_limit", 0))
        if prompt_limit_int < 0:
            prompt_limit_int = 0
        self.stack_prompt_limit = prompt_limit_int

        self.roi_tag_penalties = roi_tag_penalties
        self.retriever = retriever or Retriever(context_builder=self)
        if patch_retriever is None:
            self.patch_retriever = PatchRetriever(
                metric=similarity_metric,
                enhancement_weight=enhancement_weight,
                context_builder=self,
                service_url=os.environ.get("VECTOR_SERVICE_URL"),
            )
        else:
            self.patch_retriever = patch_retriever
            try:
                self.patch_retriever.enhancement_weight = enhancement_weight
                if not self.patch_retriever.roi_tag_weights:
                    self.patch_retriever.roi_tag_weights = roi_tag_penalties
            except Exception:
                logger.exception("patch_retriever configuration failed")
        self.stack_retriever = stack_retriever
        self.similarity_metric = similarity_metric
        self.enhancement_weight = enhancement_weight

        if ranking_model is None:
            try:  # pragma: no cover - best effort model load
                from pathlib import Path

                try:  # package relative import when available
                    from .. import retrieval_ranker as _rr  # type: ignore
                except Exception:  # pragma: no cover - fallback
                    import retrieval_ranker as _rr  # type: ignore

                cfg = Path("retrieval_ranker.json")
                model_path = cfg
                if cfg.exists():
                    try:
                        data = json.loads(cfg.read_text())
                        if isinstance(data, dict) and data.get("current"):
                            model_path = Path(str(data["current"]))
                    except Exception:
                        logger.exception("failed to parse retrieval_ranker.json")
                self.ranking_model = _rr.load_model(model_path)
            except Exception:
                self.ranking_model = None
        else:
            self.ranking_model = ranking_model
        self.roi_tracker = roi_tracker
        self.ranking_weight = ranking_weight
        self.roi_weight = roi_weight
        self.recency_weight = recency_weight
        self.safety_weight = safety_weight
        self.regret_penalty = regret_penalty
        self.alignment_penalty = alignment_penalty
        self.alert_penalty = alert_penalty
        self.risk_penalty = risk_penalty
        self.max_alignment_severity = max_alignment_severity
        self.max_alerts = max_alerts
        self.license_denylist = set(license_denylist or ())
        self.memory = memory_manager
        self.summariser = summariser or (lambda text: text)
        self._cache: Dict[Tuple[str, int, Tuple[str, ...], Tuple[str, ...]], str] = {}
        self._summary_cache: Dict[int, Dict[str, str]] = {}
        self._excluded_failed_strategies: set[str] = set()
        self.db_weights = db_weights or {}
        if not self.db_weights:
            try:
                self.refresh_db_weights()
            except Exception:
                logger.exception("refresh_db_weights failed")
        self.db_weights.setdefault("stack", 1.0)
        self.max_tokens = max_tokens
        self.precise_token_count = precise_token_count
        self.max_diff_lines = max_diff_lines
        self.patch_safety = patch_safety or PatchSafety()
        self.patch_safety.max_alert_severity = max_alignment_severity
        self.patch_safety.max_alerts = max_alerts
        self.patch_safety.license_denylist = self.license_denylist
        self.prompt_score_weight = prompt_score_weight
        self.prompt_max_tokens = prompt_max_tokens

        self._stack_ready_checked = False
        self._stack_last_ingest = 0
        self._stack_ingest_lock = threading.Lock()

        stack_env_enabled = _truthy(os.environ.get("STACK_STREAMING", "1"))
        if self.stack_enabled and not stack_env_enabled:
            _warn_once(
                "stack_streaming_disabled",
                "STACK_STREAMING disabled via environment; skipping Stack retrieval",
            )
        if self.stack_enabled and stack_env_enabled:
            if not _resolve_hf_token():
                _warn_once(
                    "missing_hf_token",
                    "HUGGINGFACE_TOKEN not configured; Stack ingestion will be disabled",
                )
            try:
                service = getattr(self.patch_retriever, "vector_service", None)
                if service is None:
                    service = getattr(self.retriever, "vector_service", None)
                stack_store = get_stack_vector_store()
                if stack_store is None and service is not None:
                    stack_store = getattr(service, "vector_store", None)
                metadata_path = self.stack_metadata_path or get_stack_metadata_path()
                self.register_stack_index(
                    stack_index=stack_store,
                    metadata_path=metadata_path,
                    vector_service=service,
                )
            except Exception:
                logger.exception("stack retriever initialisation failed")
                self.stack_retriever = None

        # Attempt to use tokenizer from retriever or embedder if provided.
        tok = getattr(self.retriever, "tokenizer", None)
        if tok is None:
            tok = getattr(getattr(self.retriever, "embedder", None), "tokenizer", None)
        self._tokenizer = tok
        if self.precise_token_count and self._tokenizer is None and _FALLBACK_ENCODER is None:
            raise RuntimeError(
                "precise token counting requires the 'tiktoken' package"
            )
        self._fallback_tokenizer = (
            _FALLBACK_ENCODER if self.precise_token_count else None
        )

        # propagate thresholds to retriever when possible
        try:
            self.retriever.max_alert_severity = max_alignment_severity
            self.retriever.max_alerts = max_alerts
            self.retriever.license_denylist = self.license_denylist
        except Exception:
            logger.exception("failed to propagate thresholds to retriever")

        self._embedding_check_interval = embedding_check_interval
        if embedding_check_interval > 0:
            threading.Thread(
                target=self._embedding_checker, daemon=True
            ).start()

    # ------------------------------------------------------------------
    def _embedding_checker(self) -> None:
        backfill = EmbeddingBackfill()
        interval = self._embedding_check_interval * 60
        while True:
            time.sleep(interval)
            try:
                dbs = list(self.db_weights.keys()) or ["code", "bot", "error", "workflow"]
                out_of_sync = backfill.check_out_of_sync(dbs=dbs)
                if out_of_sync:
                    asyncio.run(schedule_backfill(dbs=out_of_sync))
            except Exception:
                logger.exception("background embedding check failed")

    # ------------------------------------------------------------------
    def exclude_failed_strategies(self, tags: List[str]) -> None:
        """Remember strategy tags that should be excluded from results."""
        for tag in tags:
            if isinstance(tag, str) and tag:
                self._excluded_failed_strategies.add(tag)

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_embedding(vector: Any) -> List[float] | None:
        """Return ``vector`` as a list of floats when possible."""

        if vector is None:
            return None
        try:
            seq = list(vector)
        except TypeError:
            return None
        result: List[float] = []
        for value in seq:
            try:
                result.append(float(value))
            except Exception:
                return None
        return result if result else None

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, Any]:
        """Best-effort conversion of ``value`` to a dictionary."""

        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_dict"):
            try:
                payload = value.to_dict()
                if isinstance(payload, dict):
                    return dict(payload)
            except Exception:
                pass
        if hasattr(value, "_asdict"):
            try:
                payload = value._asdict()
                if isinstance(payload, dict):
                    return dict(payload)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float] | None:
        """Return an embedding for ``query`` if an encoder is available."""

        sources: List[Any] = [self.retriever]
        base = getattr(self.retriever, "retriever", None)
        if base is not None:
            sources.append(base)
        stack_cfg = _stack_dataset_config()
        if self.stack_retriever is not None and self.stack_enabled:
            sources.append(self.stack_retriever)
            embedder = getattr(self.stack_retriever, "embedder", None)
            if embedder is not None:
                sources.append(embedder)
        seen: set[int] = set()
        for source in sources:
            if source is None:
                continue
            ident = id(source)
            if ident in seen:
                continue
            seen.add(ident)
            for attr in ("embed_query", "encode", "embed"):
                func = getattr(source, attr, None)
                if not callable(func):
                    continue
                try:
                    vector = func(query)
                except Exception:
                    continue
                coerced = self._coerce_embedding(vector)
                if coerced:
                    return coerced
            embedder = getattr(source, "embedder", None)
            if embedder is None or id(embedder) in seen:
                continue
            seen.add(id(embedder))
            for attr in ("embed_query", "encode", "embed"):
                func = getattr(embedder, attr, None)
                if not callable(func):
                    continue
                try:
                    vector = func(query)
                except Exception:
                    continue
                coerced = self._coerce_embedding(vector)
                if coerced:
                    return coerced
        return None

    # ------------------------------------------------------------------
    def _stack_namespace(self) -> str:
        retr = getattr(self, "stack_retriever", None)
        if retr is None:
            return "stack"
        return str(getattr(retr, "namespace", "stack") or "stack")

    # ------------------------------------------------------------------
    def _stack_metadata_path(self) -> Path | None:
        if isinstance(self.stack_metadata_path, Path):
            return self.stack_metadata_path
        retr = getattr(self, "stack_retriever", None)
        if retr is None:
            return None
        getter = getattr(retr, "get_metadata_path", None)
        if callable(getter):
            try:
                path = getter()
            except Exception:
                path = None
            if isinstance(path, Path):
                return path
            if isinstance(path, str):
                try:
                    return Path(path)
                except Exception:
                    return None
        candidate = getattr(retr, "metadata_db_path", None)
        if isinstance(candidate, Path):
            return candidate
        if isinstance(candidate, str):
            try:
                return Path(candidate)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    def _stack_index_path(self) -> Path | None:
        if isinstance(self.stack_index_path, Path):
            return self.stack_index_path
        retr = getattr(self, "stack_retriever", None)
        if retr is None:
            return None
        getter = getattr(retr, "get_index_path", None)
        if callable(getter):
            try:
                path = getter()
            except Exception:
                path = None
            if isinstance(path, Path):
                return path
            if isinstance(path, str):
                try:
                    return Path(path)
                except Exception:
                    return None
        store = getattr(retr, "stack_index", None) or getattr(retr, "vector_store", None)
        candidate = getattr(store, "path", None) or getattr(store, "index_path", None)
        if isinstance(candidate, Path):
            return candidate
        if isinstance(candidate, str):
            try:
                return Path(candidate)
            except Exception:
                return None
        metadata = self._stack_metadata_path()
        if isinstance(metadata, Path):
            return metadata.with_suffix(".index")
        return None

    # ------------------------------------------------------------------
    def register_stack_index(
        self,
        *,
        stack_index: Any | None = None,
        metadata_path: Path | str | None = None,
        vector_service: Any | None = None,
        languages: Iterable[str] | None = None,
    ) -> Any | None:
        if not self.stack_enabled:
            return None

        resolved_languages = {
            str(lang).strip().lower()
            for lang in (languages or self.stack_languages or set())
            if isinstance(lang, str) and lang.strip()
        }

        service = vector_service
        if service is None:
            try:
                service = getattr(self.patch_retriever, "vector_service", None)
            except Exception:
                service = None
        if service is None:
            try:
                service = getattr(self.retriever, "vector_service", None)
            except Exception:
                service = None

        store = stack_index
        if store is None and service is not None:
            store = getattr(service, "vector_store", None)

        meta_path: Path | None
        if isinstance(metadata_path, Path):
            meta_path = metadata_path
        elif isinstance(metadata_path, str):
            try:
                meta_path = Path(metadata_path)
            except Exception:
                meta_path = None
        else:
            meta_path = None
        if meta_path is None:
            candidate = self.stack_metadata_path or get_stack_metadata_path()
            if isinstance(candidate, Path):
                meta_path = candidate
            elif isinstance(candidate, str):
                try:
                    meta_path = Path(candidate)
                except Exception:
                    meta_path = None

        retr = getattr(self, "stack_retriever", None)
        if retr is None and store is not None:
            top_k = self.stack_top_k or 0
            if top_k <= 0:
                top_k = 5
            max_lines = self.stack_max_lines or 0
            try:
                retr = StackRetriever(
                    context_builder=self,
                    vector_service=service,
                    stack_index=store,
                    metadata_db_path=meta_path,
                    top_k=max(1, int(top_k)),
                    max_lines=max(0, int(max_lines)),
                    patch_safety=self.patch_safety,
                    license_denylist=set(self.license_denylist),
                    risk_penalty=self.risk_penalty,
                    roi_tag_weights=dict(self.roi_tag_penalties),
                    max_alert_severity=self.max_alignment_severity,
                    max_alerts=self.max_alerts,
                    languages=resolved_languages or None,
                )
            except Exception:
                logger.exception("failed to create stack retriever")
                retr = None
            self.stack_retriever = retr
        elif retr is not None:
            try:
                if store is not None:
                    setattr(retr, "stack_index", store)
                    setattr(retr, "vector_store", store)
                if service is not None:
                    setattr(retr, "vector_service", service)
                if meta_path is not None:
                    setattr(retr, "metadata_db_path", meta_path)
                    setattr(retr, "_metadata_conn", None)
                if hasattr(retr, "patch_safety"):
                    setattr(retr, "patch_safety", self.patch_safety)
                if hasattr(retr, "license_denylist"):
                    setattr(retr, "license_denylist", set(self.license_denylist))
                if hasattr(retr, "risk_penalty"):
                    setattr(retr, "risk_penalty", self.risk_penalty)
                if hasattr(retr, "roi_tag_weights") and not getattr(
                    retr, "roi_tag_weights", {}
                ):
                    setattr(retr, "roi_tag_weights", dict(self.roi_tag_penalties))
                if hasattr(retr, "max_alert_severity"):
                    setattr(retr, "max_alert_severity", self.max_alignment_severity)
                if hasattr(retr, "max_alerts"):
                    setattr(retr, "max_alerts", self.max_alerts)
                if hasattr(retr, "max_lines") and self.stack_max_lines:
                    setattr(retr, "max_lines", max(0, int(self.stack_max_lines)))
                if hasattr(retr, "top_k") and self.stack_top_k:
                    setattr(retr, "top_k", max(1, int(self.stack_top_k)))
                if resolved_languages:
                    if hasattr(retr, "set_languages"):
                        retr.set_languages(resolved_languages)
                    elif hasattr(retr, "languages"):
                        setattr(retr, "languages", set(resolved_languages))
            except Exception:
                logger.exception("stack retriever configuration failed")

        if isinstance(store, Path):
            self.stack_index_path = store
        else:
            index_path = getattr(store, "path", None) or getattr(store, "index_path", None)
            if isinstance(index_path, (str, Path)):
                try:
                    self.stack_index_path = Path(index_path)
                except Exception:
                    pass
        if meta_path is not None:
            self.stack_metadata_path = meta_path

        if retr is None:
            return None

        if resolved_languages:
            try:
                self.stack_languages = set(resolved_languages)
            except Exception:
                pass

        self._stack_ready_checked = False
        return retr

    # ------------------------------------------------------------------
    def _is_stack_index_stale(self) -> bool:
        retr = getattr(self, "stack_retriever", None)
        if retr is None:
            return False
        checker = getattr(retr, "is_index_stale", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                logger.exception("stack retriever staleness check failed")
                return True
        path = self._stack_metadata_path()
        if path is None or not path.exists():
            return True
        try:
            conn = sqlite3.connect(str(path))
        except Exception:
            return True
        tables = {
            getattr(retr, "_metadata_table", ""),
            f"{self._stack_namespace()}_embeddings",
        }
        try:
            tables = {t for t in tables if t}
            for table in tables:
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                except Exception:
                    continue
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        if int(row[0]) > 0:
                            return False
                    except Exception:
                        return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
        store = getattr(retr, "stack_index", None) or getattr(retr, "vector_store", None)
        ids = getattr(store, "ids", None)
        if isinstance(ids, list) and ids:
            return False
        return True

    # ------------------------------------------------------------------
    def _ingest_stack_embeddings(
        self,
        *,
        resume: bool = True,
        limit: int | None = None,
    ) -> int:
        retr = getattr(self, "stack_retriever", None)
        if retr is None:
            return 0
        try:
            from .stack_ingestor import StackIngestor  # type: ignore
        except Exception:
            logger.exception("stack ingestion helpers unavailable")
            raise

        stack_cfg = _stack_dataset_config()
        languages = [
            str(lang).strip().lower()
            for lang in (self.stack_languages or set())
            if isinstance(lang, str) and lang.strip()
        ]
        if not languages:
            try:
                languages = [
                    str(lang).strip().lower()
                    for lang in getattr(stack_cfg, "languages", set())
                    if isinstance(lang, str) and lang.strip()
                ]
            except Exception:
                languages = []
        language_seq: Tuple[str, ...] | None = tuple(sorted(set(languages))) or None

        max_lines = max(0, int(self.stack_max_lines)) if self.stack_max_lines else 0
        if not max_lines:
            try:
                max_lines = max(
                    0, int(getattr(stack_cfg, "max_lines", 0))
                )
            except Exception:
                max_lines = 0
        max_lines_arg: int | None = max_lines or None

        max_bytes: int | None = None
        if getattr(self, "stack_max_bytes", None):
            try:
                max_bytes = max(0, int(self.stack_max_bytes)) or None
            except Exception:
                max_bytes = None
        if max_bytes is None:
            try:
                candidate_bytes = getattr(stack_cfg, "max_bytes", None)
                if candidate_bytes:
                    max_bytes = max(0, int(candidate_bytes)) or None
            except Exception:
                max_bytes = None

        metadata_path = self._stack_metadata_path() or get_stack_metadata_path()
        if metadata_path is None:
            cache_dir = getattr(self, "stack_cache_dir", None)
            if isinstance(cache_dir, Path):
                metadata_path = cache_dir / "stack_embeddings.db"
            else:
                metadata_path = Path("stack_embeddings.db")
        elif not isinstance(metadata_path, Path):
            metadata_path = Path(str(metadata_path))

        cache_dir = getattr(self, "stack_cache_dir", None)
        if metadata_path is not None and isinstance(cache_dir, Path):
            try:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        index_path = self._stack_index_path()
        if index_path is None and isinstance(metadata_path, Path):
            index_path = metadata_path.with_suffix(".index")
        if index_path is None and isinstance(cache_dir, Path):
            index_path = cache_dir / "stack.index"

        vector_store = getattr(retr, "stack_index", None) or getattr(retr, "vector_store", None)

        chunk_lines: int | None = None
        stack_config = getattr(self, "stack_config", None)
        try:
            candidate_chunk = getattr(stack_config, "chunk_lines", None)
        except Exception:
            candidate_chunk = None
        if candidate_chunk is None:
            try:
                candidate_chunk = getattr(stack_cfg, "chunk_lines", None)
            except Exception:
                candidate_chunk = None
        if candidate_chunk:
            try:
                chunk_lines = max(1, int(candidate_chunk))
            except Exception:
                chunk_lines = None

        token = _resolve_hf_token()
        if token is None:
            _warn_once(
                "stack_ingest_missing_token",
                "HUGGINGFACE_TOKEN not configured; Stack ingestion may fail",
            )

        ingestor = StackIngestor(
            languages=language_seq,
            max_lines=max_lines_arg,
            max_bytes=max_bytes,
            chunk_lines=chunk_lines if chunk_lines is not None else 200,
            namespace=self._stack_namespace(),
            metadata_path=metadata_path,
            index_path=index_path,
            vector_store=vector_store,
            use_auth_token=token,
        )
        processed = ingestor.ingest(resume=resume, limit=limit)

        updated_meta_path = Path(ingestor.metadata_store.path)
        self.stack_metadata_path = updated_meta_path
        try:
            retr.metadata_db_path = updated_meta_path
        except Exception:
            pass
        try:
            retr._metadata_conn = None  # type: ignore[attr-defined]
        except Exception:
            pass

        if hasattr(ingestor.vector_store, "path"):
            try:
                self.stack_index_path = Path(ingestor.vector_store.path)
            except Exception:
                pass
        try:
            retr.stack_index = ingestor.vector_store
            setattr(retr, "vector_store", ingestor.vector_store)
        except Exception:
            pass

        self._stack_last_ingest = processed
        return processed

    # ------------------------------------------------------------------
    def ensure_stack_embeddings(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        resume: bool = True,
    ) -> int:
        if not self.stack_enabled or self.stack_retriever is None:
            return 0
        with self._stack_ingest_lock:
            needs_refresh = force or self._is_stack_index_stale()
            if not needs_refresh:
                self._stack_ready_checked = True
                return 0
            try:
                processed = self._ingest_stack_embeddings(resume=resume, limit=limit)
            except Exception:
                logger.exception("stack ingestion failed")
                self._stack_ready_checked = False
                return 0
            self._stack_ready_checked = True
            return processed

    # ------------------------------------------------------------------
    def trigger_stack_ingestion(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        resume: bool = True,
    ) -> int:
        """Explicitly refresh Stack embeddings when external caches change."""

        return self.ensure_stack_embeddings(force=force, limit=limit, resume=resume)

    # ------------------------------------------------------------------
    def _ensure_stack_ready(self) -> None:
        if not self.stack_enabled or self.stack_retriever is None:
            return
        if self._stack_ready_checked:
            return
        self.ensure_stack_embeddings()

    # ------------------------------------------------------------------
    def ingest(
        self,
        *,
        force_stack: bool = True,
        stack_limit: int | None = None,
        stack_resume: bool = True,
    ) -> Dict[str, Any]:
        """Trigger background ingestion tasks such as Stack streaming."""

        results: Dict[str, Any] = {}
        if not self.stack_enabled or self.stack_retriever is None:
            return results
        processed = self.ensure_stack_embeddings(
            force=force_stack, limit=stack_limit, resume=stack_resume
        )
        results["stack"] = processed
        return results

    # ------------------------------------------------------------------
    def _normalise_stack_hit(self, raw: Any, query: str) -> Dict[str, Any] | None:
        """Normalise a stack retrieval ``raw`` result into bundle format."""

        data = self._coerce_mapping(raw)
        if not data:
            return None

        metadata_raw = data.get("metadata")
        if metadata_raw is None:
            metadata_raw = {
                key: value
                for key, value in data.items()
                if key
                not in {
                    "score",
                    "similarity",
                    "distance",
                    "vector",
                    "id",
                    "record_id",
                }
            }
        meta = self._coerce_mapping(metadata_raw)

        for key in (
            "repo",
            "path",
            "language",
            "summary",
            "snippet",
            "content",
            "text",
            "start",
            "end",
            "license",
            "license_fingerprint",
            "semantic_alerts",
            "alignment_severity",
            "tags",
        ):
            if key not in meta and key in data:
                meta[key] = data[key]

        clean_meta: Dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            if isinstance(value, str):
                clean_meta[key] = redact_text(value)
            else:
                clean_meta[key] = value

        snippet_source = clean_meta.get("summary") or clean_meta.get("snippet")
        snippet_text = None
        if isinstance(snippet_source, str) and snippet_source.strip():
            snippet_text = redact_text(snippet_source)
        if snippet_text is None:
            for candidate in ("content", "text"):
                value = clean_meta.get(candidate)
                if isinstance(value, str) and value.strip():
                    snippet_text = redact_text(value)
                    break
        clean_meta.pop("snippet", None)
        clean_meta.pop("content", None)
        clean_meta.pop("text", None)

        clean_meta.setdefault("origin", "stack")
        clean_meta.setdefault("redacted", True)

        repo = clean_meta.get("repo")
        if isinstance(repo, str) and repo:
            repo = redact_text(repo)
            clean_meta["repo"] = repo
        path = clean_meta.get("path")
        if isinstance(path, str) and path:
            path = redact_text(path)
            clean_meta["path"] = path
        language = clean_meta.get("language")
        if isinstance(language, str) and language:
            language = language.strip().lower()
            clean_meta["language"] = language

        summary_lines: list[str] = []
        location_bits = [
            part
            for part in (repo, path)
            if isinstance(part, str) and part.strip()
        ]
        if location_bits:
            location = "/".join(location_bits)
            if isinstance(language, str) and language:
                location = f"{location} [{language}]"
            summary_lines.append(location)
        if isinstance(snippet_text, str) and snippet_text.strip():
            summary_lines.append(snippet_text.strip())

        text = "\n".join(line for line in summary_lines if line).strip()
        if text:
            clean_meta["summary"] = text
        else:
            text = clean_meta.get("summary") or ""

        record_id = (
            data.get("id")
            or data.get("record_id")
            or clean_meta.get("record_id")
            or clean_meta.get("id")
        )
        if not record_id:
            repo = clean_meta.get("repo") or ""
            path = clean_meta.get("path") or ""
            start = clean_meta.get("start") or ""
            end = clean_meta.get("end") or ""
            composite = "|".join(str(part) for part in (repo, path, start, end) if part)
            if composite:
                record_id = composite
            else:
                try:
                    payload = json.dumps(
                        {
                            "query": query,
                            "repo": repo,
                            "path": path,
                            "summary": clean_meta.get("summary", ""),
                        },
                        sort_keys=True,
                    )
                except Exception:
                    payload = f"{repo}:{path}:{clean_meta.get('summary', '')}"
                record_id = uuid.uuid5(uuid.NAMESPACE_URL, payload).hex

        raw_score = (
            data.get("score")
            or data.get("similarity")
            or clean_meta.get("score")
            or 0.0
        )
        try:
            score = float(raw_score)
        except Exception:
            score = 0.0

        text = clean_meta.get("summary") or ""
        if text and isinstance(self.stack_max_bytes, int) and self.stack_max_bytes > 0:
            try:
                encoded = text.encode("utf-8")
                if len(encoded) > self.stack_max_bytes:
                    text = encoded[: self.stack_max_bytes].decode("utf-8", errors="ignore")
                    clean_meta["summary"] = text
                    clean_meta["stack_truncated_bytes"] = self.stack_max_bytes
            except Exception:
                pass

        bundle = {
            "origin_db": "stack",
            "record_id": record_id,
            "metadata": clean_meta,
            "text": text,
            "score": score,
            "similarity": score,
        }
        return bundle

    # ------------------------------------------------------------------
    def refresh_db_weights(
        self,
        weights: Dict[str, float] | None = None,
        *,
        vector_metrics: "VectorMetricsDB" | None = None,
    ) -> None:
        """Refresh ranking weights for origin databases.

        Parameters
        ----------
        weights:
            Optional mapping of database name to weight. When omitted the
            method attempts to load weights from ``vector_metrics`` or the
            global :class:`VectorMetricsDB` instance.
        vector_metrics:
            Database from which weights are loaded when ``weights`` is ``None``.
            The argument defaults to the module-level instance when available.
        """

        global _VEC_METRICS
        if weights is None:
            vm = vector_metrics or _VEC_METRICS
            if vm is None:
                return
            try:
                weights = vm.get_db_weights()
            except Exception:
                return
            if vector_metrics is not None:
                _VEC_METRICS = vector_metrics
        if not isinstance(weights, dict):  # pragma: no cover - defensive
            return
        # Replace existing mapping so each refresh reflects the latest weights
        # from the metrics database.  This avoids stale entries lingering after
        # patches adjust the ranking model.
        try:
            self.db_weights.clear()
            self.db_weights.update(weights)
        except Exception:  # pragma: no cover - best effort
            self.db_weights = dict(weights)

        return dict(self.db_weights)

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        try:
            return self.summariser(text)
        except Exception:  # pragma: no cover - summariser failure
            pass
        if self.memory and hasattr(self.memory, "_summarise_text"):
            try:
                return self.memory._summarise_text(text)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fallback
                pass
        return text

    def _truncate_diff(self, diff: str) -> str:
        if self.max_diff_lines and self.max_diff_lines > 0:
            lines = diff.splitlines()
            if len(lines) > self.max_diff_lines:
                return "\n".join(lines[: self.max_diff_lines])
        return diff

    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Return an estimate of tokens for ``text``.

        The method prefers a tokenizer supplied by the retriever or its
        embedder.  When unavailable, and ``precise_token_count`` is enabled, it
        attempts to use a lightweight dependency such as ``tiktoken`` for more
        accurate measurement.  If this dependency is missing or disabled the
        method falls back to a regex approximation which counts contiguous word
        characters.  The goal here is not perfect parity with any model but a
        consistent budget estimate for trimming.
        """
        if self._tokenizer is not None:
            try:  # pragma: no cover - defensive against tokeniser failures
                return len(self._tokenizer.encode(text))
            except Exception:
                logger.exception("tokenizer encode failed; falling back to regex")
        if self.precise_token_count:
            if self._fallback_tokenizer is None:
                raise RuntimeError(
                    "tiktoken encoding not available for precise token counting"
                )
            return len(self._fallback_tokenizer.encode(text))
        return len(re.findall(r"\w+", text))

    # ------------------------------------------------------------------
    def _metric(
        self,
        origin: str,
        meta: Dict[str, Any],
        query: str,
        text: str,
        vector_id: str,
    ) -> float | None:
        """Extract ROI/success metrics from metadata and ranking model.

        The method merges metadata from the retriever with historical safety
        signals stored in :class:`VectorMetricsDB`.  When a vector ID is
        provided, win/regret rates and alignment severity are looked up and
        merged into *meta* so downstream consumers can surface them.
        """

        metric: float | None = None
        try:
            if origin == "error":
                freq = meta.get("frequency")
                if freq is not None:
                    metric = 1.0 / (1.0 + float(freq))
            elif origin == "bot":
                for key in ("roi", "deploy_count"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "workflow":
                for key in ("roi", "usage", "runs"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "enhancement":
                for key in ("roi", "adoption"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "information":
                for key in ("roi", "data_depth", "data_depth_score", "quality"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "code":
                for key in ("roi", "patch_success"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
            elif origin == "discrepancy":
                for key in ("roi", "severity", "impact"):
                    if key in meta and meta[key] is not None:
                        metric = float(meta[key])
                        break
        except Exception:  # pragma: no cover - defensive
            metric = None

        # Patch safety metrics supplied by PatchLogger or VectorMetricsDB
        win_rate = meta.get("win_rate")
        regret_rate = meta.get("regret_rate")
        sev = meta.get("alignment_severity")

        if _VEC_METRICS is not None and vector_id:
            try:  # pragma: no cover - best effort lookup
                if win_rate is None or regret_rate is None:
                    cur = _VEC_METRICS.conn.execute(
                        "SELECT AVG(win), AVG(regret) FROM vector_metrics WHERE vector_id=?",
                        (vector_id,),
                    )
                    row = cur.fetchone()
                    if win_rate is None and row and row[0] is not None:
                        win_rate = float(row[0])
                        meta["win_rate"] = win_rate
                    if regret_rate is None and row and row[1] is not None:
                        regret_rate = float(row[1])
                        meta["regret_rate"] = regret_rate
                if sev is None:
                    cur = _VEC_METRICS.conn.execute(
                        "SELECT MAX(alignment_severity) FROM patch_ancestry WHERE vector_id=?",
                        (vector_id,),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        sev = float(row[0])
                        meta["alignment_severity"] = sev
            except Exception:
                pass

        alerts = meta.get("semantic_alerts")
        try:
            if win_rate is not None or regret_rate is not None:
                win = min(max(float(win_rate or 0.0), 0.0), 1.0)
                regret = min(max(float(regret_rate or 0.0), 0.0), 1.0)
                metric = (metric or 0.0) + self.safety_weight * (win - regret)
            if alerts:
                metric = (metric or 0.0) - (
                    float(len(alerts))
                    if isinstance(alerts, (list, tuple, set))
                    else 1.0
                )
        except Exception:  # pragma: no cover - defensive
            pass

        if sev is not None:
            try:
                sev_val = min(max(float(sev), 0.0), 1.0)
                metric = (metric or 0.0) - self.safety_weight * sev_val
            except Exception:
                pass

        lic = meta.get("license")
        fp = meta.get("license_fingerprint")
        if lic in self.license_denylist or _LICENSE_DENYLIST.get(fp) in self.license_denylist:
            metric = (metric or 0.0) - self.safety_weight

        if self.ranking_model is not None:
            try:
                if hasattr(self.ranking_model, "score"):
                    metric = (metric or 0.0) + float(
                        self.ranking_model.score(query, text)
                    )
                elif hasattr(self.ranking_model, "rank"):
                    metric = (metric or 0.0) + float(
                        self.ranking_model.rank(query, text)
                    )
                else:
                    metric = (metric or 0.0) + float(
                        self.ranking_model(query, text)  # type: ignore[misc]
                    )
            except Exception:
                pass

        if metric is not None and self.roi_tracker is not None:
            try:
                _base, raroi, _ = self.roi_tracker.calculate_raroi(float(metric))
                metric = float(raroi)
            except Exception:
                pass

        return metric

    # ------------------------------------------------------------------
    def _bundle_to_entry(self, bundle: Dict[str, Any], query: str) -> Tuple[str, _ScoredEntry]:
        meta = bundle.get("metadata", {}) or {}
        origin = bundle.get("origin_db", "")

        text = bundle.get("text") or ""
        vec_id = str(bundle.get("record_id", ""))

        supplied_risk = any(
            (isinstance(meta, dict) and meta.get(k) is not None) or bundle.get(k) is not None
            for k in ("risk_score", "final_risk_score", "risk")
        )

        # Evaluate patch safety before any scoring so risky vectors can be
        # dropped early and their similarity scores propagated downstream.
        self.patch_safety.max_alert_severity = self.max_alignment_severity
        self.patch_safety.max_alerts = self.max_alerts
        self.patch_safety.license_denylist = self.license_denylist
        passed, similarity, risks = self.patch_safety.evaluate(meta, meta, origin=origin)
        if not passed:
            if _VECTOR_RISK is not None:
                try:
                    _VECTOR_RISK.labels("filtered").inc()
                except Exception:
                    pass
            return "", _ScoredEntry({}, 0.0, origin, vec_id, {})

        # Preserve any existing risk score but ensure the similarity is
        # recorded so later stages can apply ranking penalties.
        existing = 0.0
        try:
            if isinstance(meta, dict) and meta.get("risk_score") is not None:
                existing = float(meta.get("risk_score", 0.0))
        except Exception:
            existing = 0.0
        risk_score = max(existing, risks.get(origin, similarity))
        try:
            if isinstance(meta, dict):
                meta["risk_score"] = risk_score
                if not supplied_risk:
                    meta["risk_score_defaulted"] = True
                    logger.warning(
                        "risk_score missing for %s; defaulting to 0.0", vec_id
                    )
        except Exception:
            pass

        entry: Dict[str, Any] = {"id": bundle.get("record_id")}
        alerts = bundle.get("semantic_alerts") or meta.get("semantic_alerts")
        severity = bundle.get("alignment_severity") or meta.get("alignment_severity")
        lic = bundle.get("license") or meta.get("license")
        fp = bundle.get("license_fingerprint") or meta.get("license_fingerprint")

        if origin == "error":
            text = text or meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = text or meta.get("name") or meta.get("purpose") or ""
            if "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
        elif origin == "workflow":
            text = text or meta.get("title") or meta.get("description") or ""
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
        elif origin == "enhancement":
            text = (
                text
                or meta.get("title")
                or meta.get("description")
                or meta.get("lessons")
                or ""
            )
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
            elif "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
            if meta.get("lessons"):
                entry["lessons"] = self._summarise(
                    redact_text(str(meta["lessons"]))
                )
        elif origin == "action":
            text = (
                text
                or meta.get("action_description")
                or meta.get("description")
                or ""
            )
            if "action_type" in meta:
                entry["action_type"] = redact_text(str(meta["action_type"]))
            if "target_domain" in meta:
                entry["target_domain"] = redact_text(str(meta["target_domain"]))
        elif origin == "information":
            text = (
                text
                or meta.get("title")
                or meta.get("summary")
                or meta.get("content")
                or meta.get("lessons")
                or ""
            )
            if "title" in meta:
                entry["title"] = redact_text(str(meta["title"]))
            elif "name" in meta:
                entry["name"] = redact_text(str(meta["name"]))
            if meta.get("lessons"):
                entry["lessons"] = self._summarise(
                    redact_text(str(meta["lessons"]))
                )
        elif origin == "stack":
            text = text or meta.get("summary") or ""
            summary = meta.get("summary") or text
            summary = self._summarise(redact_text(str(summary)))
            entry.setdefault("desc", summary)
            entry["summary"] = summary
            meta["summary"] = summary
            path = meta.get("path")
            if path:
                entry["path"] = redact_text(str(path))
            repo = meta.get("repo")
            if repo:
                entry["repo"] = redact_text(str(repo))
            language = meta.get("language")
            if language:
                entry["language"] = str(language)
            start = meta.get("start")
            end = meta.get("end")
            if start is not None:
                try:
                    entry["start"] = int(start)
                except Exception:
                    entry["start"] = start
            if end is not None:
                try:
                    entry["end"] = int(end)
                except Exception:
                    entry["end"] = end
        elif origin == "discrepancy":
            text = text or meta.get("message") or meta.get("description") or ""
        elif origin == "patch":
            text = (
                text
                or meta.get("diff")
                or meta.get("description")
                or meta.get("summary")
                or ""
            )
        elif origin == "code":
            text = text or meta.get("summary") or meta.get("code") or ""

        text = redact_text(str(text))
        entry["desc"] = text
        metric = self._metric(origin, meta, query, text, vec_id)
        if metric is not None:
            entry["metric"] = metric
            entry["roi"] = metric

        roi_val = meta.get("roi") if isinstance(meta, dict) else None
        if roi_val is None:
            roi_val = bundle.get("roi")
        if roi_val is not None:
            try:
                roi_val = float(roi_val)
                if self.roi_tracker is not None:
                    try:
                        _b, roi_val, _ = self.roi_tracker.calculate_raroi(roi_val)
                    except Exception:
                        pass
                entry["roi"] = float(roi_val)
            except Exception:
                pass

        risk_val = None
        for key in ("risk_score", "final_risk_score", "risk"):
            if isinstance(meta, dict) and meta.get(key) is not None:
                risk_val = meta.get(key)
                break
            if bundle.get(key) is not None:
                risk_val = bundle.get(key)
                break
        if risk_val is None:
            risk_val = 0.0
            entry["risk_score"] = 0.0
        else:
            try:
                risk_val = float(risk_val)
                entry["risk_score"] = risk_val
            except Exception:
                risk_val = 0.0
                entry["risk_score"] = 0.0
        if meta.get("risk_score_defaulted"):
            entry["risk_score_defaulted"] = True

        # Surface patch safety metrics when available
        win_rate = meta.get("win_rate")
        regret_rate = meta.get("regret_rate")
        if win_rate is not None:
            try:
                entry["win_rate"] = float(win_rate)
            except Exception:
                pass
        if regret_rate is not None:
            try:
                entry["regret_rate"] = float(regret_rate)
            except Exception:
                pass

        # Patch safety flags
        flags: Dict[str, Any] = {}
        lic = bundle.get("license") or meta.get("license")
        fp = bundle.get("license_fingerprint") or meta.get("license_fingerprint")
        if lic:
            flags["license"] = lic
        if fp:
            flags["license_fingerprint"] = fp
        if alerts:
            flags["semantic_alerts"] = alerts
        if severity is not None:
            flags["alignment_severity"] = severity
        if flags:
            entry["flags"] = flags

        if _VEC_METRICS is not None and origin:
            try:  # pragma: no cover - best effort metrics lookup
                entry.setdefault(
                    "win_rate", _VEC_METRICS.retriever_win_rate(origin)
                )
                entry.setdefault(
                    "regret_rate", _VEC_METRICS.retriever_regret_rate(origin)
                )
            except Exception:
                pass

        penalty = 0.0
        if regret_rate is not None:
            try:
                penalty += float(regret_rate) * self.regret_penalty
            except Exception:
                pass
        if severity is not None:
            try:
                penalty += float(severity) * self.alignment_penalty
            except Exception:
                pass
        if alerts:
            penalty += (
                len(alerts) if isinstance(alerts, (list, tuple, set)) else 1.0
            ) * self.alert_penalty
        if lic in self.license_denylist or _LICENSE_DENYLIST.get(fp) in self.license_denylist:
            penalty += 1.0
        penalty *= self.safety_weight

        similarity = float(bundle.get("similarity", bundle.get("score", 0.0)))
        try:
            entry["similarity"] = similarity
        except Exception:
            pass
        enhancement = bundle.get("enhancement_score")
        if enhancement is None and isinstance(meta, dict):
            enhancement = meta.get("enhancement_score")
        if enhancement is None and origin == "patch" and PatchHistoryDB is not None:
            try:
                pid = meta.get("patch_id") if isinstance(meta, dict) else None
                if pid is None:
                    pid = vec_id
                pid = int(pid)
                rec = PatchHistoryDB().get(pid)  # type: ignore[operator]
                if rec is not None:
                    enhancement = getattr(rec, "enhancement_score", None)
            except Exception:
                enhancement = None
        if enhancement is not None:
            try:
                enhancement = float(enhancement)
            except Exception:
                enhancement = None
        if enhancement is not None:
            entry["enhancement_score"] = enhancement

        rank_prob = self.ranking_weight
        roi_bias = self.roi_weight
        if self.roi_tracker is not None:
            try:
                roi_bias = float(
                    self.roi_tracker.retrieval_bias().get(origin, self.roi_weight)
                )
            except Exception:
                roi_bias = self.roi_weight

        roi_score = entry.get("roi")
        base = similarity * rank_prob * roi_bias
        if enhancement is not None:
            try:
                base *= 1.0 + enhancement * self.enhancement_weight
            except Exception:
                pass
        if roi_score is not None:
            try:
                base *= 1.0 + float(roi_score) * self.roi_weight
            except Exception:
                pass
        try:
            penalty += (
                float(risk_val)
                * self.risk_penalty
                * rank_prob
                * roi_bias
                * self.safety_weight
            )
        except Exception:
            pass
        score = base - penalty
        score *= self.db_weights.get(origin, 1.0)

        if self.roi_tracker is not None and roi_score is not None:
            try:
                self.roi_tracker.update_db_metrics(
                    {origin: {"roi": float(roi_score)}},
                    sqlite_path="db_roi_metrics.db",
                )
                try:
                    self.roi_tracker.save_history("roi_history.db")
                except Exception:
                    pass
            except Exception:
                logger.exception("roi tracker logging failed")

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "enhancement": "enhancements",
            "action": "actions",
            "information": "information",
            "code": "code",
            "discrepancy": "discrepancies",
            "patch": "patches",
            "stack": "stack",
        }
        return key_map.get(origin, ""), _ScoredEntry(entry, score, origin, vec_id, meta)

    # ------------------------------------------------------------------
    def _merge_metadata(self, scored: _ScoredEntry, bucket: str) -> Dict[str, Any]:
        """Merge metadata and inject patch history details.

        The merged record summarises descriptions, diffs and outcomes so
        callers receive compact patch context suitable for prompts.

        Parameters
        ----------
        scored:
            Entry produced by :meth:`_bundle_to_entry`.
        bucket:
            Target bucket name used for final context payload.
        """

        full: Dict[str, Any] = dict(scored.entry)
        meta = scored.metadata or {}
        full.update(meta)
        full.setdefault("vector_id", scored.vector_id)
        full.setdefault("origin", scored.origin)
        full.setdefault("bucket", bucket)
        full.setdefault("score", scored.score)

        patch_id = meta.get("patch_id") if isinstance(meta, dict) else None
        try:
            patch_id = int(patch_id) if patch_id is not None else None
        except Exception:
            patch_id = None
        cache = self._summary_cache.get(patch_id) if patch_id is not None else None

        # Normalise patch-specific fields from retrieval metadata when present.
        summary = meta.get("summary") or meta.get("description")
        diff = meta.get("diff")
        outcome = meta.get("outcome")
        roi_delta = meta.get("roi_delta")
        lines_changed = meta.get("lines_changed")
        tests_passed = meta.get("tests_passed")
        if summary:
            if cache and "summary" in cache:
                summary = cache["summary"]
            else:
                summary = self._summarise(str(summary))
                if patch_id is not None:
                    self._summary_cache.setdefault(patch_id, {})["summary"] = summary
            full.setdefault("desc", summary)
            full["summary"] = summary
        if diff:
            if cache and "diff" in cache:
                diff = cache["diff"]
            else:
                diff = self._truncate_diff(str(diff))
                diff = self._summarise(diff)
                if patch_id is not None:
                    self._summary_cache.setdefault(patch_id, {})["diff"] = diff
            full["diff"] = diff
        if outcome:
            if cache and "outcome" in cache:
                outcome = cache["outcome"]
            else:
                outcome = self._summarise(str(outcome))
                if patch_id is not None:
                    self._summary_cache.setdefault(patch_id, {})["outcome"] = outcome
            full["outcome"] = outcome
        if roi_delta is not None:
            full["roi_delta"] = roi_delta
        if lines_changed is not None:
            full["lines_changed"] = lines_changed
        if tests_passed is not None:
            full["tests_passed"] = tests_passed

        if patch_id and PatchHistoryDB is not None:
            cache = self._summary_cache.get(patch_id)
            try:
                rec = PatchHistoryDB().get(patch_id)  # type: ignore[operator]
            except Exception:
                rec = None
            if rec is not None:
                rec_dict = rec.__dict__ if hasattr(rec, "__dict__") else dict(rec)
                desc = rec_dict.get("description") or rec_dict.get("summary") or ""
                diff = rec_dict.get("diff") or ""
                outcome = rec_dict.get("outcome") or ""
                roi_delta = rec_dict.get("roi_delta")
                lines_changed = rec_dict.get("lines_changed")
                tests_passed = rec_dict.get("tests_passed")
                enh_score = rec_dict.get("enhancement_score")
                if enh_score is not None and "enhancement_score" not in full:
                    full["enhancement_score"] = enh_score
                if desc and "summary" not in full:
                    if cache and "summary" in cache:
                        desc = cache["summary"]
                    else:
                        desc = self._summarise(str(desc))
                        self._summary_cache.setdefault(patch_id, {})["summary"] = desc
                    full.setdefault("desc", desc)
                    full["summary"] = desc
                if diff and "diff" not in full:
                    if cache and "diff" in cache:
                        diff = cache["diff"]
                    else:
                        diff = self._truncate_diff(str(diff))
                        diff = self._summarise(diff)
                        self._summary_cache.setdefault(patch_id, {})["diff"] = diff
                    full["diff"] = diff
                if outcome:
                    if cache and "outcome" in cache:
                        outcome = cache["outcome"]
                    else:
                        outcome = self._summarise(str(outcome))
                        self._summary_cache.setdefault(patch_id, {})["outcome"] = outcome
                    full["outcome"] = outcome
                if roi_delta is not None and "roi_delta" not in full:
                    full["roi_delta"] = roi_delta
                if lines_changed is not None and "lines_changed" not in full:
                    full["lines_changed"] = lines_changed
                if tests_passed is not None and "tests_passed" not in full:
                    full["tests_passed"] = tests_passed

        if bucket == "stack":
            desc = full.get("summary") or full.get("desc") or ""
            if desc:
                desc = self._summarise(redact_text(str(desc)))
                full["summary"] = desc
                full["desc"] = desc
            for key in ("path", "repo"):
                val = full.get(key)
                if isinstance(val, str):
                    full[key] = redact_text(val)
            if "language" in full and full["language"] is not None:
                full["language"] = str(full["language"])
            full.pop("content", None)
            full.pop("text", None)

        summary = {
            k: v
            for k, v in full.items()
            if k not in {"win_rate", "regret_rate", "flags"}
        }
        cand = {
            "bucket": bucket,
            "summary": summary,
            "meta": full,
            "raw": meta,
            "score": scored.score,
            "origin": scored.origin,
            "vector_id": scored.vector_id,
            "summarised": False,
        }
        cand["tokens"] = self._count_tokens(
            json.dumps(summary, separators=(",", ":"))
        )
        return cand

    # ------------------------------------------------------------------
    @log_and_measure
    def build_context(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_vectors: bool = False,
        return_metadata: bool = False,
        session_id: str | None = None,
        return_stats: bool = False,
        prioritise: str | None = None,
        exclude_tags: Iterable[str] | None = None,
        exclude_strategies: Iterable[str] | None = None,
        failure: dict | None = None,
        **_: Any,
    ) -> Any:
        """Return a compact JSON context for ``query``.

        Parameters
        ----------
        query:
            Search query used when retrieving vectors.
        top_k:
            Maximum number of entries from each bucket to consider before
            trimming.
        include_vectors:
            When ``True`` the return value includes vector IDs and scores.
        return_metadata:
            When ``True`` the full metadata for each entry is returned.
        prioritise:
            Optional trimming strategy. ``"newest"`` prefers more recent
            entries while ``"roi"`` favours higher ROI vectors.
        exclude_tags:
            Optional iterable of tag strings. Any retrieved vector containing
            one of these tags in its metadata is discarded before ranking.
        exclude_strategies:
            Optional iterable of strategy hashes.  Vectors whose metadata
            ``strategy_hash`` matches any value in this collection are skipped.
        failure:
            Optional parsed failure metadata. When provided the error details
            are appended to the retrieval query to help surface relevant
            context for remediation.

        When ``include_vectors`` is True, the return value is a tuple of
        ``(context_json, session_id, vectors)`` where *vectors* is a list of
        ``(origin, vector_id, score)`` triples.  If ``return_metadata`` is
        enabled, the metadata dictionary is appended as the final element of the
        tuple and contains the full entries including reliability metrics and
        safety flags.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        try:
            self.refresh_db_weights()
        except Exception:
            pass
        _ensure_vector_service()
        dbs_to_check = list(self.db_weights.keys()) or ["code", "bot", "error", "workflow"]
        try:
            ensure_embeddings_fresh(dbs_to_check)
        except StaleEmbeddingsError as exc:
            details = ", ".join(f"{n} ({r})" for n, r in exc.stale_dbs.items())
            logger.error("embeddings missing or stale: %s", details)
            raise VectorServiceError(f"embeddings missing or stale: {details}") from exc
        try:
            self.patch_safety.load_failures()
        except Exception:
            pass

        prompt_tokens = len(query.split())
        if failure:
            parts = [query]
            if failure.error_type:
                parts.append(failure.error_type)
            parts.extend(failure.reproduction_steps)
            query = " ".join(parts)
        query = redact_text(query)
        exclude = set(exclude_tags or [])
        exclude.update(_get_failed_tags())
        exclude_strats = set(exclude_strategies or [])
        exclude_strats.update(getattr(self, "_excluded_failed_strategies", set()))
        cache_key = (query, top_k, tuple(sorted(exclude)), tuple(sorted(exclude_strats)))
        if not include_vectors and not return_metadata and cache_key in self._cache:
            return self._cache[cache_key]

        session_id = session_id or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            hits = self.retriever.search(
                query,
                top_k=top_k * 5,
                session_id=session_id,
                max_alert_severity=self.max_alignment_severity,
            )
        except RateLimitError:
            raise
        except VectorServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorServiceError("retriever failure") from exc
        patch_hits: List[Dict[str, Any]] = []
        if self.patch_retriever is not None:
            try:
                patch_hits = self.patch_retriever.search(query, top_k=top_k)
            except Exception:
                patch_hits = []
        if patch_hits:
            try:
                hits.extend(patch_hits)
            except Exception:
                pass
        self._ensure_stack_ready()
        stack_cfg = _stack_dataset_config()
        stack_hits: List[Dict[str, Any]] = []
        if self.stack_retriever is not None and self.stack_enabled:
            try:
                retrieval_k = self.stack_top_k or getattr(stack_cfg, "retrieval_top_k", 0)
                retrieval_k = max(0, int(retrieval_k))
            except Exception:
                retrieval_k = 0
            if retrieval_k > 0:
                try:
                    stack_hits = self.stack_retriever.retrieve(
                        query, k=retrieval_k, exclude_tags=exclude
                    )
                except Exception:
                    logger.exception("stack retriever failed")
                    stack_hits = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(hits, ErrorResult):
            return "{}"
        if isinstance(hits, FallbackResult):
            logger.debug(
                "retriever returned fallback for %s: %s",
                query,
                getattr(hits, "reason", ""),
            )
            hits = list(hits)

        buckets: Dict[str, List[_ScoredEntry]] = {
            "errors": [],
            "bots": [],
            "workflows": [],
            "enhancements": [],
            "actions": [],
            "information": [],
            "code": [],
            "discrepancies": [],
            "patches": [],
            "stack": [],
        }

        for bundle in hits:
            meta = bundle.get("metadata", {}) or {}
            tags = ()
            try:
                tags = meta.get("tags") or bundle.get("tags") or ()
            except Exception:
                tags = ()
            tag_set = (
                {str(t) for t in tags}
                if isinstance(tags, (list, tuple, set))
                else {str(tags)}
                if tags
                else set()
            )
            strat_hash = meta.get("strategy_hash")
            if exclude and tag_set & exclude:
                continue
            if exclude_strats and strat_hash and str(strat_hash) in exclude_strats:
                continue
            bucket, scored = self._bundle_to_entry(bundle, query)
            if bucket:
                buckets[bucket].append(scored)

        allowed_stack_languages = {lang.lower() for lang in self.stack_languages}
        if not allowed_stack_languages:
            try:
                allowed_stack_languages = {
                    str(lang).lower()
                    for lang in getattr(stack_cfg, "languages", set())
                    if isinstance(lang, str)
                }
            except Exception:
                allowed_stack_languages = set()
        try:
            max_stack_lines = max(0, int(self.stack_max_lines))
        except Exception:
            max_stack_lines = 0
        if not max_stack_lines:
            try:
                max_stack_lines = max(
                    0, int(getattr(stack_cfg, "max_lines", 0))
                )
            except Exception:
                max_stack_lines = 0

        if stack_hits:
            for item in stack_hits:
                bundle = self._normalise_stack_hit(item, query)
                if not bundle:
                    continue
                metadata = dict(bundle.get("metadata") or {})
                language = str(metadata.get("language", "")).lower()
                if allowed_stack_languages and language and language not in allowed_stack_languages:
                    continue
                if language:
                    metadata["language"] = language
                if max_stack_lines:
                    summary = bundle.get("text") or ""
                    lines = summary.splitlines()
                    if len(lines) > max_stack_lines:
                        summary = "\n".join(lines[:max_stack_lines])
                        metadata["summary"] = summary
                        metadata["stack_truncated_lines"] = max_stack_lines
                        bundle["text"] = summary
                bundle["metadata"] = metadata
                bucket, scored = self._bundle_to_entry(bundle, query)
                if bucket:
                    buckets[bucket].append(scored)

        patch_confidence = 0.0
        if buckets["patches"]:
            try:
                patch_db = PatchHistoryDB() if PatchHistoryDB is not None else None
            except Exception:
                patch_db = None
            ranked, patch_confidence = rank_patches(
                buckets["patches"],
                roi_tracker=self.roi_tracker,
                patch_db=patch_db,
                similarity_weight=self.ranking_weight,
                roi_weight=self.roi_weight,
                recency_weight=self.recency_weight,
                exclude_tags=exclude,
            )
            buckets["patches"] = ranked

        # Flatten scored entries and compute token estimates so we can trim
        # globally across buckets.
        bucket_order = list(buckets.keys())
        candidates: List[Dict[str, Any]] = []
        for key in bucket_order:
            items = buckets[key]
            if not items:
                continue
            items.sort(key=lambda e: e.score, reverse=True)
            for e in items[:top_k]:
                cand = self._merge_metadata(e, key)
                candidates.append(cand)

        def estimate_tokens(cands: List[Dict[str, Any]]) -> int:
            ctx: Dict[str, List[Dict[str, Any]]] = {}
            for c in cands:
                ctx.setdefault(c["bucket"], []).append(c["summary"])
            return self._count_tokens(json.dumps(ctx, separators=(",", ":")))
        sum_tokens = sum(c["tokens"] for c in candidates)
        total_tokens = estimate_tokens(candidates)
        overhead = total_tokens - sum_tokens
        total_tokens = sum_tokens + overhead

        if total_tokens > self.max_tokens and candidates:
            if prioritise == "newest":
                candidates.sort(
                    key=lambda c: (
                        c["score"],
                        c["raw"].get("timestamp")
                        or c["raw"].get("ts")
                        or c["raw"].get("created_at")
                        or c["raw"].get("id", 0),
                    )
                )
            elif prioritise == "roi":
                candidates.sort(key=lambda c: (c["score"], c["raw"].get("roi", 0)))
            else:
                candidates.sort(key=lambda c: c["score"])

            idx = 0
            while total_tokens > self.max_tokens and candidates:
                cand = candidates[idx]
                desc = cand["summary"].get("desc", "")
                if not cand["summarised"]:
                    cand["summary"]["desc"] = self._summarise(desc)
                    cand["meta"]["desc"] = cand["summary"]["desc"]
                    cand["meta"]["truncated"] = True
                    cand["summarised"] = True
                    new_tokens = self._count_tokens(
                        json.dumps(cand["summary"], separators=(",", ":"))
                    )
                    sum_tokens += new_tokens - cand["tokens"]
                    cand["tokens"] = new_tokens
                    total_tokens = sum_tokens + overhead
                else:
                    truncated = desc.rsplit(" ", 1)[0] if " " in desc else ""
                    if not truncated or truncated == desc:
                        sum_tokens -= cand["tokens"]
                        candidates.pop(idx)
                        if candidates:
                            sum_tokens = sum(c["tokens"] for c in candidates)
                            overhead = estimate_tokens(candidates) - sum_tokens
                            total_tokens = sum_tokens + overhead
                        else:
                            total_tokens = 0
                            sum_tokens = 0
                            overhead = 0
                    else:
                        cand["summary"]["desc"] = truncated + "..."
                        cand["meta"]["desc"] = cand["summary"]["desc"]
                        cand["meta"]["truncated"] = True
                        new_tokens = self._count_tokens(
                            json.dumps(cand["summary"], separators=(",", ":"))
                        )
                        sum_tokens += new_tokens - cand["tokens"]
                        cand["tokens"] = new_tokens
                        total_tokens = sum_tokens + overhead

        result: Dict[str, List[Dict[str, Any]]] = {}
        meta: Dict[str, List[Dict[str, Any]]] = {}
        vectors: List[Tuple[str, str, float]] = []
        for key in bucket_order:
            for c in candidates:
                if c["bucket"] == key:
                    result.setdefault(key, []).append(c["summary"])
                    if return_metadata:
                        meta.setdefault(key, []).append(c["meta"])
                    vectors.append((c["origin"], c["vector_id"], c["score"]))

        context = json.dumps(result, separators=(",", ":"))
        total_tokens = self._count_tokens(context)
        if not include_vectors and not return_metadata:
            self._cache[cache_key] = context
        stats = {
            "tokens": total_tokens,
            "wall_time_ms": elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "patch_confidence": patch_confidence,
        }
        if include_vectors and return_metadata:
            if return_stats:
                return context, session_id, vectors, meta, stats
            return context, session_id, vectors, meta
        if include_vectors:
            if return_stats:
                return context, session_id, vectors, stats
            return context, session_id, vectors
        if return_metadata:
            if return_stats:
                return context, meta, stats
            return context, meta
        if return_stats:
            return context, stats
        return context

    # ------------------------------------------------------------------
    @log_and_measure
    def build_prompt(
        self,
        query: str,
        *,
        intent: Dict[str, Any] | None = None,
        intent_metadata: Dict[str, Any] | None = None,
        error_log: str | None = None,
        latent_queries: Iterable[str] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> Prompt:
        """Construct a :class:`Prompt` for ``query`` using ``self``.

        Parameters
        ----------
        intent:
            Optional intent metadata fused into the returned prompt's
            ``metadata`` field.  ``intent_metadata`` is accepted as a backwards
            compatible alias.
        """

        if intent is None and intent_metadata is not None:
            intent = intent_metadata
        try:
            return _build_prompt_internal(
                query,
                intent,
                latent_queries=latent_queries,
                top_k=top_k,
                error_log=error_log,
                context_builder=self,
                **kwargs,
            )
        except PromptBuildError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            handle_failure(
                f"failed to build prompt for {query!r}",
                exc,
                logger=logger,
            )

    # ------------------------------------------------------------------
    @log_and_measure
    def build(self, query: str, **kwargs: Any) -> Any:
        """Backward compatible alias for :meth:`build_context`.

        Older modules invoked :meth:`build` on the service layer.  The
        canonical interface is :meth:`build_context`; this wrapper simply
        forwards the call so legacy imports continue to function.
        """

        try:
            return self.build_context(query, **kwargs)
        except RuntimeError as exc:
            raise RuntimeError(
                f"embedding verification failed: {exc}") from exc

    # ------------------------------------------------------------------
    @log_and_measure
    def query(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_vectors: bool = False,
        return_metadata: bool = False,
        session_id: str | None = None,
        return_stats: bool = False,
        prioritise: str | None = None,
        exclude_tags: Iterable[str] | None = None,
        exclude_strategies: Iterable[str] | None = None,
        failure: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for :meth:`build_context` with optional tag exclusion."""

        params = {
            "include_vectors": include_vectors,
            "return_metadata": return_metadata,
            "session_id": session_id,
            "return_stats": return_stats,
            "prioritise": prioritise,
            "exclude_tags": exclude_tags,
            "exclude_strategies": exclude_strategies,
            "failure": failure,
        }
        params.update(kwargs)
        return self.build_context(query, top_k=top_k, **params)

    # ------------------------------------------------------------------
    @log_and_measure
    async def build_async(self, query: str, **kwargs: Any) -> Any:
        """Asynchronous wrapper for :meth:`build_context`."""

        return await asyncio.to_thread(self.build_context, query, **kwargs)

    # ------------------------------------------------------------------
    def enrich_prompt(
        self,
        prompt: Prompt,
        *,
        tags: Iterable[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        origin: str | None = None,
    ) -> Prompt:
        """Merge ``tags`` and ``metadata`` into ``prompt`` in place.

        Parameters
        ----------
        prompt:
            Prompt instance to enrich.
        tags:
            Iterable of tags that should be merged with any existing tags on the
            prompt as well as those stored in metadata.  Values are normalised to
            ``str`` and deduplicated while preserving order.
        metadata:
            Optional metadata to merge into the prompt's metadata dictionary.
            Existing keys are preserved; only missing entries are added.
        origin:
            Optional origin string forced onto the prompt when provided.  When
            omitted the prompt's origin or metadata derived origin is retained
            with a default of ``"context_builder"``.
        """

        if prompt is None:
            raise ValueError("prompt is required for enrichment")

        meta_attr = getattr(prompt, "metadata", None)
        if isinstance(meta_attr, dict):
            meta: Dict[str, Any] = meta_attr
        else:
            meta = dict(meta_attr or {})

        additional_meta = dict(metadata or {})

        def _normalise_tags(source: Any) -> List[str]:
            if not source:
                return []
            if isinstance(source, dict):
                source = source.get("tags", [])
            if isinstance(source, str):
                source_iter: Iterable[Any] = [source]
            elif isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray, str)):
                source_iter = source
            else:
                source_iter = [source]
            values: List[str] = []
            for item in source_iter:
                if isinstance(item, str):
                    text = item.strip()
                else:
                    text = str(item).strip()
                if text and text not in values:
                    values.append(text)
            return values

        base_tags = _normalise_tags(getattr(prompt, "tags", []))
        existing_meta_tags = _normalise_tags(meta.get("tags"))
        new_meta_tags = _normalise_tags(additional_meta.get("tags"))
        extra_tags = _normalise_tags(tags)
        combined_tags = list(
            dict.fromkeys([*base_tags, *existing_meta_tags, *new_meta_tags, *extra_tags])
        )

        for key, value in additional_meta.items():
            if key == "tags":
                continue
            meta.setdefault(key, value)

        if combined_tags or "tags" in meta or "tags" in additional_meta:
            meta["tags"] = list(combined_tags)
            try:
                setattr(prompt, "tags", list(combined_tags))
            except Exception:  # pragma: no cover - defensive assignment
                pass

        try:
            setattr(prompt, "metadata", meta)
        except Exception:  # pragma: no cover - defensive assignment
            pass

        resolved_origin = (
            origin
            or getattr(prompt, "origin", "")
            or meta.get("origin")
            or "context_builder"
        )
        meta.setdefault("origin", resolved_origin)
        if getattr(prompt, "origin", None) != resolved_origin:
            try:
                setattr(prompt, "origin", resolved_origin)
            except Exception:  # pragma: no cover - defensive assignment
                pass

        return prompt


@log_and_measure
def _build_prompt_internal(
    query: str,
    intent: Dict[str, Any] | None = None,
    *,
    latent_queries: Iterable[str] | None = None,
    top_k: int = 5,
    error_log: str | None = None,
    context_builder: "ContextBuilder",
    **kwargs: Any,
) -> Prompt:
    """Build a :class:`Prompt` for ``query``.

    The function retrieves context snippets via :func:`build_context`, performs
    latent query expansion, semantic deduplication and priority scoring before
    fusing results with ``intent`` metadata.  The highest scoring examples are
    compressed to respect the configured token budget.
    """

    intent_meta = intent
    if intent_meta is None and "intent_metadata" in kwargs:
        intent_meta = kwargs.pop("intent_metadata")

    if context_builder is None:
        raise ValueError("context_builder is required")
    builder = context_builder

    if not isinstance(query, str) or not query.strip():
        raise MalformedPromptError("query must be a non-empty string")

    queries: List[str] = [query]
    if latent_queries:
        queries.extend(q for q in latent_queries if isinstance(q, str) and q)
    else:
        expander = getattr(getattr(builder, "memory", None), "expand_intent", None)
        if callable(expander):
            try:
                extra = expander(query)
            except Exception:
                extra = None
            if extra:
                if isinstance(extra, str):
                    queries.append(extra)
                else:
                    queries.extend(str(q) for q in extra if q)

    if error_log:
        queries = [f"{q} {error_log}" for q in queries]

    include_stack = kwargs.pop("include_stack_snippets", None)
    stack_limit_override = kwargs.pop("stack_snippet_limit", None)

    combined_meta: Dict[str, List[Dict[str, Any]]] = {}
    vectors: List[Tuple[str, str, float]] = []
    for q in queries:
        ctx_data = builder.build_context(
            q,
            top_k=top_k,
            include_vectors=True,
            return_metadata=True,
            **kwargs,
        )
        try:
            _ctx, _sid, vecs, meta = ctx_data  # type: ignore[misc]
        except Exception:
            vecs = []
            meta = {}
        for bucket, items in meta.items():
            bucket_items = combined_meta.setdefault(bucket, [])
            for item in items:
                if isinstance(item, dict):
                    item.setdefault("bucket", bucket)
                bucket_items.append(item)
        vectors.extend(vecs)

    if include_stack is None:
        include_stack = getattr(builder, "stack_prompt_enabled", True)
    include_stack = bool(include_stack)
    if stack_limit_override is None:
        stack_limit_override = getattr(builder, "stack_prompt_limit", None)
    try:
        stack_limit = (
            None
            if stack_limit_override is None
            else int(stack_limit_override)
        )
    except Exception:
        stack_limit = 0
    if stack_limit is not None and stack_limit < 0:
        stack_limit = 0
    if not include_stack:
        combined_meta.pop("stack", None)
    else:
        stack_items = combined_meta.get("stack")
        if stack_items:
            stack_items.sort(
                key=lambda item: float((item or {}).get("score") or 0.0),
                reverse=True,
            )
            if stack_limit == 0:
                combined_meta.pop("stack", None)
            elif stack_limit is not None:
                combined_meta["stack"] = stack_items[:stack_limit]

    dedup: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    scores: List[float] = []
    for bucket, items in combined_meta.items():
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                item.update(compress_snippets(item))
            except Exception:
                pass
            desc = str(item.get("desc") or "")
            if not desc:
                continue
            key = re.sub(r"\s+", " ", desc).strip().lower()
            score = float(item.get("score") or 0.0)
            roi = float(item.get("roi") or item.get("roi_delta") or 0.0)
            recency = float(item.get("recency") or 0.0)
            risk = float(item.get("risk_score") or 0.0)
            rec_w = getattr(builder, "recency_weight", 1.0)
            saf_w = getattr(builder, "safety_weight", 1.0)
            priority = (
                score * builder.prompt_score_weight
                + roi * builder.roi_weight
                + recency * rec_w
                - risk * saf_w
            )
            scores.append(score)
            meta_item = dict(item)
            meta_item.setdefault("bucket", bucket)
            cur = dedup.get(key)
            if cur is None or priority > cur[0]:
                dedup[key] = (priority, desc, meta_item)

    ranked = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    examples: List[str] = []
    used = 0
    used_entries: List[Tuple[str, Dict[str, Any]]] = []
    for _, desc, meta_item in ranked:
        tokens = builder._count_tokens(desc)
        if used + tokens > builder.prompt_max_tokens:
            break
        examples.append(desc)
        used_entries.append((desc, meta_item))
        used += tokens

    retrieval_meta: Dict[str, Dict[str, Any]] = {}
    stack_snippet_meta: List[Dict[str, Any]] = []
    for desc, meta_item in used_entries:
        origin = str(meta_item.get("origin") or meta_item.get("bucket") or "")
        vector_id = str(meta_item.get("vector_id") or meta_item.get("record_id") or "")
        key = ":".join([part for part in (origin, vector_id) if part])
        if not key:
            key = f"snippet:{len(retrieval_meta)}"
        prompt_tokens = builder._count_tokens(desc)
        bucket = meta_item.get("bucket") or origin
        entry: Dict[str, Any] = {
            "bucket": bucket,
            "origin": origin,
            "vector_id": vector_id,
            "score": float(meta_item.get("score") or 0.0),
            "prompt_tokens": prompt_tokens,
            "desc": desc,
        }
        for extra_key in (
            "repo",
            "path",
            "language",
            "roi",
            "roi_delta",
            "recency",
            "risk_score",
            "tags",
            "summary",
        ):
            if meta_item.get(extra_key) is not None:
                entry[extra_key] = meta_item.get(extra_key)
        retrieval_meta[key] = entry
        if bucket == "stack":
            stack_snippet_meta.append(
                {
                    "key": key,
                    "repo": meta_item.get("repo"),
                    "path": meta_item.get("path"),
                    "language": meta_item.get("language"),
                    "summary": desc,
                    "score": entry["score"],
                    "prompt_tokens": prompt_tokens,
                }
            )

    avg_conf = sum(scores) / len(scores) if scores else None
    meta_out: Dict[str, Any] = {
        "vector_confidences": scores,
        "vectors": vectors,
    }
    meta_out["stack_snippets_enabled"] = bool(include_stack)
    if stack_limit is not None:
        meta_out["stack_snippet_limit"] = stack_limit
    if retrieval_meta:
        meta_out["retrieval_metadata"] = retrieval_meta
    if stack_snippet_meta:
        meta_out["stack_snippets"] = stack_snippet_meta
    if intent_meta:
        if isinstance(intent_meta, dict):
            meta_out["intent"] = dict(intent_meta)
        else:
            meta_out["intent"] = intent_meta
    if error_log:
        meta_out["error_log"] = error_log

    prompt = Prompt(
        user=query,
        examples=examples,
        vector_confidence=avg_conf,
        metadata=meta_out,
        origin="context_builder",
    )
    return prompt


@log_and_measure
def build_prompt(
    goal: str,
    *,
    intent: Dict[str, Any] | None = None,
    intent_metadata: Dict[str, Any] | None = None,
    latent_queries: Iterable[str] | None = None,
    top_k: int = 5,
    context_builder: "ContextBuilder",
    **kwargs: Any,
) -> Prompt:
    """Build a :class:`Prompt` for ``goal`` using vectorised context.

    The helper retrieves relevant context snippets via
    :meth:`ContextBuilder.build_context`, performs semantic de-duplication and
    priority scoring, merges ``intent`` metadata and delegates construction of
    the final prompt text to :func:`prompt_engine.build_prompt`.
    """

    if intent is None and intent_metadata is not None:
        intent = intent_metadata

    if context_builder is None:
        raise ValueError("context_builder is required")
    builder = context_builder

    queries: List[str] = [goal]
    if latent_queries:
        queries.extend(q for q in latent_queries if isinstance(q, str) and q)

    include_stack = kwargs.pop("include_stack_snippets", None)
    stack_limit_override = kwargs.pop("stack_snippet_limit", None)

    combined_meta: Dict[str, List[Dict[str, Any]]] = {}
    vectors: List[Tuple[str, str, float]] = []
    for q in queries:
        ctx_data = builder.build_context(
            q,
            top_k=top_k,
            include_vectors=True,
            return_metadata=True,
            **kwargs,
        )
        try:
            _ctx, _sid, vecs, meta = ctx_data  # type: ignore[misc]
        except Exception:
            vecs = []
            meta = {}
        for bucket, items in meta.items():
            bucket_items = combined_meta.setdefault(bucket, [])
            for item in items:
                if isinstance(item, dict):
                    item.setdefault("bucket", bucket)
                bucket_items.append(item)
        vectors.extend(vecs)

    if include_stack is None:
        include_stack = getattr(builder, "stack_prompt_enabled", True)
    include_stack = bool(include_stack)
    if stack_limit_override is None:
        stack_limit_override = getattr(builder, "stack_prompt_limit", None)
    try:
        stack_limit = (
            None if stack_limit_override is None else int(stack_limit_override)
        )
    except Exception:
        stack_limit = 0
    if stack_limit is not None and stack_limit < 0:
        stack_limit = 0
    if not include_stack:
        combined_meta.pop("stack", None)
    else:
        stack_items = combined_meta.get("stack")
        if stack_items:
            stack_items.sort(
                key=lambda item: float((item or {}).get("score") or 0.0),
                reverse=True,
            )
            if stack_limit == 0:
                combined_meta.pop("stack", None)
            elif stack_limit is not None:
                combined_meta["stack"] = stack_items[:stack_limit]

    dedup: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    scores: List[float] = []
    for bucket, items in combined_meta.items():
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                item.update(compress_snippets(item))
            except Exception:
                pass
            desc = str(item.get("desc") or "")
            if not desc:
                continue
            key = re.sub(r"\s+", " ", desc).strip().lower()
            score = float(item.get("score") or 0.0)
            roi = float(item.get("roi") or item.get("roi_delta") or 0.0)
            recency = float(item.get("recency") or 0.0)
            risk = float(item.get("risk_score") or 0.0)
            priority = (
                score * builder.prompt_score_weight
                + roi * builder.roi_weight
                + recency * getattr(builder, "recency_weight", 1.0)
                - risk * getattr(builder, "safety_weight", 1.0)
            )
            scores.append(score)
            meta_item = dict(item)
            meta_item.setdefault("bucket", bucket)
            cur = dedup.get(key)
            if cur is None or priority > cur[0]:
                dedup[key] = (priority, desc, meta_item)

    ranked_entries = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    used_entries: List[Tuple[str, Dict[str, Any]]] = []
    retrieval_lines: List[str] = []
    used_tokens = 0
    for _, desc, meta_item in ranked_entries:
        tokens = builder._count_tokens(desc)
        if (
            builder.prompt_max_tokens
            and used_tokens + tokens > builder.prompt_max_tokens
        ):
            break
        retrieval_lines.append(desc)
        used_entries.append((desc, meta_item))
        used_tokens += tokens

    retrieval_context = "\n".join(retrieval_lines)

    from prompt_engine import build_prompt as _pe_build_prompt  # local import to avoid cycle

    prompt = _pe_build_prompt(
        goal,
        retrieval_context=retrieval_context,
        context_builder=builder,
        **kwargs,
    )

    avg_conf = sum(scores) / len(scores) if scores else None
    existing_meta = dict(getattr(prompt, "metadata", {}) or {})
    retrieval_meta: Dict[str, Dict[str, Any]] = {}
    stack_snippet_meta: List[Dict[str, Any]] = []
    for desc, meta_item in used_entries:
        origin = str(meta_item.get("origin") or meta_item.get("bucket") or "")
        vector_id = str(meta_item.get("vector_id") or meta_item.get("record_id") or "")
        key = ":".join([part for part in (origin, vector_id) if part])
        if not key:
            key = f"snippet:{len(retrieval_meta)}"
        prompt_tokens = builder._count_tokens(desc)
        bucket = meta_item.get("bucket") or origin
        entry: Dict[str, Any] = {
            "bucket": bucket,
            "origin": origin,
            "vector_id": vector_id,
            "score": float(meta_item.get("score") or 0.0),
            "prompt_tokens": prompt_tokens,
            "desc": desc,
        }
        for extra_key in (
            "repo",
            "path",
            "language",
            "roi",
            "roi_delta",
            "recency",
            "risk_score",
            "tags",
            "summary",
        ):
            if meta_item.get(extra_key) is not None:
                entry[extra_key] = meta_item.get(extra_key)
        retrieval_meta[key] = entry
        if bucket == "stack":
            stack_snippet_meta.append(
                {
                    "key": key,
                    "repo": meta_item.get("repo"),
                    "path": meta_item.get("path"),
                    "language": meta_item.get("language"),
                    "summary": desc,
                    "score": entry["score"],
                    "prompt_tokens": prompt_tokens,
                }
            )

    meta_out: Dict[str, Any] = {
        "vector_confidences": scores,
        "vectors": vectors,
    }
    meta_out["stack_snippets_enabled"] = bool(include_stack)
    if stack_limit is not None:
        meta_out["stack_snippet_limit"] = stack_limit
    if retrieval_meta:
        meta_out["retrieval_metadata"] = retrieval_meta
    if stack_snippet_meta:
        meta_out["stack_snippets"] = stack_snippet_meta

    combined_meta = dict(meta_out)
    combined_meta.update(existing_meta)
    if "retrieval_metadata" in meta_out:
        merged_retrieval = dict(existing_meta.get("retrieval_metadata", {}))
        merged_retrieval.update(meta_out["retrieval_metadata"])
        combined_meta["retrieval_metadata"] = merged_retrieval
    if "stack_snippets" in meta_out:
        merged_snippets = list(existing_meta.get("stack_snippets", []))
        merged_snippets.extend(meta_out["stack_snippets"])
        combined_meta["stack_snippets"] = merged_snippets
    if "stack_snippets_enabled" in meta_out:
        combined_meta["stack_snippets_enabled"] = meta_out["stack_snippets_enabled"]
    if "stack_snippet_limit" in meta_out:
        combined_meta["stack_snippet_limit"] = meta_out["stack_snippet_limit"]

    if intent:
        merged_intent: Dict[str, Any] = {}
        existing_intent = existing_meta.get("intent")
        if isinstance(existing_intent, dict):
            merged_intent.update(existing_intent)
        elif existing_intent is not None:
            merged_intent["original"] = existing_intent
        merged_intent.update(intent)
        combined_meta["intent"] = merged_intent
    elif "intent" in existing_meta:
        combined_meta["intent"] = existing_meta["intent"]

    prompt.metadata = combined_meta

    if avg_conf is not None:
        current_conf = getattr(prompt, "vector_confidence", None)
        if current_conf is None:
            try:
                prompt.vector_confidence = avg_conf
            except Exception:
                prompt.metadata.setdefault("vector_confidence", avg_conf)
        else:
            prompt.metadata.setdefault("vector_confidence", avg_conf)

    if not getattr(prompt, "origin", None):
        prompt.origin = "context_builder"
    else:
        prompt.metadata.setdefault("origin", prompt.origin)
    return prompt


__all__ = ["ContextBuilder", "record_failed_tags", "load_failed_tags", "build_prompt"]


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _warn_once(flag: str, message: str) -> None:
    if flag in _STACK_WARNED_FLAGS:
        return
    logger.warning(message)
    _STACK_WARNED_FLAGS.add(flag)


def _resolve_hf_token() -> str | None:
    for key in _HF_ENV_KEYS:
        token = os.environ.get(key)
        if token:
            if key != "HUGGINGFACE_TOKEN":
                os.environ.setdefault("HUGGINGFACE_TOKEN", token)
            return token
    return None
