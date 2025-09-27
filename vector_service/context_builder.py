"""Cross-database context builder used by language model prompts."""

from __future__ import annotations

import json
import re
import contextlib
from dataclasses import dataclass, fields, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Mapping
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
from .retriever import Retriever, PatchRetriever, FallbackResult, StackRetriever
from config import ContextBuilderConfig
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

try:  # pragma: no cover - optional dependency
    from .stack_ingestion import (
        ensure_background_task as _ensure_stack_background,
        StackDatasetStreamer as _StackDatasetStreamer,
    )
except Exception:  # pragma: no cover - stack ingestion optional
    _ensure_stack_background = None  # type: ignore
    _StackDatasetStreamer = None  # type: ignore

# Alias retained for backward compatibility with tests expecting
# ``UniversalRetriever`` to be injectable.
UniversalRetriever = Retriever

logger = logging.getLogger(__name__)


def _to_dict(value: Any) -> Dict[str, Any]:
    """Best-effort conversion to a dictionary for config-like objects."""

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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


@dataclass
class _StackContextConfig:
    """Configuration controlling Stack dataset retrieval within prompts."""

    enabled: bool = False
    top_k: int = 3
    summary_tokens: int = 160
    text_max_tokens: int = 320
    weight: float = 1.0
    penalty: float = 0.0
    ingestion_enabled: bool = True
    ingestion_throttle_seconds: float = 600.0
    ingestion_batch_limit: int | None = None
    ensure_before_search: bool = True
    languages: Tuple[str, ...] = field(default_factory=tuple)
    max_lines: int | None = None
    chunk_overlap: int | None = None
    streaming: bool | None = None
    cache_dir: str | None = None
    index_path: str | None = None
    metadata_path: str | None = None
    document_cache: str | None = None

    @classmethod
    def from_any(cls, value: Any) -> "_StackContextConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        value_dict = _to_dict(value)
        data: Dict[str, Any] = {}
        for field_def in fields(cls):
            if field_def.name in value_dict and value_dict[field_def.name] is not None:
                data[field_def.name] = value_dict[field_def.name]

        ingestion = _to_dict(value_dict.get("ingestion"))
        retrieval = _to_dict(value_dict.get("retrieval"))
        cache_cfg = _to_dict(value_dict.get("cache"))

        if "languages" not in data and ingestion.get("languages") is not None:
            data["languages"] = ingestion.get("languages")
        if "max_lines" not in data and ingestion.get("max_document_lines") is not None:
            data["max_lines"] = ingestion.get("max_document_lines")
        if "chunk_overlap" not in data and ingestion.get("chunk_overlap") is not None:
            data["chunk_overlap"] = ingestion.get("chunk_overlap")
        if "streaming" not in data and ingestion.get("streaming") is not None:
            data["streaming"] = ingestion.get("streaming")

        for key in ("top_k", "weight"):
            if key not in data and retrieval.get(key) is not None:
                data[key] = retrieval.get(key)

        if "cache_dir" not in data and cache_cfg.get("data_dir"):
            data["cache_dir"] = cache_cfg.get("data_dir")
        if "index_path" not in data and cache_cfg.get("index_path"):
            data["index_path"] = cache_cfg.get("index_path")
        if "metadata_path" not in data and cache_cfg.get("metadata_path"):
            data["metadata_path"] = cache_cfg.get("metadata_path")
        if "document_cache" not in data and cache_cfg.get("document_cache"):
            data["document_cache"] = cache_cfg.get("document_cache")

        langs = data.get("languages")
        if langs is not None:
            if isinstance(langs, (list, tuple, set)):
                data["languages"] = tuple(
                    str(lang).strip() for lang in langs if str(lang).strip()
                )
            elif isinstance(langs, str):
                parts = [part.strip() for part in langs.split(",") if part.strip()]
                data["languages"] = tuple(parts)
            else:
                data["languages"] = tuple()
        if "max_lines" in data and data["max_lines"] is not None:
            try:
                data["max_lines"] = int(data["max_lines"])
            except Exception:
                data["max_lines"] = None
        if "chunk_overlap" in data and data["chunk_overlap"] is not None:
            try:
                data["chunk_overlap"] = int(data["chunk_overlap"])
            except Exception:
                data["chunk_overlap"] = None
        if "streaming" in data and data["streaming"] is not None:
            data["streaming"] = bool(data["streaming"])
        for key in ("cache_dir", "index_path", "metadata_path", "document_cache"):
            if key in data and data[key] is not None:
                data[key] = str(data[key])
        return cls(**data)

    def cache_marker(self) -> Tuple[Any, ...]:
        return (
            self.enabled,
            self.top_k,
            self.summary_tokens,
            self.text_max_tokens,
            self.weight,
            self.penalty,
            tuple(self.languages),
            self.max_lines,
            self.chunk_overlap,
            self.streaming,
            self.cache_dir,
            self.index_path,
            self.metadata_path,
            self.document_cache,
        )


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        patch_retriever: PatchRetriever | None = None,
        ranking_model: Any | None = None,
        roi_tracker: Any | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        summariser: Callable[[str], str] | None = None,
        db_weights: Dict[str, float] | None = None,
        ranking_weight: float = ContextBuilderConfig().ranking_weight,
        roi_weight: float = ContextBuilderConfig().roi_weight,
        recency_weight: float = ContextBuilderConfig().recency_weight,
        safety_weight: float = ContextBuilderConfig().safety_weight,
        max_tokens: int = ContextBuilderConfig().max_tokens,
        regret_penalty: float = ContextBuilderConfig().regret_penalty,
        alignment_penalty: float = ContextBuilderConfig().alignment_penalty,
        alert_penalty: float = ContextBuilderConfig().alert_penalty,
        risk_penalty: float = ContextBuilderConfig().risk_penalty,
        roi_tag_penalties: Dict[str, float] = ContextBuilderConfig().roi_tag_penalties,
        enhancement_weight: float = ContextBuilderConfig().enhancement_weight,
        max_alignment_severity: float = getattr(
            ContextBuilderConfig(), "max_alignment_severity", 1.0
        ),
        max_alerts: int = getattr(ContextBuilderConfig(), "max_alerts", 5),
        license_denylist: set[str] | None = getattr(
            ContextBuilderConfig(), "license_denylist", _DEFAULT_LICENSE_DENYLIST
        ),
        precise_token_count: bool = getattr(
            ContextBuilderConfig(), "precise_token_count", True
        ),
        max_diff_lines: int = getattr(ContextBuilderConfig(), "max_diff_lines", 200),
        patch_safety: PatchSafety | None = None,
        similarity_metric: str = getattr(ContextBuilderConfig(), "similarity_metric", "cosine"),
        embedding_check_interval: float = getattr(
            ContextBuilderConfig(), "embedding_check_interval", 0
        ),
        prompt_score_weight: float = getattr(
            ContextBuilderConfig(), "prompt_score_weight", 1.0
        ),
        prompt_max_tokens: int = getattr(
            ContextBuilderConfig(), "prompt_max_tokens", 800
        ),
        stack_retriever: Any | None = None,
        stack_ingestor: Any | None = None,
        stack_config: Any | None = None,
        stack_score_weight: float | None = None,
        stack_penalty_weight: float | None = None,
    ) -> None:
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
        self.max_tokens = max_tokens
        self.precise_token_count = precise_token_count
        self.max_diff_lines = max_diff_lines
        self.patch_safety = patch_safety or PatchSafety()
        self.patch_safety.max_alert_severity = max_alignment_severity
        self.patch_safety.max_alerts = max_alerts
        self.patch_safety.license_denylist = self.license_denylist
        self.prompt_score_weight = prompt_score_weight
        self.prompt_max_tokens = prompt_max_tokens
        cfg_obj = stack_config
        if cfg_obj is None:
            cfg = ContextBuilderConfig()
            cfg_obj = getattr(cfg, "stack", None)
            if cfg_obj is None:
                cfg_obj = getattr(cfg, "stack_dataset", None)
        self.stack_config = _StackContextConfig.from_any(cfg_obj)
        if stack_score_weight is not None:
            try:
                self.stack_config.weight = float(stack_score_weight)
            except Exception:
                logger.exception("invalid stack_score_weight override")
        if stack_penalty_weight is not None:
            try:
                self.stack_config.penalty = float(stack_penalty_weight)
            except Exception:
                logger.exception("invalid stack_penalty_weight override")
        # Environment overrides allow runtime experimentation without config
        # reloads.  ``STACK_CONTEXT_ENABLED`` takes precedence over the value
        # in configuration, while ``STACK_CONTEXT_DISABLED`` always forces the
        # integration off.
        if "STACK_CONTEXT_ENABLED" in os.environ:
            self.stack_config.enabled = _env_flag(
                "STACK_CONTEXT_ENABLED", self.stack_config.enabled
            )
        if _env_flag("STACK_CONTEXT_DISABLED", False):
            self.stack_config.enabled = False
        if "STACK_CONTEXT_INGESTION" in os.environ:
            self.stack_config.ingestion_enabled = _env_flag(
                "STACK_CONTEXT_INGESTION", self.stack_config.ingestion_enabled
            )
        if _env_flag("STACK_CONTEXT_INGESTION_DISABLED", False):
            self.stack_config.ingestion_enabled = False
        if self.stack_config.enabled:
            self.db_weights.setdefault("stack", getattr(self.stack_config, "weight", 1.0))
        self._stack_retriever: Any | None = stack_retriever
        self._stack_ingestor: Any | None = stack_ingestor
        self._stack_retriever_factory: Callable[[], Any] | None = None
        self._stack_ingestor_factory: Callable[[], Any] | None = None
        if self.stack_config.enabled and self._stack_retriever is None:
            self._stack_retriever_factory = lambda: StackRetriever(
                top_k=max(int(self.stack_config.top_k or 0), 0),
                similarity_threshold=0.0,
                max_alert_severity=self.max_alignment_severity,
                max_alerts=self.max_alerts,
                license_denylist=self.license_denylist,
                roi_tag_weights=self.roi_tag_penalties,
            )
        if self.stack_config.enabled and self._stack_ingestor is None:
            if _StackDatasetStreamer is not None:
                self._stack_ingestor_factory = (
                    lambda: _StackDatasetStreamer.from_environment()
                )
        self._last_stack_ingest: float = 0.0

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

        if _ensure_stack_background is not None:
            try:
                _ensure_stack_background()
            except Exception:
                logger.exception("failed to start stack ingestion background task")

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

    def _trim_tokens(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        original_tokens = text.split()
        if not original_tokens:
            return text
        tokens = list(original_tokens)
        while tokens and self._count_tokens(" ".join(tokens)) > limit:
            tokens.pop()
        if not tokens:
            return ""
        truncated = " ".join(tokens)
        if len(tokens) < len(original_tokens):
            truncated = truncated.rstrip() + "..."
        return truncated

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
            elif origin == "stack":
                score = meta.get("score")
                if score is not None:
                    try:
                        metric = float(score) * getattr(self.stack_config, "weight", 1.0)
                    except Exception:
                        metric = float(score)
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
        elif origin == "stack":
            text = text or meta.get("summary") or meta.get("desc") or meta.get("snippet") or ""
            if "repo" in meta:
                entry["repository"] = redact_text(str(meta.get("repo")))
            if "path" in meta:
                entry["path"] = redact_text(str(meta.get("path")))
            if "language" in meta:
                entry["language"] = str(meta.get("language"))
            snippet_val = meta.get("snippet") or text
            if snippet_val:
                entry["snippet"] = redact_text(str(snippet_val))
            if meta.get("license"):
                entry.setdefault("flags", {})
                entry["flags"]["license"] = meta.get("license")
            if meta.get("license_fingerprint"):
                entry.setdefault("flags", {})
                entry["flags"]["license_fingerprint"] = meta.get("license_fingerprint")

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
        if origin == "stack":
            weight = getattr(self.stack_config, "weight", 1.0)
            penalty_bias = getattr(self.stack_config, "penalty", 0.0)
            score = float(bundle.get("score", score)) * weight - penalty_bias

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
        full.setdefault("origin_db", scored.origin)

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
    def _get_stack_retriever(self) -> Any | None:
        if not self.stack_config.enabled:
            return None
        if self._stack_retriever is None and self._stack_retriever_factory is not None:
            try:
                self._stack_retriever = self._stack_retriever_factory()
            except Exception:
                logger.exception("failed to initialise stack retriever")
                self._stack_retriever = None
        retriever = self._stack_retriever
        if retriever is not None:
            try:
                retriever.top_k = max(int(self.stack_config.top_k or 0), 0)
            except Exception:
                pass
            try:
                retriever.max_alert_severity = self.max_alignment_severity
            except Exception:
                pass
            try:
                retriever.max_alerts = self.max_alerts
            except Exception:
                pass
            try:
                retriever.license_denylist = set(self.license_denylist)
            except Exception:
                pass
            try:
                retriever.roi_tag_weights = dict(self.roi_tag_penalties)
            except Exception:
                pass
        return retriever

    def _apply_stack_preferences(
        self, preferences: Mapping[str, Any] | None
    ) -> Callable[[], None] | None:
        if not preferences:
            return None
        cfg = getattr(self, "stack_config", None)
        if cfg is None:
            return None
        data = _to_dict(preferences)
        if not data:
            return None
        original: Dict[str, Any] = {}
        mutated = False

        def _store(attr: str, value: Any) -> None:
            nonlocal mutated
            if attr not in original:
                original[attr] = getattr(cfg, attr, None)
            try:
                setattr(cfg, attr, value)
                mutated = True
            except Exception:
                pass

        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            if key in {"enabled", "ingestion_enabled", "ensure_before_search"}:
                _store(key, bool(value))
            elif key in {"top_k", "summary_tokens", "text_max_tokens", "max_lines"}:
                if value is None:
                    _store(key, None)
                else:
                    try:
                        _store(key, int(value))
                    except Exception:
                        continue
            elif key == "languages":
                if value is None:
                    _store(key, tuple())
                elif isinstance(value, str):
                    parts = [part.strip() for part in value.split(",") if part.strip()]
                    _store(key, tuple(parts))
                elif isinstance(value, (list, tuple, set)):
                    langs = tuple(
                        str(lang).strip() for lang in value if str(lang).strip()
                    )
                    _store(key, langs)
                else:
                    continue
            else:
                _store(key, value)

        if not mutated:
            return None

        def restore() -> None:
            for attr, value in original.items():
                with contextlib.suppress(Exception):
                    setattr(cfg, attr, value)

        return restore

    def _get_stack_ingestor(self) -> Any | None:
        if self._stack_ingestor is None and self._stack_ingestor_factory is not None:
            try:
                self._stack_ingestor = self._stack_ingestor_factory()
            except Exception:
                logger.exception("failed to initialise stack ingestor")
                self._stack_ingestor = None
        return self._stack_ingestor

    def _get_query_embedding(self, query: str) -> List[float] | None:
        embedder = getattr(self.retriever, "embed_query", None)
        if callable(embedder):
            try:
                vec = embedder(query)
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                if isinstance(vec, (list, tuple)):
                    return [float(x) for x in vec]
            except Exception:
                logger.exception("stack embed_query failed")
        base = getattr(self.retriever, "_get_retriever", None)
        if callable(base):
            try:
                inner = base()
            except Exception:
                inner = None
            if inner is not None:
                for attr in ("embed_query", "embed_text", "embed"):
                    fn = getattr(inner, attr, None)
                    if not callable(fn):
                        continue
                    try:
                        vec = fn(query)
                        if isinstance(vec, tuple):
                            vec = vec[0]
                        if hasattr(vec, "tolist"):
                            vec = vec.tolist()
                        if isinstance(vec, (list, tuple)):
                            return [float(x) for x in vec]
                    except Exception:
                        logger.exception("stack embedding via %s failed", attr)
        return None

    def _prepare_stack_bundle(self, hit: Dict[str, Any], *, index: int) -> Dict[str, Any]:
        meta = dict(hit.get("metadata", {}) or {})
        repo = (
            hit.get("repo")
            or meta.get("repo")
            or meta.get("repository")
            or meta.get("owner")
        )
        path = hit.get("path") or meta.get("path") or meta.get("file_path")
        language = hit.get("language") or meta.get("language") or meta.get("lang")
        license_name = hit.get("license") or meta.get("license")
        license_fp = hit.get("license_fingerprint") or meta.get("license_fingerprint")
        snippet_source = (
            hit.get("snippet")
            or meta.get("snippet")
            or hit.get("text")
            or meta.get("text")
            or meta.get("content")
            or ""
        )
        snippet_source = redact_text(str(snippet_source))
        if not snippet_source.strip():
            snippet_source = ""
        summary_source = (
            hit.get("summary") or meta.get("summary") or snippet_source
        )
        summary_source = redact_text(str(summary_source))
        if not summary_source.strip():
            summary_source = snippet_source
        limit = max(int(self.stack_config.summary_tokens or 0), 0)
        desc = summary_source
        tokens = self._count_tokens(desc) if desc else 0
        if limit and tokens > limit:
            desc = self._summarise(desc)
            tokens = self._count_tokens(desc)
        if limit and tokens > limit:
            desc = self._trim_tokens(desc, limit)
            tokens = self._count_tokens(desc)
        text_limit = max(int(self.stack_config.text_max_tokens or 0), 0)
        if text_limit and tokens > text_limit:
            desc = self._trim_tokens(desc, text_limit)
            tokens = self._count_tokens(desc)
        meta.setdefault("summary", desc)
        meta.setdefault("desc", desc)
        if snippet_source and snippet_source != desc:
            snippet_tokens = self._count_tokens(snippet_source)
            if text_limit and snippet_tokens > text_limit:
                snippet_source = self._trim_tokens(snippet_source, text_limit)
            meta.setdefault("snippet", snippet_source)
        else:
            meta.setdefault("snippet", desc)
        if repo:
            meta.setdefault("repo", repo)
            meta.setdefault("repository", repo)
        if path:
            meta.setdefault("path", path)
            meta.setdefault("file_path", path)
        if language:
            meta.setdefault("language", language)
        score = hit.get("score")
        if score is None:
            score = meta.get("score")
        try:
            score_val = float(score or 0.0)
        except Exception:
            score_val = 0.0
        meta["score"] = score_val
        meta.setdefault("source", "stack")
        meta["stack_tokens"] = tokens
        flags = dict(meta.get("flags") or {})
        if license_name:
            meta["license"] = license_name
            flags.setdefault("license", license_name)
        if license_fp:
            meta["license_fingerprint"] = license_fp
            flags.setdefault("license_fingerprint", license_fp)
        if flags:
            meta["flags"] = flags
        identifier = (
            hit.get("identifier")
            or meta.get("identifier")
            or meta.get("checksum")
            or hit.get("checksum")
            or f"stack-{index}"
        )
        bundle = {
            "origin_db": "stack",
            "record_id": identifier,
            "score": score_val,
            "text": desc,
            "metadata": meta,
        }
        return bundle

    def _retrieve_stack_hits(self, query: str) -> List[Dict[str, Any]]:
        retriever = self._get_stack_retriever()
        if retriever is None:
            return []
        embedding = self._get_query_embedding(query)
        if embedding is None:
            return []
        try:
            hits = retriever.retrieve(
                embedding,
                k=self.stack_config.top_k,
                languages=self.stack_config.languages,
                max_lines=self.stack_config.max_lines,
            )
        except Exception:
            logger.exception("stack retriever failure")
            return []
        bundles: List[Dict[str, Any]] = []
        for idx, hit in enumerate(hits):
            if not isinstance(hit, dict):
                continue
            try:
                bundles.append(self._prepare_stack_bundle(hit, index=idx))
            except Exception:
                logger.exception("failed to prepare stack bundle")
        return bundles

    def _deduplicate_bundles(
        self, bundles: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            origin = str(bundle.get("origin_db") or bundle.get("origin") or "")
            meta = bundle.get("metadata") or {}
            meta_dict = meta if isinstance(meta, dict) else {}
            key_candidates: List[str] = []
            identifiers = [
                bundle.get("record_id"),
                meta_dict.get("record_id"),
                meta_dict.get("id"),
                meta_dict.get("uuid"),
                meta_dict.get("identifier"),
                meta_dict.get("checksum"),
            ]
            if origin == "stack":
                repo = meta_dict.get("repo") or meta_dict.get("repository")
                path = meta_dict.get("path") or meta_dict.get("file_path")
                if repo or path:
                    identifiers.append((repo or "", path or ""))
            for candidate in identifiers:
                if not candidate:
                    continue
                try:
                    if isinstance(candidate, (dict, list, tuple, set)):
                        normalised = json.dumps(candidate, sort_keys=True, default=str)
                    else:
                        normalised = str(candidate)
                except Exception:
                    normalised = str(candidate)
                key_candidates.append(f"{origin}:{normalised}")
            if not key_candidates:
                fallback = (
                    bundle.get("text")
                    or meta_dict.get("summary")
                    or meta_dict.get("desc")
                    or meta_dict.get("snippet")
                    or ""
                )
                try:
                    if isinstance(fallback, (dict, list, tuple, set)):
                        fallback_norm = json.dumps(fallback, sort_keys=True, default=str)
                    else:
                        fallback_norm = str(fallback)
                except Exception:
                    fallback_norm = str(fallback)
                key_candidates.append(f"{origin}:{fallback_norm}")
            if any(key in seen for key in key_candidates):
                continue
            for key in key_candidates:
                seen.add(key)
            deduped.append(bundle)
        return deduped

    # ------------------------------------------------------------------
    def ensure_stack_index_up_to_date(self, *, force: bool = False) -> bool:
        """Kick opportunistic Stack dataset ingestion when throttle permits.

        The ingestion helpers can run either in-process via the lazily
        instantiated ingestor or by delegating to the shared background task.
        The ``force`` flag is primarily used by tests but also provides a
        manual override for administrative tooling so operators can bypass the
        throttle window when recovering from outages.
        """
        if not self.stack_config.enabled or not self.stack_config.ingestion_enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_stack_ingest < self.stack_config.ingestion_throttle_seconds:
            return False
        ingestor = self._get_stack_ingestor()
        triggered = False
        if ingestor is not None:
            ensure_fn = getattr(ingestor, "ensure_index_up_to_date", None)
            process_fn = getattr(ingestor, "process", None)
            try:
                if callable(ensure_fn):
                    triggered = bool(ensure_fn(self.stack_config))
                elif callable(process_fn):
                    process_fn(limit=self.stack_config.ingestion_batch_limit)
                    triggered = True
            except Exception:
                logger.exception("stack ingestion failed")
        elif _ensure_stack_background is not None:
            try:
                triggered = bool(_ensure_stack_background())
            except Exception:
                logger.exception("stack background ensure failed")
        if triggered:
            self._last_stack_ingest = now
        return triggered

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
        stack_preferences: Mapping[str, Any] | None = None,
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

        restore_stack_config = None
        if stack_preferences:
            restore_stack_config = self._apply_stack_preferences(stack_preferences)

        try:
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
            cache_key = (
                query,
                top_k,
                tuple(sorted(exclude)),
                tuple(sorted(exclude_strats)),
                self.stack_config.cache_marker(),
            )
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
            stack_stats: Dict[str, Any] = {
                "enabled": bool(self.stack_config.enabled),
                "hits": 0,
                "ingestion_triggered": False,
            }
            stack_bundles: List[Dict[str, Any]] = []
            if self.stack_config.enabled:
                try:
                    if self.stack_config.ensure_before_search:
                        stack_stats["ingestion_triggered"] = self.ensure_stack_index_up_to_date()
                except Exception:
                    stack_stats["ingestion_triggered"] = False
                stack_bundles = self._retrieve_stack_hits(query)
                stack_stats["hits"] = len(stack_bundles)
                if stack_bundles:
                    hits.extend(stack_bundles)
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
            elif not isinstance(hits, list):
                hits = list(hits)

            hits = self._deduplicate_bundles(hits)

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
                if self.stack_config.enabled:
                    selected = [c for c in candidates if c["bucket"] == "stack"]
                    stack_stats["selected"] = len(selected)
                    stack_stats["tokens"] = sum(c["tokens"] for c in selected)
                    stats["stack"] = stack_stats
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
        finally:
            if restore_stack_config:
                restore_stack_config()

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
    def run_stack_ingestion(
        self, *, limit: int | None = None, continuous: bool = False
    ) -> int:
        """Trigger Stack dataset ingestion using current configuration."""

        if _StackDatasetStreamer is None:
            raise RuntimeError("stack ingestion unavailable")
        streamer = _StackDatasetStreamer.from_environment()
        return streamer.process(limit=limit, continuous=continuous)

    async def run_stack_ingestion_async(
        self, *, limit: int | None = None, continuous: bool = False
    ) -> int:
        if _StackDatasetStreamer is None:
            raise RuntimeError("stack ingestion unavailable")
        streamer = _StackDatasetStreamer.from_environment()
        return await streamer.process_async(limit=limit, continuous=continuous)

    def start_stack_ingestion_background(self) -> bool:
        if _ensure_stack_background is None:
            raise RuntimeError("stack ingestion unavailable")
        return _ensure_stack_background()

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
            combined_meta.setdefault(bucket, []).extend(items)
        vectors.extend(vecs)

    dedup: Dict[str, Tuple[float, str]] = {}
    scores: List[float] = []
    for items in combined_meta.values():
        for item in items:
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
            cur = dedup.get(key)
            if cur is None or priority > cur[0]:
                dedup[key] = (priority, desc)

    ranked = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    examples: List[str] = []
    used = 0
    for _, desc in ranked:
        tokens = builder._count_tokens(desc)
        if used + tokens > builder.prompt_max_tokens:
            break
        examples.append(desc)
        used += tokens

    avg_conf = sum(scores) / len(scores) if scores else None
    meta_out: Dict[str, Any] = {
        "vector_confidences": scores,
        "vectors": vectors,
    }
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

    combined_meta: Dict[str, List[Dict[str, Any]]] = {}
    vectors: List[Tuple[str, str, float]] = []
    scores: List[float] = []
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
            combined_meta.setdefault(bucket, []).extend(items)
        vectors.extend(vecs)
        for items in meta.values():
            for item in items:
                try:
                    scores.append(float(item.get("score") or 0.0))
                except Exception:
                    pass

    dedup: Dict[str, Tuple[float, str]] = {}
    for items in combined_meta.values():
        for item in items:
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
            cur = dedup.get(key)
            if cur is None or priority > cur[0]:
                dedup[key] = (priority, desc)

    ranked = [desc for _, desc in sorted(dedup.values(), key=lambda x: x[0], reverse=True)]
    retrieval_context = "\n".join(ranked)

    from prompt_engine import build_prompt as _pe_build_prompt  # local import to avoid cycle

    prompt = _pe_build_prompt(
        goal,
        retrieval_context=retrieval_context,
        context_builder=builder,
        **kwargs,
    )

    avg_conf = sum(scores) / len(scores) if scores else None
    existing_meta = dict(getattr(prompt, "metadata", {}) or {})
    meta_out: Dict[str, Any] = {
        "vector_confidences": scores,
        "vectors": vectors,
    }
    combined_meta = dict(meta_out)
    combined_meta.update(existing_meta)

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
