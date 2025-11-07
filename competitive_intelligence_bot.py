"""Competitive Intelligence Bot for rival monitoring and analysis."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import sqlite3
import logging
import os
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Any, Dict
import re
import math
import threading

from .retry_utils import with_retry
from security.secret_redactor import redact
import license_detector
from analysis.semantic_diff_filter import find_semantic_risks
from governed_embeddings import governed_embed, get_embedder
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

try:  # optional dependency for embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


_CONFIG_PATH = os.getenv(
    "CI_CONFIG_PATH", "config/competitive_intelligence.json"
)
_REQUIRED_CFG_KEYS: Dict[str, type] = {
    "positive": list,
    "negative": list,
    "ai_keywords": list,
    "entity_blacklist": list,
}


def _load_fallback_config(path: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if not Path(path).is_file():
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # pragma: no cover - optional
        logger.exception("failed loading CI config %s: %s", path, exc)
        return {}
    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        logger.warning("CI config missing keys: %s", ", ".join(missing))
    return cfg


_CFG = _load_fallback_config(_CONFIG_PATH)


def _parse_env_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def _parse_env_dict(val: str | None) -> Dict[str, float]:
    if not val:
        return {}
    result: Dict[str, float] = {}
    for part in val.split(","):
        if not part.strip():
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                result[k.strip()] = float(v)
            except ValueError:
                continue
        else:
            result[part.strip()] = 1.0
    return result


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Return cosine similarity between vectors ``a`` and ``b``."""

    vec_a = list(a)
    vec_b = list(b)
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _normalize_timestamp(ts: Any) -> str:
    """Return ISO timestamp without microseconds."""
    if isinstance(ts, (int, float)):
        dt = datetime.utcfromtimestamp(float(ts))
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return str(ts)
    return dt.replace(microsecond=0).isoformat()


_POSITIVE = set(
    _parse_env_list(os.getenv("CI_POSITIVE")) or _CFG.get("positive", [])
)
_NEGATIVE = set(
    _parse_env_list(os.getenv("CI_NEGATIVE")) or _CFG.get("negative", [])
)
_AI_KEYWORDS = _parse_env_list(os.getenv("CI_AI_KEYWORDS")) or _CFG.get(
    "ai_keywords",
    [],
)

_SENTIMENT_WEIGHTS: Dict[str, float] = _parse_env_dict(
    os.getenv("CI_SENTIMENT_WEIGHTS")
) or _CFG.get("sentiment_weights", {})

# Configurable embedding model and detection parameters
_EMBED_MODEL = os.getenv(
    "CI_EMBED_MODEL", _CFG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
)


def _parse_float(val: str | None, default: float) -> float:
    try:
        if val is None:
            raise ValueError
        return float(val)
    except Exception:
        logger.warning("invalid CI_AI_THRESHOLD %r, using %s", val, default)
        return default


_AI_THRESHOLD = _parse_float(
    os.getenv("CI_AI_THRESHOLD"),
    float(_CFG.get("ai_threshold", 0.6)),
)
_EMBED_STRATEGY = os.getenv(
    "CI_EMBED_STRATEGY", _CFG.get("embed_strategy", "max")
)
_EMBED_TOPK = int(os.getenv("CI_EMBED_TOPK", str(_CFG.get("embed_top_k", 3))))
_ENTITY_BLACKLIST = set(
    _parse_env_list(os.getenv("CI_ENTITY_BLACKLIST"))
    or _CFG.get("entity_blacklist", [])
)

_RETENTION_DAYS = int(
    os.getenv("CI_RETENTION_DAYS", str(_CFG.get("retention_days", 365)))
)

_STRICT_MODE = os.getenv(
    "CI_STRICT_MODE", str(_CFG.get("strict_mode", "false"))
).lower() in {"1", "true", "yes"}

_SENTIMENT_MODEL = None
if pipeline is not None:
    try:
        _SENTIMENT_MODEL = pipeline("sentiment-analysis")
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("sentiment model init failed: %s", exc)
        _SENTIMENT_MODEL = None

try:
    _CI_EMBEDDER_TIMEOUT = float(os.getenv("CI_EMBEDDER_TIMEOUT", "5"))
except Exception:
    logger.warning("invalid CI_EMBEDDER_TIMEOUT; defaulting to 5s")
    _CI_EMBEDDER_TIMEOUT = 5.0
else:
    if _CI_EMBEDDER_TIMEOUT < 0:
        logger.warning("CI_EMBEDDER_TIMEOUT must be non-negative; defaulting to 5s")
        _CI_EMBEDDER_TIMEOUT = 5.0

_EMBEDDER: "SentenceTransformer | None" = None
_AI_EMBEDDINGS: List[List[float]] | None = None
_EMBEDDER_LOCK = threading.Lock()


def _ensure_ai_embeddings() -> tuple["SentenceTransformer | None", List[List[float]] | None]:
    """Initialise and cache embeddings for AI keyword heuristics."""

    global _EMBEDDER, _AI_EMBEDDINGS
    if _EMBEDDER is not None and _AI_EMBEDDINGS is not None:
        return _EMBEDDER, _AI_EMBEDDINGS

    if SentenceTransformer is None:
        return None, None

    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None and _AI_EMBEDDINGS is not None:
            return _EMBEDDER, _AI_EMBEDDINGS

        embedder = _EMBEDDER or get_embedder(timeout=_CI_EMBEDDER_TIMEOUT)
        if embedder is None:
            _EMBEDDER = None
            _AI_EMBEDDINGS = None
            return None, None

        _EMBEDDER = embedder
        if _AI_EMBEDDINGS is None:
            embeddings: List[List[float]] = []
            for kw in _AI_KEYWORDS:
                vec = governed_embed(kw, embedder)
                if vec is not None:
                    embeddings.append(vec)
            _AI_EMBEDDINGS = embeddings
            if not _AI_EMBEDDINGS:
                logger.debug("CI embedder produced no keyword embeddings")
        return _EMBEDDER, _AI_EMBEDDINGS

_NLP_MODEL = None
if spacy is not None:
    model_name = os.getenv(
        "CI_SPACY_MODEL",
        _CFG.get("spacy_model", "en_core_web_sm"),
    )
    try:
        _NLP_MODEL = spacy.load(model_name)  # type: ignore
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("spaCy model load failed: %s", exc)
        try:
            _NLP_MODEL = spacy.blank("en")  # type: ignore
        except Exception:
            _NLP_MODEL = None


@dataclass
class CompetitorUpdate:
    """Simple representation of competitor related news."""

    title: str
    content: str
    source: str
    timestamp: str
    category: str = ""
    sentiment: float = 0.0
    entities: List[str] = field(default_factory=list)
    ai_signals: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligenceDB:
    """SQLite-backed storage for competitor updates."""

    SCHEMA_VERSION = 2

    def __init__(
        self, path: Path | str = Path("intelligence.db"), *, router: DBRouter | None = None
    ) -> None:
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "updates", local_db_path=str(path), shared_db_path=str(path)
        )
        self._init()

    def _get_conn(self) -> sqlite3.Connection:
        return self.router.get_connection("updates")

    def _init(self) -> None:
        conn = self._get_conn()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._create_v1_schema(conn)
            version = 1
            conn.execute("PRAGMA user_version=1")
            conn.commit()
        if version < self.SCHEMA_VERSION:
            self._migrate(conn, version, self.SCHEMA_VERSION)
        elif version > self.SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported DB schema version {version}")

    def _create_v1_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                source TEXT,
                timestamp TEXT,
                sentiment REAL,
                entities TEXT,
                ai_signals INTEGER,
                category TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_updates_ts ON updates(timestamp)"
        )

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Add ``category`` column to ``updates`` table."""
        cols = [r[1] for r in conn.execute("PRAGMA table_info(updates)").fetchall()]
        if "category" not in cols:
            conn.execute(
                "ALTER TABLE updates ADD COLUMN category TEXT DEFAULT ''"
            )
        conn.commit()

    _MIGRATIONS: Dict[int, Any] = {
        0: _create_v1_schema,
        1: _migrate_v1_to_v2,
    }

    def _migrate(
        self, conn: sqlite3.Connection, from_version: int, to_version: int
    ) -> None:
        current = from_version
        while current < to_version:
            step = self._MIGRATIONS.get(current)
            if step is None:
                raise RuntimeError(f"No migration path from {current}")
            step(self, conn)  # type: ignore[misc]
            current += 1
        conn.execute(f"PRAGMA user_version={to_version}")
        conn.commit()

    def add(self, update: CompetitorUpdate) -> int:
        conn = self._get_conn()
        norm_ts = _normalize_timestamp(update.timestamp)
        cur = conn.execute(
            "SELECT id FROM updates WHERE title=? AND content=? AND timestamp=? LIMIT 1",
            (update.title, update.content, norm_ts),
        )
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur = conn.execute(
            """
            INSERT INTO updates
                (title, content, source, timestamp, sentiment, entities, ai_signals, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                update.title,
                update.content,
                update.source,
                norm_ts,
                update.sentiment,
                ",".join(update.entities),
                int(update.ai_signals),
                update.category,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def fetch(self, limit: int = 50) -> List[CompetitorUpdate]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT title, content, source, timestamp, sentiment, entities, ai_signals, category"
            " FROM updates ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results: List[CompetitorUpdate] = []
        for r in rows:
            results.append(
                CompetitorUpdate(
                    title=r[0],
                    content=r[1],
                    source=r[2],
                    timestamp=r[3],
                    sentiment=r[4],
                    entities=r[5].split(",") if r[5] else [],
                    ai_signals=bool(r[6]),
                    category=r[7],
                )
            )
        return results

    def prune(self, older_than_days: int = _RETENTION_DAYS) -> int:
        """Delete records older than ``older_than_days`` days."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM updates WHERE timestamp < ?", (cutoff.isoformat(),)
        )
        conn.commit()
        return int(cur.rowcount)

    def close_all(self) -> None:
        """Close underlying database connections."""
        self.router.close()


def fetch_updates(url: str) -> List[CompetitorUpdate]:
    """Fetch competitor updates from a JSON endpoint."""
    if requests is None:
        logger.error("requests library not available")
        return []

    def _get() -> "requests.Response":
        return requests.get(url, timeout=10)

    try:
        resp = with_retry(_get, attempts=3, delay=1.0, logger=logger)
    except Exception as exc:
        logger.exception("failed fetching %s: %s", url, exc)
        return []

    if resp.status_code != 200:
        logger.error(
            "fetch_updates %s returned status %s", url, resp.status_code
        )
        return []

    try:
        data = resp.json()
    except Exception as exc:
        logger.exception("invalid JSON from %s: %s", url, exc)
        return []
    updates: List[CompetitorUpdate] = []
    for item in data.get("items", []):
        raw_ts = (
            item.get("timestamp") or item.get("time") or item.get("published")
        )
        ts = (
            _normalize_timestamp(raw_ts)
            if raw_ts is not None
            else datetime.utcnow().isoformat()
        )
        updates.append(
            CompetitorUpdate(
                title=str(item.get("title", "")),
                content=str(item.get("content", "")),
                source=url,
                timestamp=ts,
            )
        )
    return updates


def analyse_sentiment(text: str, *, strict_mode: bool | None = None) -> float:
    """Return sentiment score using transformers pipeline when available."""
    if _SENTIMENT_MODEL is not None:
        try:
            sent = _SENTIMENT_MODEL([text])[0]
            label = sent.get("label", "POSITIVE")
            score = float(sent.get("score", 0.5))
            return score if label == "POSITIVE" else -score
        except Exception:
            logger.exception("sentiment pipeline failed")
            if strict_mode if strict_mode is not None else _STRICT_MODE:
                raise
    text_lower = text.lower()
    score = 0.0
    for word, weight in _SENTIMENT_WEIGHTS.items():
        if f"not {word}" in text_lower or f"n't {word}" in text_lower:
            score -= weight
        elif word in text_lower:
            score += weight
    return score


def extract_entities(
    text: str, *, strict_mode: bool | None = None
) -> List[str]:
    """Extract entities using spaCy when available."""
    if _NLP_MODEL is not None:
        try:
            doc = _NLP_MODEL(text)
            ents = [ent.text for ent in getattr(doc, "ents", [])]
            if ents:
                return ents
        except Exception:
            logger.exception("entity extraction failed")
            if strict_mode if strict_mode is not None else _STRICT_MODE:
                raise
    matches = re.findall(r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b", text)
    results = [m for m in matches if m not in _ENTITY_BLACKLIST]
    return results


def detect_ai_signals(
    update: CompetitorUpdate, *, strict_mode: bool | None = None
) -> bool:
    """Heuristic detection of AI-related signals."""
    text = f"{update.title} {update.content}"
    cleaned = redact(text)
    meta = update.metadata.setdefault("scoring", {})
    alerts = find_semantic_risks(cleaned.splitlines())
    if alerts:
        meta["semantic_alerts"] = alerts
    lic = license_detector.detect(cleaned)
    if lic:
        logger.warning("license detected: %s", lic)
        return False
    embedder, ai_embeddings = _ensure_ai_embeddings()
    if embedder and ai_embeddings:
        try:
            vec = governed_embed(cleaned, embedder)
            if vec is not None:
                scores = [_cosine_similarity(vec, kw) for kw in ai_embeddings]
                if _EMBED_STRATEGY == "average":
                    score = sum(scores) / len(scores)
                elif _EMBED_STRATEGY == "topk":
                    k = min(_EMBED_TOPK, len(scores))
                    scores.sort(reverse=True)
                    score = sum(scores[:k]) / k
                else:
                    score = max(scores)
                meta["embedding_score"] = score
                if score > _AI_THRESHOLD:
                    return True
        except Exception:
            logger.exception("AI embedding detection failed")
            if strict_mode if strict_mode is not None else _STRICT_MODE:
                raise
    text_lower = cleaned.lower()
    return sum(k in text_lower for k in _AI_KEYWORDS) > 1


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class CompetitiveIntelligenceBot:
    """Gather and analyse competitor information."""

    def __init__(
        self,
        db: IntelligenceDB | None = None,
        *,
        router: DBRouter | None = None,
        strict_mode: bool | None = None,
    ) -> None:
        self.db = db or IntelligenceDB(router=router)
        self.strict_mode = _STRICT_MODE if strict_mode is None else strict_mode

    def collect(self, urls: Iterable[str]) -> List[CompetitorUpdate]:
        updates: List[CompetitorUpdate] = []
        for url in urls:
            updates.extend(fetch_updates(url))
        return updates

    def analyse(
        self, updates: Iterable[CompetitorUpdate]
    ) -> List[CompetitorUpdate]:
        analysed: List[CompetitorUpdate] = []
        for up in updates:
            up.sentiment = analyse_sentiment(
                up.content, strict_mode=self.strict_mode
            )
            up.entities = extract_entities(
                up.content, strict_mode=self.strict_mode
            )
            up.ai_signals = detect_ai_signals(up, strict_mode=self.strict_mode)
            analysed.append(up)
        return analysed

    def store(self, updates: Iterable[CompetitorUpdate]) -> None:
        for up in updates:
            self.db.add(up)

    def process(self, urls: Iterable[str]) -> List[CompetitorUpdate]:
        updates = self.collect(urls)
        analysed = self.analyse(updates)
        self.store(analysed)
        return analysed


__all__ = [
    "CompetitorUpdate",
    "IntelligenceDB",
    "fetch_updates",
    "analyse_sentiment",
    "extract_entities",
    "detect_ai_signals",
    "CompetitiveIntelligenceBot",
]