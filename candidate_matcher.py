"""Compare a new NicheCandidate against stored models."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import log, sqrt

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime
import logging

from .task_handoff_bot import WorkflowDB, WorkflowRecord
from .unified_event_bus import UnifiedEventBus
from .chatgpt_enhancement_bot import EnhancementDB, Enhancement
from dynamic_path_router import resolve_path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

from .normalize_scraped_data import NicheCandidate
from .database_manager import DB_PATH, get_connection, init_db


logger = logging.getLogger(__name__)

# Cached corpus of tokenized documents for fallback TF-IDF computation
# When ``sklearn`` is unavailable this growing list provides an approximate
# corpus for inverse document frequency calculations across calls.
_TFIDF_CORPUS: List[List[str]] = []

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ForkedModel:
    """Representation of a forked model and its workflow."""

    workflow: WorkflowRecord
    parent_model_id: str
    model_name: str
    tags: List[str]
    model_id: str = field(default_factory=lambda: uuid4().hex)
    source: str = "Competitor Parasite System"
    date_discovered: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )
    initial_roi_prediction: float = 0.0


_def_cols = ["id", "name", "niche", "tags", "price_point"]


def _simple_tfidf_similarity(a: str, b: str) -> float:
    """Return cosine similarity between two pieces of text.

    When ``sklearn`` is installed a small ``TfidfVectorizer`` is used with
    bigram support.  Otherwise a very small TF‑IDF implementation acts as a
    fallback.  In either mode common stop words are removed and token length is
    taken into account when weighting terms.
    """

    if TfidfVectorizer and cosine_similarity:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        ).fit([a, b])
        tf = vec.transform([a, b])
        return float(cosine_similarity(tf[0], tf[1])[0][0])

    stop = {
        "the",
        "is",
        "in",
        "and",
        "to",
        "of",
        "a",
        "an",
        "on",
        "for",
        "with",
        "that",
        "this",
        "it",
        "as",
        "at",
        "by",
        "be",
        "from",
        "are",
        "was",
        "were",
        "or",
        "but",
        "not",
        "no",
        "can",
        "could",
        "should",
        "would",
        "will",
        "their",
        "there",
        "if",
        "then",
        "than",
        "so",
        "such",
        "very",
        "may",
        "might",
        "also",
        "into",
        "about",
        "over",
        "after",
        "before",
        "between",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "done",
    }

    def tok(s: str) -> List[str]:
        s = _TOKEN_RE.sub(" ", s.lower())
        words = [t for t in s.split() if t and t not in stop]
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)] if words else []
        return words + bigrams

    t1 = tok(a)
    t2 = tok(b)

    docs = list(_TFIDF_CORPUS)
    full_vocab = {tok for doc in docs for tok in doc} | set(t1) | set(t2)
    vocab = sorted(full_vocab)
    if not vocab:
        return 1.0

    df = {t: sum(t in d for d in docs) for t in vocab}
    idf = {t: log((len(docs) + 1) / (df[t] + 1)) + 1 for t in vocab}

    def vec(doc: List[str]):
        vec_vals = []
        for t in vocab:
            count = doc.count(t)
            if count:
                tf_val = (1.0 + log(count)) * (1.0 + len(t) / 10)
                vec_vals.append(tf_val * idf[t])
            else:
                vec_vals.append(0.0)
        return vec_vals

    v1 = vec(t1)
    v2 = vec(t2)
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = sqrt(sum(x * x for x in v1))
    norm2 = sqrt(sum(x * x for x in v2))
    sim = dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    _TFIDF_CORPUS.append(t1)
    _TFIDF_CORPUS.append(t2)

    return sim


def _text_similarity(a: str, b: str) -> float:
    """Return similarity between ``a`` and ``b``.

    This delegates to :func:`_simple_tfidf_similarity` which will try to use
    ``sklearn`` when available and otherwise fall back to a minimal
    implementation.  If all approaches fail ``SequenceMatcher`` is used as a
    last resort.
    """

    a = a or ""
    b = b or ""
    if a == "" and b == "":
        return 1.0
    try:
        return _simple_tfidf_similarity(a, b)
    except Exception as exc:
        logger.warning(
            "simple TF-IDF similarity failed for %r and %r: %s", a, b, exc
        )
        return SequenceMatcher(None, a, b).ratio()


def _jaccard(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    intersect = set_a & set_b
    union = set_a | set_b
    return len(intersect) / len(union)


def find_matching_models(
    candidate: NicheCandidate, *, threshold: float = 0.8, db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        init_db(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
        sel = [c for c in _def_cols if c in cols]
        query = f"SELECT {', '.join(sel)} FROM models"
        rows = [dict(zip(sel, row)) for row in conn.execute(query)]

    matches: List[Dict[str, Any]] = []
    for row in rows:
        niche_other = row.get("niche") or row.get("name", "")
        niche_score = _text_similarity(candidate.niche or "", str(niche_other))
        tags_other = str(row.get("tags", "")).split()
        tag_score = _jaccard(candidate.tags, tags_other)
        price_score = 0.0
        if candidate.price_point is not None and row.get("price_point") is not None:
            try:
                other = float(row["price_point"])
                diff = abs(candidate.price_point - other)
                max_price = max(candidate.price_point, other)
                if max_price:
                    price_score = max(0.0, 1.0 - diff / max_price)
            except Exception:
                price_score = 0.0
        score = 0.5 * niche_score + 0.3 * tag_score + 0.2 * price_score
        if score >= threshold:
            row["similarity"] = round(score * 100, 2)
            matches.append(row)
    return matches


def generate_uuid() -> str:
    return uuid4().hex


def fork_model_from_candidate(
    candidate: NicheCandidate,
    *,
    threshold: float = 0.8,
    models_db: Path = DB_PATH,
    workflows_db: Path = resolve_path("workflows.db"),
    event_bus: UnifiedEventBus | None = None,
) -> Optional[str]:
    matches = find_matching_models(candidate, threshold=threshold, db_path=models_db)
    if not matches:
        return None
    match = matches[0]

    workflows_db = Path(workflows_db).resolve()
    wf_db = WorkflowDB(workflows_db, event_bus=event_bus)
    base = None
    for rec in wf_db.fetch():
        if (
            rec.wid == match.get("id")
            or rec.title.lower() == str(match.get("name", "")).lower()
        ):
            base = rec
            break
    if not base:
        return None

    cloned = WorkflowRecord(
        workflow=base.workflow,
        assigned_bots=base.assigned_bots,
        enhancements=base.enhancements,
        title=candidate.product_name,
        description=base.description,
        task_sequence=base.task_sequence,
        tags=candidate.tags,
        category=base.category,
        type_=base.type_,
        status=base.status,
        rejection_reason=base.rejection_reason,
        workflow_duration=base.workflow_duration,
        performance_data=base.performance_data,
    )
    forked_workflow_id = wf_db.add(cloned)
    if forked_workflow_id is None:
        logger.warning(
            "duplicate workflow ignored for %s", candidate.product_name
        )

    model_uuid = generate_uuid()
    with get_connection(models_db) as conn:
        init_db(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
        if "model_id" not in cols:
            conn.execute("ALTER TABLE models ADD COLUMN model_id TEXT")
        if "initial_roi_prediction" not in cols:
            conn.execute(
                "ALTER TABLE models ADD COLUMN initial_roi_prediction REAL DEFAULT 0"
            )
        if "workflow_id" not in cols:
            conn.execute("ALTER TABLE models ADD COLUMN workflow_id INTEGER")
        if "current_roi" not in cols:
            conn.execute("ALTER TABLE models ADD COLUMN current_roi REAL DEFAULT 0")
        conn.execute(
            """
            INSERT INTO models (
                name,
                source,
                date_discovered,
                tags,
                roi_metadata,
                exploration_status,
                model_id,
                initial_roi_prediction,
                workflow_id,
                current_roi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"Forked: {candidate.product_name}",
                "Competitor Parasite System",
                datetime.utcnow().isoformat(timespec="seconds"),
                " ".join(candidate.tags),
                "",
                "new",
                model_uuid,
                0.0,
                forked_workflow_id,
                0.0,
            ),
        )

    return model_uuid


def insert_forked_entry(
    candidate: NicheCandidate,
    fork: ForkedModel,
    *,
    models_db: Path = DB_PATH,
    workflows_db: Path = resolve_path("workflows.db"),
    enhancements_db: Path = resolve_path("enhancements.db"),
    event_bus: UnifiedEventBus | None = None,
) -> bool:
    try:
        workflows_db = Path(workflows_db).resolve()
        wf_db = WorkflowDB(workflows_db, event_bus=event_bus)
        workflow_id = wf_db.add(fork.workflow)
    except Exception as exc:
        logger.exception("Failed to insert workflow: %s", exc)
        return False

    try:
        with get_connection(models_db) as conn:
            init_db(conn)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
            if "model_id" not in cols:
                conn.execute("ALTER TABLE models ADD COLUMN model_id TEXT")
            if "initial_roi_prediction" not in cols:
                conn.execute(
                    "ALTER TABLE models ADD COLUMN initial_roi_prediction REAL DEFAULT 0"
                )
            if "workflow_id" not in cols:
                conn.execute("ALTER TABLE models ADD COLUMN workflow_id INTEGER")
            if "current_roi" not in cols:
                conn.execute("ALTER TABLE models ADD COLUMN current_roi REAL DEFAULT 0")
            conn.execute(
                """
                INSERT INTO models (
                    name,
                    source,
                    date_discovered,
                    tags,
                    roi_metadata,
                    exploration_status,
                    model_id,
                    initial_roi_prediction,
                    workflow_id,
                    current_roi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fork.model_name or f"Forked: {candidate.product_name}",
                    fork.source,
                    fork.date_discovered,
                    " ".join(fork.tags or candidate.tags),
                    "",
                    "new",
                    fork.model_id,
                    fork.initial_roi_prediction,
                    workflow_id,
                    0.0,
                ),
            )
    except Exception as exc:
        logger.exception("Failed to insert model: %s", exc)
        return False

    try:
        enhancements_db = Path(enhancements_db).resolve()
        enh_db = EnhancementDB(enhancements_db)
        enh_db.add(
            Enhancement(
                idea="fork_lineage",
                rationale=f"{fork.parent_model_id}->{fork.model_id}",
                context=candidate.product_name,
                tags=["fork", "lineage"],
            )
        )
    except Exception as exc:
        logger.exception("Failed to log fork lineage: %s", exc)
        return False

    logger.info(
        "Inserted forked model %s derived from %s", fork.model_id, fork.parent_model_id
    )
    return True


__all__ = [
    "find_matching_models",
    "generate_uuid",
    "fork_model_from_candidate",
    "ForkedModel",
    "insert_forked_entry",
]
