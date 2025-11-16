from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import os

try:
    from sklearn.cluster import KMeans
    import numpy as np
except Exception:  # pragma: no cover - optional heavy dep
    KMeans = None
    np = None

import logging
from .embedding import embed_text

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    timestamp: float
    text: str
    issue: str
    correction: Optional[str] = None
    engagement: float = 1.0


class SelfLearningRepo:
    """Simple DB helper for persisting learning events and state."""

    def __init__(
        self,
        session_factory: Optional[Callable] = None,
        *,
        db_url: Optional[str] = None,
    ) -> None:
        if session_factory is None:
            from .sql_db import create_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_session(db_url)
        self.session_factory = session_factory

    # ------------------------------------------------------------------

    def add_event(
        self,
        text: str,
        issue: str,
        *,
        correction: Optional[str] = None,
        engagement: float = 1.0,
    ) -> None:
        from .sql_db import SelfLearningEvent

        Session = self.session_factory
        with Session() as s:
            rec = SelfLearningEvent(
                text=text,
                issue=issue,
                correction=correction,
                engagement=engagement,
            )
            s.add(rec)
            s.commit()

    def fetch_events(self, since: Optional[float] = None) -> List[LearningEvent]:
        from .sql_db import SelfLearningEvent

        Session = self.session_factory
        with Session() as s:
            q = s.query(SelfLearningEvent)
            if since is not None:
                q = q.filter(SelfLearningEvent.timestamp >= since)
            rows = q.order_by(SelfLearningEvent.timestamp.asc()).all()
        return [
            LearningEvent(r.timestamp, r.text, r.issue, r.correction, r.engagement)
            for r in rows
        ]

    # ------------------------------------------------------------------
    def load_state(self) -> Dict[str, any]:
        """Return persisted engine state if available."""
        from .sql_db import SelfLearningState

        Session = self.session_factory
        with Session() as s:
            row = (
                s.query(SelfLearningState).order_by(SelfLearningState.id.desc()).first()
            )
            return row.data if row is not None and row.data is not None else {}

    # ------------------------------------------------------------------
    def save_state(self, data: Dict[str, any]) -> None:
        from .sql_db import SelfLearningState

        Session = self.session_factory
        with Session() as s:
            row = (
                s.query(SelfLearningState).order_by(SelfLearningState.id.desc()).first()
            )
            if row is None:
                row = SelfLearningState(data=data)
                s.add(row)
            else:
                row.data = data
            s.commit()


class SelfLearningBuffer:
    """Store interactions that require additional learning."""

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        session_factory: Optional[Callable] = None,
        *,
        db_url: Optional[str] = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.repo = SelfLearningRepo(session_factory=session_factory, db_url=db_url)

    def add_event(
        self,
        text: str,
        issue: str,
        *,
        correction: Optional[str] = None,
        engagement: float = 1.0,
    ) -> None:
        self.repo.add_event(text, issue, correction=correction, engagement=engagement)

    def prune(self) -> None:
        """No-op for compatibility."""
        # pruning handled by database query in get_events
        return

    def get_events(self) -> List[LearningEvent]:
        since = None
        if self.ttl_seconds is not None:
            since = time.time() - self.ttl_seconds
        return self.repo.fetch_events(since=since)


class SelfLearningEngine:
    """Continuous improvement engine using buffered feedback."""

    def __init__(
        self,
        decay_rate: float = 0.9,
        session_factory: Optional[Callable] = None,
        *,
        db_url: Optional[str] = None,
    ) -> None:
        self.decay_rate = decay_rate
        self.repo = SelfLearningRepo(session_factory=session_factory, db_url=db_url)
        self.buffer = SelfLearningBuffer(session_factory=session_factory, db_url=db_url)
        self.session_factory = self.repo.session_factory
        self.fact_weights: Dict[str, float] = {}
        self.main_weights: Dict[str, float] = {}
        self.micro_jobs: List[str] = []
        self.corrections_history: List[Tuple[str, float]] = []
        self.embedding_cache: Dict[str, List[float]] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        data = self.repo.load_state()
        self.fact_weights = data.get("fact_weights", {})
        raw_hist = data.get("corrections_history", [])
        self.corrections_history = [tuple(item) for item in raw_hist]
        self.embedding_cache = data.get("embedding_cache", {})

    def _save_state(self) -> None:
        data = {
            "fact_weights": self.fact_weights,
            "corrections_history": self.corrections_history,
            "embedding_cache": self.embedding_cache,
        }
        self.repo.save_state(data)

    # -------------------------- Buffer management --------------------------
    def log_interaction(
        self,
        text: str,
        issue: str,
        *,
        correction: Optional[str] = None,
        engagement: float = 1.0,
    ) -> None:
        self.buffer.add_event(text, issue, correction=correction, engagement=engagement)
        if correction:
            self.record_correction(correction, impact=1.0)
        from .sql_db import log_rl_feedback

        fb = correction if correction is not None else issue
        log_rl_feedback(
            text,
            fb,
            engagement,
            session_factory=self.session_factory,
        )

    # -------------------------- RL trainer integration --------------------------
    def export_corrections(
        self,
        trainer: "DatabaseRLResponseRanker",
        batch_size: int = 10,
    ) -> int:
        """Send recent feedback to the given trainer and mark them processed."""

        from .sql_db import RLFeedback

        Session = self.session_factory
        with Session() as s:
            rows = (
                s.query(RLFeedback)
                .filter(RLFeedback.processed.is_(False))
                .order_by(RLFeedback.id)
                .limit(batch_size)
                .all()
            )
            for r in rows:
                state = (len(r.text),)
                trainer.log_outcome(
                    "trainer",
                    state,
                    r.feedback,
                    float(r.score),
                    state,
                    [r.feedback],
                )
                r.processed = True
            s.commit()
        return len(rows)

    # -------------------------- Watchdog and decay --------------------------
    def run_watchdog(self, threshold: int = 2) -> None:
        counts: Dict[str, int] = {}
        for ev in self.buffer.get_events():
            counts[ev.issue] = counts.get(ev.issue, 0) + 1
        for issue, cnt in counts.items():
            if cnt >= threshold and issue not in self.micro_jobs:
                self.micro_jobs.append(issue)

    def record_correction(self, fact: str, impact: float = 1.0) -> None:
        self.fact_weights[fact] = self.fact_weights.get(fact, 1.0) * self.decay_rate
        self.corrections_history.append((fact, impact))
        self._save_state()

    # -------------------------- Similarity linking --------------------------
    def link_concepts(self) -> None:
        if KMeans is None or np is None or not self.corrections_history:
            return

        names = [fact for fact, _ in self.corrections_history]
        vectors: List[List[float]] = []
        for name in names:
            if name not in self.embedding_cache:
                try:
                    self.embedding_cache[name] = embed_text(name)
                except Exception as e:
                    logger.exception("Failed to embed '%s'", name)
                    raise RuntimeError(f"Failed to embed '{name}'") from e
            vectors.append(self.embedding_cache[name])

        X = np.array(vectors, dtype=float)
        labels = KMeans(n_clusters=min(2, len(names)), n_init="auto").fit_predict(X)
        for name, label in zip(names, labels):
            cat = f"cat_{label}"
            self.knowledge_graph.setdefault(cat, []).append(name)
        self._save_state()

    # -------------------------- Weekly audit --------------------------
    def weekly_audit(self) -> None:
        ranked = sorted(self.corrections_history, key=lambda x: x[1], reverse=True)
        for fact, _ in ranked[:3]:
            self.main_weights[fact] = self.fact_weights.get(fact, 1.0)
