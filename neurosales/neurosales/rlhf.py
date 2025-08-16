from __future__ import annotations

import math
import random
from typing import Dict, List, Optional
import os


class RLHFPolicyManager:
    """Simple reinforcement engine using bandit exploration."""

    def __init__(self, *, exploration_rate: float = 0.1) -> None:
        self.exploration_rate = exploration_rate
        self.weights: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.total: int = 0

    # ------------------------------------------------------------------
    def _reward(self, ctr: float, sentiment: float, session: float) -> float:
        """Synthetic reward from metrics."""
        return 0.5 * ctr + 0.3 * sentiment + 0.2 * session

    def record_result(self, response: str, *, ctr: float, sentiment: float, session: float) -> None:
        """Update policy weight from a single feedback signal."""
        r = self._reward(ctr, sentiment, session)
        w = self.weights.get(response, 0.0)
        c = self.counts.get(response, 0)
        self.counts[response] = c + 1
        self.total += 1
        self.weights[response] = w + (r - w) / (c + 1)
        self._prune()

    # ------------------------------------------------------------------
    def _ucb_score(self, response: str) -> float:
        w = self.weights.get(response, 0.0)
        c = self.counts.get(response, 0)
        if c == 0:
            return float("inf")  # force exploration
        return w + math.sqrt(2.0 * math.log(max(1, self.total)) / c)

    def _prune(self, threshold: float = -0.05) -> None:
        for resp in list(self.weights.keys()):
            if self.weights[resp] < threshold:
                self.weights.pop(resp)
                self.counts.pop(resp, None)

    # ------------------------------------------------------------------
    def best_response(self, candidates: List[str]) -> str:
        """Select a response using UCB with occasional exploration."""
        if not candidates:
            return ""
        if random.random() < self.exploration_rate:
            return random.choice(candidates)

        best = candidates[0]
        best_score = -float("inf")
        for resp in candidates:
            score = self._ucb_score(resp)
            if score > best_score:
                best = resp
                best_score = score
        return best


class DatabaseRLHFPolicyManager(RLHFPolicyManager):
    """Persistent RLHF policy manager backed by SQL."""

    def __init__(
        self,
        *,
        exploration_rate: float = 0.1,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        super().__init__(exploration_rate=exploration_rate)
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

        from .sql_db import RLPolicyWeight

        Session = self.session_factory
        with Session() as s:
            rows = s.query(RLPolicyWeight).all()
        for r in rows:
            self.weights[r.response] = r.weight
            self.counts[r.response] = r.count
            self.total += r.count

    def record_result(self, response: str, *, ctr: float, sentiment: float, session: float) -> None:  # type: ignore[override]
        super().record_result(response, ctr=ctr, sentiment=sentiment, session=session)
        from .sql_db import RLPolicyWeight

        Session = self.session_factory
        with Session() as s:
            rec = s.get(RLPolicyWeight, response)
            if rec is None:
                rec = RLPolicyWeight(response=response, weight=self.weights[response], count=self.counts[response])
                s.add(rec)
            else:
                rec.weight = self.weights[response]
                rec.count = self.counts[response]
            s.commit()
