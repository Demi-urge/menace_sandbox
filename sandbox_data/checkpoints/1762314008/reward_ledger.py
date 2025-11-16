from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

from .sentiment import SentimentAnalyzer


@dataclass
class CoinBalance:
    """Balance for each reward currency."""

    green: float = 0.0
    violet: float = 0.0
    gold: float = 0.0
    iron: float = 0.0


@dataclass
class StampedLine:
    """Record of provisional coinage for an outgoing line."""

    timestamp: float
    confidence: float
    coins: CoinBalance
    sentiment_before: float
    sentiment_after: float
    followups: int
    session_delta: float
    fact_error: bool
    lost_user: bool


class RewardLedger:
    """Tiered reward system tracking multiple ledgers."""

    def __init__(self) -> None:
        self.sentiment = SentimentAnalyzer()
        self.ledgers: Dict[str, CoinBalance] = {}
        self.lines: Dict[int, StampedLine] = {}
        self.counters: Dict[str, int] = {}
        self._line_id = 0

    # ------------------------------------------------------------------
    def _multiplier(self, user_id: str) -> float:
        count = self.counters.get(user_id, 0) + 1
        self.counters[user_id] = count
        if count <= 3:
            return 0.8  # early conversation tax
        if count >= 10:
            return 1.2  # closing leverage
        return 1.0

    def _ensure(self, user_id: str) -> CoinBalance:
        bal = self.ledgers.get(user_id)
        if bal is None:
            bal = CoinBalance()
            self.ledgers[user_id] = bal
        return bal

    # ------------------------------------------------------------------
    def stamp_line(
        self,
        user_id: str,
        text: str,
        *,
        sentiment_before: Optional[float] = None,
        followups: int = 0,
        session_delta: float = 0.0,
        pref_match: float = 0.0,
        fact_error: bool = False,
        lost_user: bool = False,
        confidence: float = 1.0,
    ) -> int:
        """Stamp a line with provisional rewards and update ledgers."""
        if sentiment_before is None:
            sentiment_before, _ = self.sentiment.analyse(text)
        sentiment_after, _ = self.sentiment.analyse(text)
        mood_alignment = 1.0 - abs(sentiment_after - sentiment_before)
        mult = self._multiplier(user_id) * max(0.0, min(1.0, confidence))
        coins = CoinBalance(
            green=(session_delta + followups) * mult,
            violet=mood_alignment * mult,
            gold=pref_match * mult,
            iron=-(1.0 if fact_error or lost_user else 0.0) * mult,
        )
        bal = self._ensure(user_id)
        bal.green += coins.green
        bal.violet += coins.violet
        bal.gold += coins.gold
        bal.iron += coins.iron
        self._line_id += 1
        self.lines[self._line_id] = StampedLine(
            timestamp=time.time(),
            confidence=confidence,
            coins=coins,
            sentiment_before=sentiment_before,
            sentiment_after=sentiment_after,
            followups=followups,
            session_delta=session_delta,
            fact_error=fact_error,
            lost_user=lost_user,
        )
        return self._line_id

    # ------------------------------------------------------------------
    def audit_line(
        self,
        line_id: int,
        *,
        session_delta: Optional[float] = None,
        sentiment_after: Optional[float] = None,
        lost_user: Optional[bool] = None,
    ) -> None:
        """Retroactively adjust rewards for a line."""
        line = self.lines.get(line_id)
        if line is None:
            return
        bal = self._ensure("audit")  # store global audit adjustments
        def adjust(attr: str, new: float) -> None:
            old = getattr(line.coins, attr)
            diff = new - old
            if diff:
                setattr(line.coins, attr, new)
                bal_attr = getattr(bal, attr)
                setattr(bal, attr, bal_attr + diff)

        if session_delta is not None:
            mult = line.confidence
            new_green = (session_delta + line.followups) * mult
            adjust("green", new_green)
        if sentiment_after is not None:
            mood_alignment = 1.0 - abs(sentiment_after - line.sentiment_before)
            adjust("violet", mood_alignment * line.confidence)
        if lost_user is not None:
            adjust("iron", -(1.0 if lost_user else 0.0) * line.confidence)


class DatabaseRewardLedger(RewardLedger):
    """Reward ledger backed by a SQL database."""

    def __init__(self, *, session_factory: Optional[callable] = None, db_url: Optional[str] = None) -> None:
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory
        super().__init__()

        # load existing entries
        from .sql_db import RewardEntry

        Session = self.session_factory
        with Session() as s:
            rows = s.query(RewardEntry).order_by(RewardEntry.id).all()
        for r in rows:
            coins = CoinBalance(r.green, r.violet, r.gold, r.iron)
            self.lines[r.id] = StampedLine(
                timestamp=r.timestamp,
                confidence=r.confidence,
                coins=coins,
                sentiment_before=r.sentiment_before,
                sentiment_after=r.sentiment_after,
                followups=r.followups,
                session_delta=r.session_delta,
                fact_error=bool(r.fact_error),
                lost_user=bool(r.lost_user),
            )
            bal = self._ensure(r.user_id)
            bal.green += coins.green
            bal.violet += coins.violet
            bal.gold += coins.gold
            bal.iron += coins.iron
            self.counters[r.user_id] = self.counters.get(r.user_id, 0) + 1
            if r.id > self._line_id:
                self._line_id = r.id

    # ------------------------------------------------------------------
    def stamp_line(
        self,
        user_id: str,
        text: str,
        *,
        sentiment_before: Optional[float] = None,
        followups: int = 0,
        session_delta: float = 0.0,
        pref_match: float = 0.0,
        fact_error: bool = False,
        lost_user: bool = False,
        confidence: float = 1.0,
    ) -> int:
        line_id = super().stamp_line(
            user_id,
            text,
            sentiment_before=sentiment_before,
            followups=followups,
            session_delta=session_delta,
            pref_match=pref_match,
            fact_error=fact_error,
            lost_user=lost_user,
            confidence=confidence,
        )
        line = self.lines[line_id]
        from .sql_db import RewardEntry

        Session = self.session_factory
        with Session() as s:
            s.add(
                RewardEntry(
                    id=line_id,
                    user_id=user_id,
                    timestamp=line.timestamp,
                    green=line.coins.green,
                    violet=line.coins.violet,
                    gold=line.coins.gold,
                    iron=line.coins.iron,
                    sentiment_before=line.sentiment_before,
                    sentiment_after=line.sentiment_after,
                    followups=line.followups,
                    session_delta=line.session_delta,
                    fact_error=line.fact_error,
                    lost_user=line.lost_user,
                    confidence=line.confidence,
                )
            )
            s.commit()
        return line_id

    # ------------------------------------------------------------------
    def audit_line(
        self,
        line_id: int,
        *,
        session_delta: Optional[float] = None,
        sentiment_after: Optional[float] = None,
        lost_user: Optional[bool] = None,
    ) -> None:
        super().audit_line(
            line_id,
            session_delta=session_delta,
            sentiment_after=sentiment_after,
            lost_user=lost_user,
        )
        line = self.lines.get(line_id)
        if line is None:
            return
        from .sql_db import RewardEntry

        Session = self.session_factory
        with Session() as s:
            entry = s.get(RewardEntry, line_id)
            if entry is None:
                return
            if session_delta is not None:
                entry.session_delta = session_delta
                entry.green = line.coins.green
            if sentiment_after is not None:
                entry.sentiment_after = sentiment_after
                entry.violet = line.coins.violet
            if lost_user is not None:
                entry.lost_user = lost_user
                entry.iron = line.coins.iron
            s.commit()


