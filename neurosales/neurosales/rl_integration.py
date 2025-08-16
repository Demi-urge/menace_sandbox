from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional
import sqlalchemy as sa
import os


@dataclass
class Experience:
    state: Tuple[int, ...]
    action: str
    reward: float
    next_state: Tuple[int, ...]
    done: bool = False


class ReplayBuffer:
    """Simple FIFO buffer for storing experiences."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.buffer: Deque[Experience] = deque()

    def add(self, exp: Experience) -> None:
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    def sample_feedback(self, batch_size: int) -> List[Experience]:
        """Hook for buffers that fetch historical feedback."""
        return []


class DatabaseReplayBuffer(ReplayBuffer):
    """Replay buffer persisted to a SQL database."""

    def __init__(
        self,
        user_id: str,
        max_size: int = 1000,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        super().__init__(max_size=max_size)
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory
        self.user_id = user_id

        from .sql_db import ReplayExperience, RLFeedback

        Session = self.session_factory
        with Session() as s:
            rows = (
                s.query(ReplayExperience)
                .filter_by(user_id=user_id)
                .order_by(ReplayExperience.id)
                .all()
            )
            fb_rows = s.query(RLFeedback).order_by(RLFeedback.id).all()
        for r in rows[-max_size:]:
            self.buffer.append(
                Experience(
                    tuple(r.state or []),
                    r.action,
                    r.reward,
                    tuple(r.next_state or []),
                )
            )
        for r in fb_rows[-max_size:]:
            self.buffer.append(
                Experience((len(r.text),), r.feedback, r.score, (len(r.text),))
            )

    def add(self, exp: Experience) -> None:  # type: ignore[override]
        super().add(exp)
        from .sql_db import ReplayExperience

        Session = self.session_factory
        with Session() as s:
            s.add(
                ReplayExperience(
                    user_id=self.user_id,
                    state=list(exp.state),
                    action=exp.action,
                    reward=exp.reward,
                    next_state=list(exp.next_state),
                )
            )
            s.commit()

    def sample_feedback(self, batch_size: int) -> List[Experience]:
        from .sql_db import RLFeedback

        Session = self.session_factory
        with Session() as s:
            rows = (
                s.query(RLFeedback)
                .order_by(sa.func.random())
                .limit(batch_size)
                .all()
            )
        return [
            Experience((len(r.text),), r.feedback, r.score, (len(r.text),))
            for r in rows
        ]


class QLearningModule:
    """Tabular Q-learning for discrete actions."""

    def __init__(self, alpha: float = 0.5, gamma: float = 0.9, epsilon: float = 0.1) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values: Dict[Tuple[Tuple[int, ...], str], float] = {}

    # ------------------------------------------------------------------
    def predict(self, state: Tuple[int, ...], action: str) -> float:
        return self.q_values.get((state, action), 0.0)

    def best_action(self, state: Tuple[int, ...], actions: List[str]) -> str:
        if not actions:
            return ""
        if random.random() < self.epsilon:
            return random.choice(actions)
        best = actions[0]
        best_q = self.predict(state, best)
        for a in actions[1:]:
            q = self.predict(state, a)
            if q > best_q:
                best = a
                best_q = q
        return best

    def update(
        self,
        state: Tuple[int, ...],
        action: str,
        reward: float,
        next_state: Tuple[int, ...],
        next_actions: List[str],
    ) -> None:
        current_q = self.predict(state, action)
        next_q = 0.0
        if next_actions:
            next_q = max(self.predict(next_state, a) for a in next_actions)
        td_target = reward + self.gamma * next_q
        self.q_values[(state, action)] = current_q + self.alpha * (td_target - current_q)


class MetaLearner:
    """Share Q-values across similar user models."""

    def __init__(self) -> None:
        self.groups: Dict[str, List[QLearningModule]] = {}

    def register(self, group: str, module: QLearningModule) -> None:
        self.groups.setdefault(group, []).append(module)

    def aggregate(self, group: str) -> None:
        modules = self.groups.get(group)
        if not modules:
            return
        all_keys = {key for m in modules for key in m.q_values}
        for key in all_keys:
            avg = sum(m.q_values.get(key, 0.0) for m in modules) / len(modules)
            for m in modules:
                m.q_values[key] = avg


class RLResponseRanker:
    """Rank responses using Q-learning predictions and replay buffers."""

    def __init__(self, meta: MetaLearner | None = None, buffer_size: int = 1000) -> None:
        self.meta = meta or MetaLearner()
        self.modules: Dict[str, QLearningModule] = {}
        self.buffers: Dict[str, ReplayBuffer] = {}
        self.buffer_size = buffer_size

    # ------------------------------------------------------------------
    def _group(self, user_id: str) -> str:
        return user_id.split("-")[0] if "-" in user_id else user_id

    def _module(self, user_id: str) -> QLearningModule:
        mod = self.modules.get(user_id)
        if mod is None:
            mod = QLearningModule()
            self.modules[user_id] = mod
            self.meta.register(self._group(user_id), mod)
        return mod

    def _buffer(self, user_id: str) -> ReplayBuffer:
        buf = self.buffers.get(user_id)
        if buf is None:
            buf = ReplayBuffer(max_size=self.buffer_size)
            self.buffers[user_id] = buf
        return buf

    # ------------------------------------------------------------------
    def rank(self, user_id: str, scores: Dict[str, float], history: List[str] | None = None) -> List[str]:
        history = history or []
        state = (len(history),)
        module = self._module(user_id)
        adjusted: Dict[str, float] = {}
        for resp, sc in scores.items():
            q = module.predict(state, resp)
            adjusted[resp] = sc + q
        ranked = sorted(adjusted, key=lambda r: adjusted[r], reverse=True)
        return ranked

    # ------------------------------------------------------------------
    def log_outcome(
        self,
        user_id: str,
        state: Tuple[int, ...],
        action: str,
        reward: float,
        next_state: Tuple[int, ...],
        next_actions: List[str],
    ) -> None:
        module = self._module(user_id)
        buffer = self._buffer(user_id)
        module.update(state, action, reward, next_state, next_actions)
        buffer.add(Experience(state, action, reward, next_state))
        for fb in buffer.sample_feedback(1):
            module.update(fb.state, fb.action, fb.reward, fb.next_state, next_actions)
        self.meta.aggregate(self._group(user_id))


class DatabaseRLResponseRanker(RLResponseRanker):
    """RL response ranker that persists replay buffers."""

    def __init__(
        self,
        meta: MetaLearner | None = None,
        buffer_size: int = 1000,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        super().__init__(meta=meta, buffer_size=buffer_size)
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

    def _buffer(self, user_id: str) -> DatabaseReplayBuffer:  # type: ignore[override]
        buf = self.buffers.get(user_id)
        if buf is None:
            buf = DatabaseReplayBuffer(
                user_id,
                max_size=self.buffer_size,
                session_factory=self.session_factory,
            )
            self.buffers[user_id] = buf
        return buf

