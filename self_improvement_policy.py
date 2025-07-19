from __future__ import annotations

"""Q-learning policy predicting ROI improvement from self-improvement cycles."""

from typing import Dict, Tuple, Optional, Callable
import pickle
import os
import random
import math
import statistics


class RLStrategy:
    """Base class for reinforcement learning update strategies."""

    def update(
        self,
        table: Dict[Tuple[int, ...], Dict[int, float]],
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def value(table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]) -> float:
        values = table.get(state, {})
        return max(values.values()) if values else 0.0


class QLearningStrategy(RLStrategy):
    """Standard Q-learning update."""

    def update(
        self,
        table: Dict[Tuple[int, ...], Dict[int, float]],
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        next_best = 0.0
        if next_state is not None:
            next_best = max(table.get(next_state, {}).values(), default=0.0)
        q += alpha * (reward + gamma * next_best - q)
        state_table[action] = q
        return q


class QLambdaStrategy(RLStrategy):
    """Q(lambda) update using eligibility traces."""

    def __init__(self, lam: float = 0.9) -> None:
        self.lam = lam
        self.eligibility: Dict[Tuple[int, ...], Dict[int, float]] = {}

    def update(
        self,
        table: Dict[Tuple[int, ...], Dict[int, float]],
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        next_best = 0.0
        if next_state is not None:
            next_best = max(table.get(next_state, {}).values(), default=0.0)
        delta = reward + gamma * next_best - q
        self.eligibility.setdefault(state, {})
        self.eligibility[state][action] = self.eligibility[state].get(action, 0.0) + 1.0
        for s, acts in list(self.eligibility.items()):
            for a, e in list(acts.items()):
                table.setdefault(s, {})
                table[s][a] = table[s].get(a, 0.0) + alpha * delta * e
                acts[a] = gamma * self.lam * e
                if acts[a] < 1e-6:
                    del acts[a]
            if not acts:
                del self.eligibility[s]
        return table[state][action]


class SarsaStrategy(RLStrategy):
    """SARSA update."""

    def update(
        self,
        table: Dict[Tuple[int, ...], Dict[int, float]],
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        next_val = 0.0
        if next_state is not None:
            next_val = table.get(next_state, {}).get(action, 0.0)
        q += alpha * (reward + gamma * next_val - q)
        state_table[action] = q
        return q


class ActorCriticStrategy(RLStrategy):
    """Simplified actor-critic update."""

    def __init__(self) -> None:
        self.state_values: Dict[Tuple[int, ...], float] = {}

    def update(
        self,
        table: Dict[Tuple[int, ...], Dict[int, float]],
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        v = self.state_values.get(state, 0.0)
        next_v = self.state_values.get(next_state, 0.0) if next_state is not None else 0.0
        td_error = reward + gamma * next_v - v
        self.state_values[state] = v + alpha * td_error
        state_table[action] = q + alpha * td_error
        return state_table[action]


class SelfImprovementPolicy:
    """Tiny reinforcement learning helper for the self-improvement engine."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.9,
        path: Optional[str] = None,
        *,
        strategy: RLStrategy | str = "q_learning",
        exploration: str = "epsilon_greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        adaptive: bool = False,
        epsilon_schedule: Optional[Callable[[int, float], float]] = None,
        temperature_schedule: Optional[Callable[[int, float], float]] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.epsilon = epsilon
        self.temperature = temperature
        self.exploration = exploration
        self.epsilon_schedule = epsilon_schedule
        self.temperature_schedule = temperature_schedule
        if isinstance(strategy, str):
            if strategy.lower() == "sarsa":
                self.strategy = SarsaStrategy()
            elif strategy.lower() in {"q_lambda", "qlambda"}:
                self.strategy = QLambdaStrategy()
            elif strategy.lower() in {"actor_critic", "actor-critic"}:
                self.strategy = ActorCriticStrategy()
            else:
                self.strategy = QLearningStrategy()
        else:
            self.strategy = strategy
        self.values: Dict[Tuple[int, ...], Dict[int, float]] = {}
        self.path = path
        self.episodes = 0
        self.rewards: list[float] = []
        if self.path:
            self.load(self.path)

    # ------------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, ...],
        reward: float,
        next_state: Tuple[int, ...] | None = None,
        action: int = 1,
    ) -> float:
        """Update ``state`` with ``reward`` and optional ``next_state``."""
        extra = 0.0
        try:
            if len(state) >= 17:
                if next_state is not None and len(next_state) >= len(state):
                    extra = (
                        (next_state[-4] - state[-4]) / 10.0
                        + (next_state[-3] - state[-3]) / 10.0
                    )
                else:
                    extra = state[-4] / 10.0 + state[-3] / 10.0
        except Exception:
            extra = 0.0

        reward += extra
        q = self.strategy.update(
            self.values,
            state,
            action,
            reward,
            next_state,
            self.alpha,
            self.gamma,
        )
        self.episodes += 1
        self.rewards.append(reward)
        if len(self.rewards) > 100:
            self.rewards.pop(0)
        if self.adaptive:
            self.alpha = max(0.1, 1.0 / (1 + self.episodes))
            if len(self.rewards) > 1:
                var = statistics.pvariance(self.rewards)
                self.gamma = max(0.5, min(0.99, 0.9 / (1 + var)))
        if self.epsilon_schedule:
            self.epsilon = self.epsilon_schedule(self.episodes, self.epsilon)
        if self.temperature_schedule:
            self.temperature = self.temperature_schedule(self.episodes, self.temperature)
        if self.path:
            self.save(self.path)
        return q

    def score(self, state: Tuple[int, ...]) -> float:
        """Return the learned value for ``state``."""
        return self.strategy.value(self.values, state)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """Choose an action using the configured exploration strategy."""
        actions = self.values.get(state)
        if not actions:
            actions = {0: 0.0, 1: 0.0}
        if self.exploration == "softmax":
            t = max(0.01, self.temperature)
            probs = [math.exp(v / t) for v in actions.values()]
            total = sum(probs)
            thresh = random.random() * total
            cumulative = 0.0
            for (act, _), p in zip(actions.items(), probs):
                cumulative += p
                if cumulative >= thresh:
                    return act
            return list(actions.keys())[-1]
        if random.random() < self.epsilon:
            return random.choice(list(actions.keys()))
        return max(actions, key=actions.get)

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        try:
            with open(fp, "wb") as fh:
                pickle.dump(self.values, fh)
        except Exception:
            raise

    def load(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp or not os.path.exists(fp):
            return
        try:
            with open(fp, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                new_table: Dict[Tuple[int, ...], Dict[int, float]] = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        new_table[tuple(k) if not isinstance(k, tuple) else k] = {
                            int(a): float(q) for a, q in v.items()
                        }
                    else:
                        new_table[tuple(k) if not isinstance(k, tuple) else k] = {1: float(v)}
                self.values = new_table
        except Exception:
            raise


__all__ = [
    "RLStrategy",
    "QLearningStrategy",
    "QLambdaStrategy",
    "SarsaStrategy",
    "ActorCriticStrategy",
    "SelfImprovementPolicy",
]
