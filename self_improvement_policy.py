from __future__ import annotations

"""Q-learning policy predicting ROI improvement from self-improvement cycles."""

from typing import Dict, Tuple, Optional, Callable
import pickle
import os
import random
import math
import statistics
from typing import List

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - fallback if torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


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


class DQNStrategy(RLStrategy):
    """Simple Deep Q-Network strategy using PyTorch."""

    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: int = 2,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 32,
        capacity: int = 1000,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.capacity = capacity
        self.memory: List[Tuple[torch.Tensor, int, float, Optional[torch.Tensor], bool]] = []
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if self.model is None:
            self.state_dim = dim
            self.model = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    def predict(self, state: Tuple[int, ...]) -> torch.Tensor:
        self._ensure_model(len(state))
        assert self.model is not None
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.model(s)
        return q.squeeze(0)

    # ------------------------------------------------------------------
    def select_action(
        self,
        state: Tuple[int, ...],
        *,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        exploration: str = "epsilon_greedy",
    ) -> int:
        q_vals = self.predict(state)
        actions = list(range(self.action_dim))
        if exploration == "softmax":
            t = max(0.01, temperature)
            probs = F.softmax(q_vals / t, dim=0).tolist()
            return random.choices(actions, weights=probs, k=1)[0]
        if random.random() < epsilon:
            return random.choice(actions)
        return int(torch.argmax(q_vals).item())

    # ------------------------------------------------------------------
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
        self._ensure_model(len(state))
        assert self.model is not None and self.optimizer is not None

        s_t = torch.tensor(state, dtype=torch.float32)
        ns_t = torch.tensor(next_state, dtype=torch.float32) if next_state is not None else None
        done = next_state is None
        self.memory.append((s_t, action, reward, ns_t, done))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([
            b[3] if b[3] is not None else torch.zeros_like(b[0]) for b in batch
        ])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = self.model(next_states).max(1).values
            targets = rewards + gamma * max_next * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            cur_q = self.model(s_t.unsqueeze(0))[0, action]
        return float(cur_q.item())

    # ------------------------------------------------------------------
    @staticmethod
    def value(table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]) -> float:  # type: ignore[override]
        raise NotImplementedError  # value should be computed via predict()


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
            elif strategy.lower() == "dqn":
                if torch is None or nn is None:
                    self.strategy = QLearningStrategy()
                else:
                    self.strategy = DQNStrategy()
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
        if hasattr(self.strategy, "predict"):
            q = self.strategy.predict(state)  # type: ignore[attr-defined]
            return float(torch.max(q).item())
        return self.strategy.value(self.values, state)

    def select_action(self, state: Tuple[int, ...]) -> int:
        """Choose an action using the configured exploration strategy."""
        if hasattr(self.strategy, "select_action"):
            return self.strategy.select_action(
                state,
                epsilon=self.epsilon,
                temperature=self.temperature,
                exploration=self.exploration,
            )
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
    "DQNStrategy",
    "SelfImprovementPolicy",
]
