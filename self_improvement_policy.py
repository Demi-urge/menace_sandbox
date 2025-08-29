from __future__ import annotations

"""Q-learning policy predicting ROI improvement from self-improvement cycles."""

from typing import Dict, Tuple, Optional, Callable
import pickle
import os
import json
import random
import math
import statistics
import warnings
from typing import List
from dataclasses import dataclass, asdict

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - fallback if torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclass
class RLConfig:
    """Common configuration for all strategies."""

    state_dim: int | None = None
    action_dim: int | None = None


class RLStrategy:
    """Base class for reinforcement learning update strategies."""

    def __init__(self, config: RLConfig | None = None) -> None:
        self.config = config or RLConfig()

    def _validate(self, state: Tuple[int, ...], action: int) -> None:
        if self.config.state_dim is not None and len(state) != self.config.state_dim:
            raise ValueError(
                f"state has dimension {len(state)} but expected {self.config.state_dim}"
            )
        if self.config.action_dim is not None and not (
            0 <= action < self.config.action_dim
        ):
            raise ValueError(
                f"action {action} outside range(0, {self.config.action_dim})"
            )

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


@dataclass
class QLearningConfig(RLConfig):
    """Configuration for tabular Q-learning style strategies."""


class QLearningStrategy(RLStrategy):
    """Standard Q-learning update."""

    Config = QLearningConfig

    def __init__(self, config: QLearningConfig | None = None) -> None:
        super().__init__(config or self.Config())

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
        self._validate(state, action)
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

    Config = QLearningConfig

    def __init__(self, lam: float = 0.9, config: QLearningConfig | None = None) -> None:
        super().__init__(config or self.Config())
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
        self._validate(state, action)
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

    Config = QLearningConfig

    def __init__(self, config: QLearningConfig | None = None) -> None:
        super().__init__(config or self.Config())

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
        self._validate(state, action)
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        next_val = 0.0
        if next_state is not None:
            next_val = table.get(next_state, {}).get(action, 0.0)
        q += alpha * (reward + gamma * next_val - q)
        state_table[action] = q
        return q


@dataclass
class ActorCriticConfig(RLConfig):
    """Configuration for :class:`ActorCriticStrategy`."""


class ActorCriticStrategy(RLStrategy):
    """Simplified actor-critic update."""

    Config = ActorCriticConfig

    def __init__(self, config: ActorCriticConfig | None = None) -> None:
        super().__init__(config or self.Config())
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
        self._validate(state, action)
        state_table = table.setdefault(state, {})
        q = state_table.get(action, 0.0)
        v = self.state_values.get(state, 0.0)
        next_v = self.state_values.get(next_state, 0.0) if next_state is not None else 0.0
        td_error = reward + gamma * next_v - v
        self.state_values[state] = v + alpha * td_error
        state_table[action] = q + alpha * td_error
        return state_table[action]


@dataclass
class DQNConfig(RLConfig):
    """Configuration for DQN-based strategies."""

    hidden_dim: int = 32
    lr: float = 1e-3
    batch_size: int = 32
    capacity: int = 1000
    target_sync: int = 10


class DQNStrategy(RLStrategy):
    """Simple Deep Q-Network strategy using PyTorch."""

    Config = DQNConfig

    def __init__(self, config: DQNConfig | None = None) -> None:
        cfg = config or self.Config()
        super().__init__(cfg)
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim or 2
        self.hidden_dim = cfg.hidden_dim
        self.lr = cfg.lr
        self.batch_size = cfg.batch_size
        self.capacity = cfg.capacity
        self.memory: List[Tuple[torch.Tensor, int, float, Optional[torch.Tensor], bool]] = []
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if self.model is None:
            if self.config.state_dim is not None and dim != self.config.state_dim:
                raise ValueError(
                    f"state has dimension {dim} but expected {self.config.state_dim}"
                )
            self.state_dim = dim
            self.config.state_dim = dim
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
        self._validate(state, action)
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
    def value(
        self, table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]
    ) -> float:  # type: ignore[override]
        q_vals = self.predict(state)
        return float(q_vals.max().item())


class DoubleDQNStrategy(RLStrategy):
    """Double DQN with target network using PyTorch."""

    Config = DQNConfig

    def __init__(self, config: DQNConfig | None = None) -> None:
        cfg = config or self.Config()
        super().__init__(cfg)
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim or 2
        self.hidden_dim = cfg.hidden_dim
        self.lr = cfg.lr
        self.batch_size = cfg.batch_size
        self.capacity = cfg.capacity
        self.target_sync = cfg.target_sync
        self.memory: List[Tuple[torch.Tensor, int, float, Optional[torch.Tensor], bool]] = []
        self.model: Optional[nn.Module] = None
        self.target_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.steps = 0

    # ------------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if self.model is None:
            if self.config.state_dim is not None and dim != self.config.state_dim:
                raise ValueError(
                    f"state has dimension {dim} but expected {self.config.state_dim}"
                )
            self.state_dim = dim
            self.config.state_dim = dim
            self.model = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
            self.target_model = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
            self.target_model.load_state_dict(self.model.state_dict())
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
        self._validate(state, action)
        self._ensure_model(len(state))
        assert (
            self.model is not None
            and self.target_model is not None
            and self.optimizer is not None
        )

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
            next_actions = self.model(next_states).max(1).indices
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        with torch.no_grad():
            cur_q = self.model(s_t.unsqueeze(0))[0, action]
        return float(cur_q.item())

    # ------------------------------------------------------------------
    def value(
        self, table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]
    ) -> float:  # type: ignore[override]
        q_vals = self.predict(state)
        return float(q_vals.max().item())


class DeepQLearningStrategy(RLStrategy):
    """Simpler online Deep Q-learning using PyTorch."""

    Config = DQNConfig

    def __init__(self, config: DQNConfig | None = None) -> None:
        cfg = config or self.Config()
        super().__init__(cfg)
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim or 2
        self.hidden_dim = cfg.hidden_dim
        self.lr = cfg.lr
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if self.model is None:
            if self.config.state_dim is not None and dim != self.config.state_dim:
                raise ValueError(
                    f"state has dimension {dim} but expected {self.config.state_dim}"
                )
            self.state_dim = dim
            self.config.state_dim = dim
            self.model = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
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
        self._validate(state, action)
        self._ensure_model(len(state))
        assert self.model is not None and self.optimizer is not None

        s_t = torch.tensor(state, dtype=torch.float32)
        q_vals = self.model(s_t)
        q_sa = q_vals[action]

        with torch.no_grad():
            if next_state is not None:
                ns_t = torch.tensor(next_state, dtype=torch.float32)
                next_q = torch.max(self.model(ns_t))
            else:
                next_q = torch.tensor(0.0)
            target = reward + gamma * next_q

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(q_sa.detach().item())

    # ------------------------------------------------------------------
    def value(
        self, table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]
    ) -> float:  # type: ignore[override]
        self._ensure_model(len(state))
        assert self.model is not None
        with torch.no_grad():
            return float(torch.max(self.model(torch.tensor(state, dtype=torch.float32))).item())


# ---------------------------------------------------------------------------
# RL strategy factory utilities
# ---------------------------------------------------------------------------
_STRATEGY_CLASSES: Dict[str, type[RLStrategy]] = {
    "q_learning": QLearningStrategy,
    "q-learning": QLearningStrategy,
    "sarsa": SarsaStrategy,
    "q_lambda": QLambdaStrategy,
    "qlambda": QLambdaStrategy,
    "actor_critic": ActorCriticStrategy,
    "actor-critic": ActorCriticStrategy,
    "deep_q": DeepQLearningStrategy,
    "dqn": DQNStrategy,
    "double_dqn": DoubleDQNStrategy,
    "double-dqn": DoubleDQNStrategy,
    "ddqn": DoubleDQNStrategy,
}


def strategy_factory(
    strategy_name: str | None = None,
    *,
    config_path: str | None = None,
    env_var: str = "SELF_IMPROVEMENT_STRATEGY",
) -> RLStrategy:
    """Instantiate an :class:`RLStrategy` from *strategy_name* or config."""

    name = strategy_name
    if not name:
        cfg_path = config_path or os.getenv("SELF_IMPROVEMENT_CONFIG")
        if cfg_path and os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                name = cfg.get("strategy") or None
            except Exception:
                name = None
        if not name:
            name = os.getenv(env_var, "q_learning")

    cls = _STRATEGY_CLASSES.get(str(name).lower(), QLearningStrategy)

    if cls in {DeepQLearningStrategy, DQNStrategy, DoubleDQNStrategy} and (
        torch is None or nn is None
    ):
        warnings.warn(
            "PyTorch not available; falling back to deterministic policy.",
            RuntimeWarning,
        )
        strat = QLearningStrategy()
        setattr(strat, "_deterministic_fallback", True)
        return strat
    return cls()


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
            self.strategy = strategy_factory(strategy)
        else:
            self.strategy = strategy
        if getattr(self.strategy, "_deterministic_fallback", False):
            self.epsilon = 0.0
            self.exploration = "epsilon_greedy"
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
        *,
        synergy_roi_delta: float | None = None,
        synergy_efficiency_delta: float | None = None,
    ) -> float:
        """Update ``state`` with ``reward`` and optional ``next_state``."""
        extra = 0.0
        if synergy_roi_delta is not None or synergy_efficiency_delta is not None:
            extra += (synergy_roi_delta or 0.0) + (synergy_efficiency_delta or 0.0)
        else:
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
    def to_json(self) -> str:
        """Serialize policy configuration and values to JSON."""
        data: Dict[str, object] = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "strategy": self.strategy.__class__.__name__,
            "values": [
                {"state": list(k), "actions": v} for k, v in self.values.items()
            ],
        }
        cfg = getattr(self.strategy, "config", None)
        if cfg is not None:
            data["strategy_config"] = asdict(cfg)
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "SelfImprovementPolicy":
        """Deserialize a policy from JSON produced by :meth:`to_json`."""
        obj = json.loads(data)
        strat_name = obj.get("strategy")
        strat_cls = _STRATEGY_CLASSES.get(str(strat_name).lower(), QLearningStrategy)
        cfg_dict = obj.get("strategy_config")
        if cfg_dict and hasattr(strat_cls, "Config"):
            cfg = getattr(strat_cls, "Config")(**cfg_dict)
            strat = strat_cls(config=cfg)  # type: ignore[arg-type]
        else:
            strat = strat_cls()
        policy = cls(
            alpha=obj.get("alpha", 0.5),
            gamma=obj.get("gamma", 0.9),
            epsilon=obj.get("epsilon", 0.1),
            strategy=strat,
        )
        values: Dict[Tuple[int, ...], Dict[int, float]] = {}
        for item in obj.get("values", []):
            state = tuple(item["state"])
            acts = {int(a): float(q) for a, q in item["actions"].items()}
            values[state] = acts
        policy.values = values
        return policy

    # ------------------------------------------------------------------
    def save_q_table(self, fp: str) -> None:
        with open(fp, "wb") as fh:
            pickle.dump(self.values, fh)

    def load_q_table(self, fp: str) -> None:
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

    def save_dqn_weights(self, base: str) -> None:
        if torch is None or not hasattr(self.strategy, "model"):
            return
        model = getattr(self.strategy, "model", None)
        if model is not None:
            torch.save(model.state_dict(), base + ".pt")
        target = getattr(self.strategy, "target_model", None)
        if target is not None:
            torch.save(target.state_dict(), base + ".target.pt")

    def load_dqn_weights(self, base: str) -> None:
        if torch is None or not hasattr(self.strategy, "model"):
            return
        model_path = base + ".pt"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            if hasattr(self.strategy, "_ensure_model"):
                if (
                    getattr(self.strategy, "config", None)
                    and self.strategy.config.state_dim is None
                ):
                    first = next(iter(state_dict.values()))
                    dim = first.shape[1]
                else:
                    dim = getattr(self.strategy.config, "state_dim", 1)
                self.strategy._ensure_model(dim)  # type: ignore[attr-defined]
            model = getattr(self.strategy, "model", None)
            if model is not None:
                model.load_state_dict(state_dict)
        target_path = base + ".target.pt"
        if os.path.exists(target_path) and hasattr(self.strategy, "target_model"):
            target_state = torch.load(target_path, map_location="cpu")
            if hasattr(self.strategy, "_ensure_model"):
                if (
                    getattr(self.strategy, "config", None)
                    and self.strategy.config.state_dim is None
                ):
                    first = next(iter(target_state.values()))
                    dim = first.shape[1]
                else:
                    dim = getattr(self.strategy.config, "state_dim", 1)
                self.strategy._ensure_model(dim)  # type: ignore[attr-defined]
            target = getattr(self.strategy, "target_model", None)
            if target is not None:
                target.load_state_dict(target_state)

    def save(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        try:
            self.save_q_table(fp)
            base = os.path.splitext(fp)[0]
            self.save_dqn_weights(base)
        except Exception:
            raise

    def load(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        try:
            if os.path.exists(fp):
                self.load_q_table(fp)
            base = os.path.splitext(fp)[0]
            self.load_dqn_weights(base)
        except Exception:
            raise


class ConfigurableSelfImprovementPolicy(SelfImprovementPolicy):
    """Policy selecting strategy via factory using config or env variables."""

    available_strategies = tuple(sorted(set(_STRATEGY_CLASSES.keys())))

    def __init__(
        self,
        *args: object,
        strategy: RLStrategy | str | None = None,
        config_path: str | None = None,
        **kwargs: object,
    ) -> None:
        if isinstance(strategy, RLStrategy):
            strat = strategy
        else:
            strat = strategy_factory(strategy, config_path=config_path)
        super().__init__(*args, strategy=strat, **kwargs)


__all__ = [
    "RLConfig",
    "QLearningConfig",
    "ActorCriticConfig",
    "DQNConfig",
    "RLStrategy",
    "QLearningStrategy",
    "QLambdaStrategy",
    "SarsaStrategy",
    "ActorCriticStrategy",
    "DeepQLearningStrategy",
    "DQNStrategy",
    "DoubleDQNStrategy",
    "SelfImprovementPolicy",
    "ConfigurableSelfImprovementPolicy",
    "strategy_factory",
]
