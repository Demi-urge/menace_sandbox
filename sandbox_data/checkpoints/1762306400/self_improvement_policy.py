from __future__ import annotations

"""Q-learning policy predicting ROI improvement from self-improvement cycles."""

import abc
import atexit
import hashlib  # used in _compute_checksum
from typing import Dict, Tuple, Optional, Callable
import pickle
import os
import json
import random
import math
import statistics
import warnings
import logging
from typing import List
from dataclasses import dataclass, asdict
from pathlib import Path
from sandbox_settings import SandboxSettings
from self_improvement.prompt_memory import load_prompt_penalties
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - fallback if torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _compute_checksum(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RLConfig:
    """Common configuration for all strategies."""

    state_dim: int | None = None
    action_dim: int | None = None


class RLStrategy(abc.ABC):
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

    @abc.abstractmethod
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
        """Update the value table for a transition."""
        ...

    @staticmethod
    def value(table: Dict[Tuple[int, ...], Dict[int, float]], state: Tuple[int, ...]) -> float:
        values = table.get(state, {})
        return max(values.values()) if values else 0.0


@dataclass(eq=True, init=False)
class PolicyConfig:
    """Configuration for :class:`SelfImprovementPolicy` hyperparameters."""

    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.1
    temperature: float = 1.0
    exploration: str = "epsilon_greedy"
    adaptive: bool = False

    def __init__(
        self,
        alpha: float | None = None,
        gamma: float | None = None,
        epsilon: float | None = None,
        temperature: float | None = None,
        exploration: str | None = None,
        adaptive: bool = False,
        settings: SandboxSettings | None = None,
    ) -> None:
        settings = settings or SandboxSettings()
        self.alpha = settings.policy_alpha if alpha is None else alpha
        self.gamma = settings.policy_gamma if gamma is None else gamma
        self.epsilon = settings.policy_epsilon if epsilon is None else epsilon
        self.temperature = (
            settings.policy_temperature if temperature is None else temperature
        )
        self.exploration = (
            settings.policy_exploration if exploration is None else exploration
        )
        self.adaptive = adaptive


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
    hidden_dim: int = 32
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    entropy_beta: float = 0.01


class ActorCriticStrategy(RLStrategy):
    """Actor-critic strategy with separate networks and advantage estimation.

    Falls back to a tabular implementation when PyTorch is unavailable."""

    Config = ActorCriticConfig

    def __init__(self, config: ActorCriticConfig | None = None) -> None:
        super().__init__(config or self.Config())
        if torch is None:  # pragma: no cover - torch not available
            self.state_values: Dict[Tuple[int, ...], float] = {}
        else:
            cfg = self.config
            self.state_dim = cfg.state_dim
            self.action_dim = cfg.action_dim or 2
            self.hidden_dim = cfg.hidden_dim
            self.actor_lr = cfg.actor_lr
            self.critic_lr = cfg.critic_lr
            self.entropy_beta = cfg.entropy_beta
            self.actor: Optional[nn.Module] = None
            self.critic: Optional[nn.Module] = None
            self.actor_opt: Optional[torch.optim.Optimizer] = None
            self.critic_opt: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if torch is None:
            return
        if self.actor is None or self.critic is None:
            if self.config.state_dim is not None and dim != self.config.state_dim:
                raise ValueError(
                    f"state has dimension {dim} but expected {self.config.state_dim}"
                )
            self.state_dim = dim
            self.config.state_dim = dim
            self.actor = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_opt = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr
            )

    # ------------------------------------------------------------------
    def predict(self, state: Tuple[int, ...]) -> torch.Tensor:
        self._ensure_model(len(state))
        assert torch is not None and self.actor is not None
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(s)
        return logits.squeeze(0)

    # ------------------------------------------------------------------
    def select_action(
        self,
        state: Tuple[int, ...],
        *,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        exploration: str = "epsilon_greedy",
    ) -> int:
        if torch is None or self.actor is None:
            actions = list(range(self.config.action_dim or 2))
            if random.random() < epsilon:
                return random.choice(actions)
            return actions[0]
        logits = self.predict(state)
        actions = list(range(self.action_dim))
        if exploration == "softmax":
            t = max(0.01, temperature)
            probs = F.softmax(logits / t, dim=0).tolist()
            return random.choices(actions, weights=probs, k=1)[0]
        if random.random() < epsilon:
            return random.choice(actions)
        return int(torch.argmax(logits).item())

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
        if torch is None or self.actor is None or self.critic is None:
            state_table = table.setdefault(state, {})
            q = state_table.get(action, 0.0)
            v = getattr(self, "state_values", {}).get(state, 0.0)
            next_v = (
                getattr(self, "state_values", {}).get(next_state, 0.0)
                if next_state is not None
                else 0.0
            )
            td_error = reward + gamma * next_v - v
            getattr(self, "state_values", {})[state] = v + alpha * td_error
            state_table[action] = q + alpha * td_error
            return state_table[action]

        self._ensure_model(len(state))
        assert self.actor_opt is not None and self.critic_opt is not None
        s_t = torch.tensor(state, dtype=torch.float32)
        v = self.critic(s_t).squeeze(0)
        logits = self.actor(s_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy()
        with torch.no_grad():
            if next_state is not None:
                ns_t = torch.tensor(next_state, dtype=torch.float32)
                next_v = self.critic(ns_t).squeeze(0)
            else:
                next_v = torch.tensor(0.0)
            advantage = reward + gamma * next_v - v

        actor_loss = -(log_prob * advantage.detach() + self.entropy_beta * entropy)
        critic_loss = advantage.pow(2)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return float(v.detach().item())


@dataclass
class DQNConfig(RLConfig):
    """Configuration for DQN-based strategies."""

    hidden_dim: int = 32
    lr: float = 1e-3
    batch_size: int = 32
    capacity: int = 1000
    target_sync: int = 10
    save_interval: int = 0


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
        self.save_interval = cfg.save_interval
        self.memory: List[Tuple[torch.Tensor, int, float, Optional[torch.Tensor], bool]] = []
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._save_callback: Optional[Callable[[], None]] = None
        self._updates = 0
        atexit.register(self._atexit_save)

    def set_save_callback(self, cb: Callable[[], None]) -> None:
        self._save_callback = cb

    def _atexit_save(self) -> None:
        if self._save_callback is not None:
            try:
                self._save_callback()
            except Exception:
                logger.exception("save callback failed during atexit")

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
        self._updates += 1
        if (
            self.save_interval
            and self._save_callback is not None
            and self._updates % self.save_interval == 0
        ):
            try:
                self._save_callback()
            except Exception:
                logger.exception("save callback failed during update")
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


if torch is None:
    class _FallbackDQNStrategy(QLearningStrategy):
        """Fallback strategy when PyTorch is unavailable."""

        Config = DQNConfig
        _deterministic_fallback = True

    DeepQLearningStrategy = _FallbackDQNStrategy  # noqa: F811
    DQNStrategy = _FallbackDQNStrategy  # noqa: F811
    DoubleDQNStrategy = _FallbackDQNStrategy  # noqa: F811

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
        config: PolicyConfig | None = None,
        save_interval: int = 0,
    ) -> None:
        if config is not None:
            alpha = config.alpha
            gamma = config.gamma
            exploration = config.exploration
            epsilon = config.epsilon
            temperature = config.temperature
            adaptive = config.adaptive
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive
        self.base_epsilon = epsilon
        self.epsilon = epsilon
        self.urgency = 0.0
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
        random.seed(0)
        if torch is not None:
            torch.manual_seed(0)
        self.values: Dict[Tuple[int, ...], Dict[int, float]] = {}
        self.path = path
        self.episodes = 0
        self.rewards: list[float] = []
        self.save_interval = save_interval
        if hasattr(self.strategy, "save_interval") and not getattr(
            self.strategy, "save_interval", 0
        ):
            setattr(self.strategy, "save_interval", save_interval)
        if hasattr(self.strategy, "set_save_callback"):
            self.strategy.set_save_callback(self._save_model_if_path)
        atexit.register(self._graceful_shutdown)
        if self.path:
            self.load(self.path)

    # ------------------------------------------------------------------
    def adjust_for_momentum(self, momentum: float) -> None:
        """Adjust exploration and urgency based on momentum."""
        if momentum < 0:
            delta = -momentum
            self.urgency = min(1.0, self.urgency + delta)
            self.epsilon = min(1.0, self.base_epsilon + delta)
        else:
            delta = momentum
            self.urgency = max(0.0, self.urgency - delta)
            self.epsilon = max(self.base_epsilon, self.epsilon - delta)

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
        if self.path and self.save_interval and self.episodes % self.save_interval == 0:
            self.save_model(self.path)
        return q

    def _save_model_if_path(self) -> None:
        if self.path:
            try:
                self.save_model(self.path)
            except Exception:
                logger.exception("failed to save model to %s", self.path)

    def _graceful_shutdown(self) -> None:
        self._save_model_if_path()

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
        actions = dict(self.values.get(state, {0: 0.0, 1: 0.0}))
        penalties = load_prompt_penalties()
        settings = SandboxSettings()
        penalised = {
            act
            for act in actions
            if penalties.get(str(act), 0) >= settings.prompt_failure_threshold
        }
        for act in penalised:
            actions[act] *= settings.prompt_penalty_multiplier
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
            eligible = [a for a in actions if a not in penalised]
            if eligible:
                return random.choice(eligible)
            return random.choice(list(actions.keys()))
        return max(actions, key=actions.get)

    # ------------------------------------------------------------------
    def get_config(self) -> PolicyConfig:
        """Return current hyperparameters."""

        return PolicyConfig(
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            temperature=self.temperature,
            exploration=self.exploration,
            adaptive=self.adaptive,
        )

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialize policy configuration and values to JSON."""
        data: Dict[str, object] = {
            "version": 1,
            "hyperparameters": asdict(self.get_config()),
            "strategy": self.strategy.__class__.__name__,
            "values": [
                {"state": list(k), "actions": v} for k, v in self.values.items()
            ],
            "save_interval": self.save_interval,
        }
        cfg = getattr(self.strategy, "config", None)
        if cfg is not None:
            data["strategy_config"] = asdict(cfg)
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "SelfImprovementPolicy":
        """Deserialize a policy from JSON produced by :meth:`to_json`."""
        obj = json.loads(data)
        obj.get("version", 1)
        strat_name = obj.get("strategy")
        strat_cls = _STRATEGY_CLASSES.get(str(strat_name).lower(), QLearningStrategy)
        cfg_dict = obj.get("strategy_config")
        if cfg_dict and hasattr(strat_cls, "Config"):
            cfg = getattr(strat_cls, "Config")(**cfg_dict)
            strat = strat_cls(config=cfg)  # type: ignore[arg-type]
        else:
            strat = strat_cls()
        hyper = obj.get("hyperparameters")
        save_interval = int(obj.get("save_interval", 0))
        if hyper:
            hp = PolicyConfig(**hyper)
            policy = cls(config=hp, strategy=strat, save_interval=save_interval)
        else:
            policy = cls(
                alpha=obj.get("alpha", 0.5),
                gamma=obj.get("gamma", 0.9),
                epsilon=obj.get("epsilon", 0.1),
                temperature=obj.get("temperature", 1.0),
                exploration=obj.get("exploration", "epsilon_greedy"),
                adaptive=obj.get("adaptive", False),
                strategy=strat,
                save_interval=save_interval,
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
        path = Path(fp)
        full = Path(resolve_path(path.parent)).joinpath(path.name)
        with full.open("wb") as fh:
            pickle.dump(self.values, fh)

    def load_q_table(self, fp: str) -> None:
        path = Path(resolve_path(fp))
        with path.open("rb") as fh:
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

    def save_model(self, path: str) -> None:
        """Save policy and strategy weights to ``path`` as JSON."""
        path_obj = Path(path)
        with path_obj.open("w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        resolved = Path(resolve_path(str(path_obj)))
        self.save_dqn_weights(str(resolved))
        base_path = resolved.with_suffix("")
        checksums: Dict[str, str] = {resolved.name: _compute_checksum(str(resolved))}
        for ext in [
            ".pt",
            ".target.pt",
            ".opt.pt",
            ".actor.pt",
            ".critic.pt",
            ".actor.opt.pt",
            ".critic.opt.pt",
            ".replay.json",
        ]:
            fp = base_path.with_suffix(ext)
            if fp.exists():
                checksums[fp.name] = _compute_checksum(str(fp))
        with base_path.with_suffix(".checksum").open("w", encoding="utf-8") as fh:
            json.dump({"version": 1, "files": checksums}, fh)

    @classmethod
    def load_model(cls, path: str) -> "SelfImprovementPolicy":
        """Load policy and associated weights from ``path``."""

        path_obj = Path(resolve_path(path))
        with path_obj.open("r", encoding="utf-8") as fh:
            data = fh.read()
        policy = cls.from_json(data)
        policy.path = str(path_obj)
        base_path = path_obj.with_suffix("")
        checksum_path = base_path.with_suffix(".checksum")
        if checksum_path.exists():
            with checksum_path.open("r", encoding="utf-8") as fh:
                info = json.load(fh)
            files = info.get("files", {})
            expected = files.get(path_obj.name)
            if expected and expected != _compute_checksum(str(path_obj)):
                raise ValueError("Checksum mismatch for policy file")
            for filename, fp in files.items():
                if filename == path_obj.name:
                    continue
                full = path_obj.with_name(filename)
                if full.exists() and fp != _compute_checksum(str(full)):
                    raise ValueError(f"Checksum mismatch for {filename}")
        policy.load_dqn_weights(str(path_obj))
        return policy

    def save_dqn_weights(self, base: str) -> None:
        if torch is None:
            return
        strat = self.strategy
        base_path = Path(resolve_path(base)).with_suffix("")
        model = getattr(strat, "model", None)
        if model is not None:
            torch.save(model.state_dict(), base_path.with_suffix(".pt"))
        target = getattr(strat, "target_model", None)
        if target is not None:
            torch.save(target.state_dict(), base_path.with_suffix(".target.pt"))
        optimizer = getattr(strat, "optimizer", None)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), base_path.with_suffix(".opt.pt"))
        actor = getattr(strat, "actor", None)
        if actor is not None:
            torch.save(actor.state_dict(), base_path.with_suffix(".actor.pt"))
        critic = getattr(strat, "critic", None)
        if critic is not None:
            torch.save(critic.state_dict(), base_path.with_suffix(".critic.pt"))
        actor_opt = getattr(strat, "actor_opt", None)
        if actor_opt is not None:
            torch.save(actor_opt.state_dict(), base_path.with_suffix(".actor.opt.pt"))
        critic_opt = getattr(strat, "critic_opt", None)
        if critic_opt is not None:
            torch.save(critic_opt.state_dict(), base_path.with_suffix(".critic.opt.pt"))
        memory = getattr(strat, "memory", None)
        if memory is not None:
            data = []
            for state, action, reward, next_state, done in memory:
                data.append(
                    {
                        "state": state.tolist(),
                        "action": int(action),
                        "reward": float(reward),
                        "next_state": None if next_state is None else next_state.tolist(),
                        "done": bool(done),
                    }
                )
            with base_path.with_suffix(".replay.json").open("w", encoding="utf-8") as fh:
                json.dump(data, fh)

    def load_dqn_weights(self, base: str) -> None:
        if torch is None:
            return
        strat = self.strategy

        def ensure(dim: int) -> None:
            if hasattr(strat, "_ensure_model"):
                strat._ensure_model(dim)  # type: ignore[attr-defined]

        def infer_dim(state_dict: Dict[str, torch.Tensor]) -> int:
            first = next(iter(state_dict.values()))
            return int(first.shape[1]) if first.ndim > 1 else int(first.shape[0])

        base_path = Path(resolve_path(base)).with_suffix("")
        model_path = base_path.with_suffix(".pt")
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            if getattr(strat, "config", None) and getattr(strat.config, "state_dim", None) is None:
                dim = infer_dim(state_dict)
                ensure(dim)
            else:
                ensure(getattr(strat.config, "state_dim", 1))
            model = getattr(strat, "model", None)
            if model is not None:
                model.load_state_dict(state_dict)
        target_path = base_path.with_suffix(".target.pt")
        if target_path.exists() and hasattr(strat, "target_model"):
            target_state = torch.load(target_path, map_location="cpu")
            if getattr(strat, "config", None) and getattr(strat.config, "state_dim", None) is None:
                dim = infer_dim(target_state)
                ensure(dim)
            else:
                ensure(getattr(strat.config, "state_dim", 1))
            target = getattr(strat, "target_model", None)
            if target is not None:
                target.load_state_dict(target_state)
        opt_path = base_path.with_suffix(".opt.pt")
        if opt_path.exists() and hasattr(strat, "optimizer"):
            opt_state = torch.load(opt_path, map_location="cpu")
            optimizer = getattr(strat, "optimizer", None)
            if optimizer is not None:
                optimizer.load_state_dict(opt_state)
        actor_path = base_path.with_suffix(".actor.pt")
        if actor_path.exists():
            state_dict = torch.load(actor_path, map_location="cpu")
            if getattr(strat, "config", None) and getattr(strat.config, "state_dim", None) is None:
                dim = infer_dim(state_dict)
                ensure(dim)
            else:
                ensure(getattr(strat.config, "state_dim", 1))
            actor = getattr(strat, "actor", None)
            if actor is not None:
                actor.load_state_dict(state_dict)
        critic_path = base_path.with_suffix(".critic.pt")
        if critic_path.exists():
            state_dict = torch.load(critic_path, map_location="cpu")
            if getattr(strat, "config", None) and getattr(strat.config, "state_dim", None) is None:
                dim = infer_dim(state_dict)
                ensure(dim)
            else:
                ensure(getattr(strat.config, "state_dim", 1))
            critic = getattr(strat, "critic", None)
            if critic is not None:
                critic.load_state_dict(state_dict)
        actor_opt_path = base_path.with_suffix(".actor.opt.pt")
        if actor_opt_path.exists() and hasattr(strat, "actor_opt"):
            actor_opt_state = torch.load(actor_opt_path, map_location="cpu")
            actor_opt = getattr(strat, "actor_opt", None)
            if actor_opt is not None:
                actor_opt.load_state_dict(actor_opt_state)
        critic_opt_path = base_path.with_suffix(".critic.opt.pt")
        if critic_opt_path.exists() and hasattr(strat, "critic_opt"):
            critic_opt_state = torch.load(critic_opt_path, map_location="cpu")
            critic_opt = getattr(strat, "critic_opt", None)
            if critic_opt is not None:
                critic_opt.load_state_dict(critic_opt_state)
        replay_path = base_path.with_suffix(".replay.json")
        if replay_path.exists() and hasattr(strat, "memory"):
            try:
                with replay_path.open("r", encoding="utf-8") as fh:
                    entries = json.load(fh)
                if not isinstance(entries, list):
                    raise ValueError("invalid replay data")
                memory = []
                for it in entries:
                    if not isinstance(it, dict):
                        raise ValueError("invalid replay item")
                    state = it.get("state")
                    action = it.get("action")
                    reward = it.get("reward")
                    next_state = it.get("next_state")
                    done = it.get("done")
                    if not (
                        isinstance(state, list)
                        and isinstance(action, int)
                        and isinstance(reward, (int, float))
                        and (next_state is None or isinstance(next_state, list))
                        and isinstance(done, bool)
                    ):
                        raise ValueError("invalid replay item")
                    s = torch.tensor(state, dtype=torch.float32) if torch is not None else state
                    ns = (
                        torch.tensor(next_state, dtype=torch.float32)
                        if (torch is not None and next_state is not None)
                        else None
                    )
                    memory.append((s, int(action), float(reward), ns, bool(done)))
                strat.memory = memory
            except Exception as exc:
                logger.warning("skipping replay memory: %s", exc)

    def save(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        path_obj = Path(fp)
        try:
            self.save_q_table(str(path_obj))
            resolved = Path(resolve_path(str(path_obj)))
            self.save_dqn_weights(str(resolved))
        except Exception:
            raise

    def load(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        path_obj = Path(resolve_path(fp))
        try:
            if path_obj.exists():
                self.load_q_table(str(path_obj))
            self.load_dqn_weights(str(path_obj))
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
    "PolicyConfig",
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
