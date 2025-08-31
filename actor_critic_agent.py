from __future__ import annotations

"""Simple actor-critic agent with replay buffer and persistence."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from sandbox_settings import SandboxSettings


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: List[Transition] = []

    def add(self, transition: Transition) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, batch_size)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)

    def to_json(self) -> List[Tuple[list, int, float, list]]:
        return [
            (t.state.tolist(), t.action, t.reward, t.next_state.tolist())
            for t in self.data
        ]

    @classmethod
    def from_json(
        cls, capacity: int, items: Iterable[Tuple[list, int, float, list]]
    ) -> "ReplayBuffer":
        buf = cls(capacity)
        for s, a, r, ns in items:
            buf.add(
                Transition(
                    np.array(s, dtype=float),
                    int(a),
                    float(r),
                    np.array(ns, dtype=float),
                )
            )
        return buf


class ActorCriticAgent:
    """Minimal actor-critic learner with replay and exploration control."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        settings: SandboxSettings | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.settings = settings or SandboxSettings()
        cfg = self.settings.actor_critic
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.replay = ReplayBuffer(cfg.buffer_size)
        self.checkpoint = Path(cfg.checkpoint_path)
        self.actor = np.zeros((state_size, action_size))
        self.critic = np.zeros(state_size)
        self.steps = 0
        self._load()

    # ------------------------------------------------------------------
    def _policy(self, state: np.ndarray) -> np.ndarray:
        logits = state @ self.actor
        logits -= np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        probs = self._policy(state)
        return int(np.argmax(probs))

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.replay.add(Transition(state, action, reward, next_state))

    def learn(self) -> None:
        if len(self.replay) < self.batch_size:
            return
        batch = self.replay.sample(self.batch_size)
        for trans in batch:
            value = float(self.critic.dot(trans.state))
            next_value = float(self.critic.dot(trans.next_state))
            td_error = trans.reward + self.gamma * next_value - value
            self.critic += self.critic_lr * td_error * trans.state
            probs = self._policy(trans.state)
            grad = -probs
            grad[trans.action] += 1.0
            self.actor += self.actor_lr * td_error * np.outer(trans.state, grad)
        self.epsilon *= self.epsilon_decay
        self.steps += 1
        self._save()

    # ------------------------------------------------------------------
    def _save(self) -> None:
        data = {
            "actor": self.actor.tolist(),
            "critic": self.critic.tolist(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "replay": self.replay.to_json(),
            "hyperparams": {
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "buffer_size": self.replay.capacity,
            },
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_text(json.dumps(data))

    def save(self) -> None:
        self._save()

    def _load(self) -> None:
        if not self.checkpoint.exists():
            return
        data = json.loads(self.checkpoint.read_text())
        self.actor = np.array(data.get("actor", self.actor))
        self.critic = np.array(data.get("critic", self.critic))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.steps = int(data.get("steps", 0))
        hp = data.get("hyperparams", {})
        self.actor_lr = float(hp.get("actor_lr", self.actor_lr))
        self.critic_lr = float(hp.get("critic_lr", self.critic_lr))
        self.gamma = float(hp.get("gamma", self.gamma))
        self.epsilon_decay = float(hp.get("epsilon_decay", self.epsilon_decay))
        self.batch_size = int(hp.get("batch_size", self.batch_size))
        buf_items = data.get("replay", [])
        self.replay = ReplayBuffer.from_json(int(hp.get("buffer_size", len(buf_items))), buf_items)

    # Convenience training step
    def step(self, state: np.ndarray, reward: float, next_state: np.ndarray) -> int:
        action = self.select_action(state)
        self.store(state, action, reward, next_state)
        self.learn()
        return action
