from __future__ import annotations

import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from dynamic_path_router import resolve_path

from .rl_integration import ReplayBuffer, Experience
from .rlhf import RLHFPolicyManager
from .genetic_hatchery import GeneticHatchery


def _softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


class BanditScout(RLHFPolicyManager):
    """Lightweight wrapper using RLHFPolicyManager for a tactic."""

    def score(self, action: str) -> float:
        return self.weights.get(action, 0.0)


class PPOBrain:
    """Minimal PPO-style policy using linear preferences."""

    def __init__(self, state_dim: int, actions: List[str], *, lr: float = 0.1, entropy: float = 0.05) -> None:
        self.state_dim = state_dim
        self.actions = actions
        self.lr = lr
        self.entropy = entropy
        self.params: List[List[float]] = [[0.0] * state_dim for _ in actions]
        self.snapshots: List[List[List[float]]] = []
        self.last_delta: float = 0.0

    # ------------------------------------------------------------------
    def _dot(self, w: List[float], s: List[float]) -> float:
        return sum(wi * si for wi, si in zip(w, s))

    def _policy(self, state: List[float]) -> List[float]:
        logits = [self._dot(w, state) for w in self.params]
        return _softmax(logits)

    def select(self, state: List[float]) -> Tuple[str, float]:
        probs = self._policy(state)
        noise = [random.random() * self.entropy for _ in probs]
        mixed = [p * (1 - self.entropy) + n for p, n in zip(probs, noise)]
        idx = mixed.index(max(mixed))
        return self.actions[idx], probs[idx]

    def update(self, state: List[float], action: str, reward: float) -> None:
        if action not in self.actions:
            return
        prev = [row[:] for row in self.params]
        probs = self._policy(state)
        idx = self.actions.index(action)
        baseline = sum(p * self._dot(w, state) for p, w in zip(probs, self.params))
        advantage = reward - baseline
        grad = [advantage * s for s in state]
        for i in range(len(self.actions)):
            if i == idx:
                self.params[i] = [w + self.lr * g for w, g in zip(self.params[i], grad)]
            else:
                self.params[i] = [w - self.lr * g * probs[i] for w, g in zip(self.params[i], grad)]
        self.snapshots.append([row[:] for row in self.params])
        delta = 0.0
        for a, b in zip(self.params, prev):
            for w_new, w_old in zip(a, b):
                delta += abs(w_new - w_old)
        self.last_delta = delta


class PolicyLearner:
    """Coordinate bandit scouts with a PPO-style brain."""

    def __init__(
        self,
        actions: List[str],
        tactics: Dict[str, str],
        *,
        state_dim: int = 4,
        lr: float = 0.1,
        min_epochs: int = 3,
        plateau_patience: int = 3,
        ga_generations: int = 5,
        ga_population: int = 10,
        weights_path: str | None = None,
    ) -> None:
        self.actions = actions
        self.tactics = tactics
        self.brain = PPOBrain(state_dim, actions, lr=lr)
        if weights_path is None:
            weights_path = resolve_path("neurosales") / "policy_params.json"
        else:
            weights_path = Path(weights_path)
        if weights_path.exists():
            try:
                with open(weights_path) as f:
                    self.brain.params = json.load(f)
            except Exception:
                pass
        self.scouts: Dict[str, BanditScout] = {t: BanditScout() for t in set(tactics.values())}
        self.buffer = ReplayBuffer(max_size=50)
        self.hatchery = GeneticHatchery(actions, state_dim, pop_size=ga_population)
        self.mode = "rl"
        self.mode_epochs = 0
        self.min_epochs = min_epochs
        self.plateau_patience = plateau_patience
        self.ga_generations = ga_generations
        self.plateau_count = 0

    def train_from_dataset(self, path: str) -> None:
        """Load state, action, reward tuples and update the brain."""
        if not path:
            return
        if path.endswith(".csv"):
            import csv

            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                state_keys = [k for k in reader.fieldnames or [] if k not in {"action", "reward"}]
                for row in reader:
                    state = [float(row[k]) for k in state_keys]
                    action = row["action"]
                    reward = float(row["reward"])
                    self.brain.update(state, action, reward)
        else:
            with open(path) as f:
                data = json.load(f)
            for item in data:
                state = [float(x) for x in item.get("state", [])]
                action = item.get("action", "")
                reward = float(item.get("reward", 0.0))
                self.brain.update(state, action, reward)

    # ------------------------------------------------------------------
    def act(self, state: List[float]) -> Tuple[str, str, float]:
        base_action, base_conf = self.brain.select(state)
        logits: Dict[str, float] = {}
        for a in self.actions:
            idx = self.actions.index(a)
            tactic = self.tactics.get(a, "")
            scout = self.scouts.get(tactic)
            weight = scout.score(a) if scout else 0.0
            logits[a] = self.brain._dot(self.brain.params[idx], state) + weight
        probs = _softmax(list(logits.values()))
        keys = list(logits.keys())
        idx = probs.index(max(probs))
        action = keys[idx]
        conf = probs[idx]
        style = "punch" if conf > 0.66 else "hedge" if conf > 0.33 else "probe"
        self.buffer.add(Experience(tuple(state), action, 0.0, tuple(state)))
        return action, style, conf

    # ------------------------------------------------------------------
    def learn(
        self,
        reward: float,
        ctr: float = 0.0,
        sentiment: float = 0.0,
        session: float = 0.0,
    ) -> None:
        if len(self.buffer) == 0:
            return
        exp = self.buffer.sample(1)[0]
        self.brain.update(list(exp.state), exp.action, reward)
        for fb in self.buffer.sample_feedback(1):
            self.brain.update(list(fb.state), fb.action, fb.reward)
        delta = self.brain.last_delta
        tactic = self.tactics.get(exp.action)
        scout = self.scouts.get(tactic)
        if scout:
            scout.record_result(exp.action, ctr=ctr, sentiment=sentiment, session=session)
        # mode management -------------------------------------------------
        self.mode_epochs += 1
        if self.mode == "rl":
            if delta < 1e-6:
                self.plateau_count += 1
            else:
                self.plateau_count = 0
            if (
                self.mode_epochs >= self.min_epochs
                and self.plateau_count >= self.plateau_patience
            ):
                self.mode = "ga"
                self.mode_epochs = 0
                self.plateau_count = 0
                self.hatchery.population[0] = [row[:] for row in self.brain.params]
        else:  # GA mode
            self.hatchery.next_generation()
            if self.mode_epochs >= self.ga_generations and self.mode_epochs >= self.min_epochs:
                self.mode = "rl"
                self.mode_epochs = 0
                self.brain.params = self.hatchery.best_genome()

