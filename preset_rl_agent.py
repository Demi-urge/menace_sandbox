from __future__ import annotations

"""Reinforcement learning agent for adjusting sandbox presets."""

import json
import os
import logging
from typing import Dict, Tuple

from .self_improvement_policy import SelfImprovementPolicy
from .roi_tracker import ROITracker

logger = logging.getLogger(__name__)


class PresetRLAgent:
    """Learn how resource changes impact ROI and synergy metrics."""

    ACTIONS: Tuple[Dict[str, int], ...] = (
        {"cpu": 1, "memory": 0, "bandwidth": 0, "threat": 0},
        {"cpu": -1, "memory": 0, "bandwidth": 0, "threat": 0},
        {"cpu": 0, "memory": 1, "bandwidth": 0, "threat": 0},
        {"cpu": 0, "memory": -1, "bandwidth": 0, "threat": 0},
        {"cpu": 0, "memory": 0, "bandwidth": 1, "threat": 0},
        {"cpu": 0, "memory": 0, "bandwidth": -1, "threat": 0},
        {"cpu": 0, "memory": 0, "bandwidth": 0, "threat": 1},
        {"cpu": 0, "memory": 0, "bandwidth": 0, "threat": -1},
        {"cpu": 0, "memory": 0, "bandwidth": 0, "threat": 0},
    )

    def __init__(self, path: str) -> None:
        self.policy = SelfImprovementPolicy(path=path)
        self.state_file = f"{path}.state.json"
        self.prev_state: Tuple[int, int] | None = None
        self.prev_action: int | None = None
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as fh:
                data = json.load(fh)
            st = data.get("state")
            self.prev_state = tuple(st) if st is not None else None
            self.prev_action = data.get("action")
        except Exception as exc:
            logger.warning("Failed to load RL state: %s", exc)
            return

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w") as fh:
                json.dump({"state": self.prev_state, "action": self.prev_action}, fh)
        except Exception as exc:
            logger.warning("Failed to save RL state: %s", exc)

    # ------------------------------------------------------------------
    def _state(self, tracker: ROITracker) -> Tuple[int, int]:
        roi = float(tracker.roi_history[-1]) if tracker.roi_history else 0.0
        syn_vals = tracker.metrics_history.get("synergy_roi", [])
        syn = float(syn_vals[-1]) if syn_vals else 0.0
        roi_s = 1 if roi > 0 else (-1 if roi < 0 else 0)
        syn_s = 1 if syn > 0 else (-1 if syn < 0 else 0)
        return roi_s, syn_s

    def _reward(self, tracker: ROITracker) -> float:
        hist = tracker.roi_history
        if len(hist) < 2:
            return 0.0
        return float(hist[-1]) - float(hist[-2])

    # ------------------------------------------------------------------
    def decide(self, tracker: ROITracker) -> Dict[str, int]:
        state = self._state(tracker)
        if self.prev_state is not None and self.prev_action is not None:
            reward = self._reward(tracker)
            self.policy.update(self.prev_state, reward, state, action=self.prev_action)
        action_idx = self.policy.select_action(state)
        self.prev_state = state
        self.prev_action = action_idx
        return dict(self.ACTIONS[action_idx])

    def save(self) -> None:
        self.policy.save()
        self._save_state()


__all__ = ["PresetRLAgent"]
