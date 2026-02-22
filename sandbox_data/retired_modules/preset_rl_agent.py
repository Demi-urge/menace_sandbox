from __future__ import annotations

"""Reinforcement learning agent for adjusting sandbox presets.

The agent persists its last action/state to ``<path>.state.json`` and keeps
rotating ``.bak`` backups.  If the primary JSON file becomes corrupted, the
loader will automatically attempt to restore from the most recent backup before
falling back to a clean slate.  Administrators can manually recover by copying
``<path>.state.json.bak`` over the main file and restarting the agent.
"""

import json
import os
import logging
from typing import Dict, Tuple

from .self_improvement_policy import SelfImprovementPolicy
from .roi_tracker import ROITracker

try:  # pragma: no cover - optional dependency
    from .logging_utils import log_record
except Exception:  # pragma: no cover - fallback
    try:
        from logging_utils import log_record  # type: ignore
    except Exception:  # pragma: no cover - last resort
        def log_record(**fields: object) -> dict[str, object]:  # type: ignore
            return fields

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
        data: dict[str, object] | None = None
        try:
            with open(self.state_file) as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "state" not in data or "action" not in data:
                raise ValueError("missing keys")
        except Exception as exc:
            logger.warning("Failed to load RL state: %s", exc)
            bak = f"{self.state_file}.bak"
            if os.path.exists(bak):
                try:
                    with open(bak) as fh:
                        data = json.load(fh)
                    if not isinstance(data, dict) or "state" not in data or "action" not in data:
                        raise ValueError("missing keys")
                    os.replace(bak, self.state_file)
                except Exception as exc2:
                    logger.warning("Failed to load backup RL state: %s", exc2)
                    try:
                        os.remove(self.state_file)
                    except OSError:
                        pass
                    return
            else:
                try:
                    os.remove(self.state_file)
                except OSError:
                    pass
                return
        st = data.get("state") if data is not None else None
        self.prev_state = tuple(st) if isinstance(st, (list, tuple)) else None
        self.prev_action = data.get("action") if data is not None else None

    def _save_state(self) -> None:
        tmp_file = f"{self.state_file}.tmp"
        bak_file = f"{self.state_file}.bak"
        old_bak = f"{bak_file}.1"
        try:
            with open(tmp_file, "w") as fh:
                json.dump({"state": self.prev_state, "action": self.prev_action}, fh)
            if os.path.exists(bak_file):
                os.replace(bak_file, old_bak)
            if os.path.exists(self.state_file):
                os.replace(self.state_file, bak_file)
            os.replace(tmp_file, self.state_file)
        except Exception as exc:
            logger.warning("Failed to save RL state: %s", exc)
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _state(self, tracker: ROITracker) -> Tuple[int, int]:
        base_roi = float(tracker.roi_history[-1]) if tracker.roi_history else 0.0
        raroi = float(tracker.raroi_history[-1]) if tracker.raroi_history else base_roi
        syn_vals = tracker.metrics_history.get("synergy_roi", [])
        syn = float(syn_vals[-1]) if syn_vals else 0.0
        roi_s = 1 if raroi > 0 else (-1 if raroi < 0 else 0)
        syn_s = 1 if syn > 0 else (-1 if syn < 0 else 0)
        logger.debug(
            "preset state", extra=log_record(base_roi=base_roi, raroi=raroi)
        )
        return roi_s, syn_s

    def _reward(self, tracker: ROITracker) -> float:
        base_hist = tracker.roi_history
        raroi_hist = tracker.raroi_history
        if len(raroi_hist) < 2:
            return 0.0
        base_delta = (
            float(base_hist[-1]) - float(base_hist[-2])
            if len(base_hist) >= 2
            else 0.0
        )
        raroi_delta = float(raroi_hist[-1]) - float(raroi_hist[-2])
        logger.debug(
            "preset reward", extra=log_record(base_roi_delta=base_delta, raroi_delta=raroi_delta)
        )
        return raroi_delta

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
