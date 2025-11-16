from __future__ import annotations

"""Reward sanity checker for Security AI.

This module provides simple utilities to validate Menace reward values.
It is intended to detect anomalous rewards or mismatches between risk and
reward levels. The implementation is intentionally lightweight and uses
only Python's standard library.
"""

from typing import Any, Dict, List, Tuple
import json
import os
import statistics
from datetime import datetime


def load_recent_rewards(log_path: str, window_size: int = 100) -> List[float]:
    """Return the most recent *window_size* reward values from *log_path*.

    The log may be a ``.json`` file containing a list of records or a
    ``.jsonl`` file with one JSON object per line. Only values that look
    like numbers are returned.
    """

    rewards: List[float] = []
    if not os.path.exists(log_path):
        return rewards

    try:
        if log_path.endswith(".jsonl"):
            from collections import deque

            dq: deque[str] = deque(maxlen=window_size)
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        dq.append(line)
            lines = list(dq)
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("reward", "reward_score", "reward_value"):
                    if key in rec:
                        try:
                            rewards.append(float(rec[key]))
                            break
                        except Exception:
                            pass
        else:
            with open(log_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records = data[-window_size:]
            elif isinstance(data, dict):
                found: List[Any] = []
                for k in ("entries", "logs", "data"):
                    if k in data and isinstance(data[k], list):
                        found = data[k][-window_size:]
                        break
                records = found
            else:
                records = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                for key in ("reward", "reward_score", "reward_value"):
                    if key in rec:
                        try:
                            rewards.append(float(rec[key]))
                            break
                        except Exception:
                            pass
    except Exception:
        # On any unexpected failure, return what has been collected so far
        pass

    return rewards


def detect_outliers(reward_list: List[float], threshold_stddev: float = 2.5) -> List[Tuple[int, float]]:
    """Return indices and values of rewards that deviate strongly from the mean."""

    if len(reward_list) < 2:
        return []

    mean = statistics.mean(reward_list)
    stdev = statistics.stdev(reward_list)
    if stdev == 0:
        return []

    outliers: List[Tuple[int, float]] = []
    for idx, value in enumerate(reward_list):
        if abs(value - mean) > threshold_stddev * stdev:
            outliers.append((idx, value))
    return outliers


def check_risk_reward_alignment(
    actions: List[Dict[str, Any]],
    reward_threshold: float = 80.0,
    risk_threshold: float = 7.0,
) -> List[Dict[str, Any]]:
    """Flag actions with high risk and high reward simultaneously."""

    misaligned: List[Dict[str, Any]] = []
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        reward = None
        risk = None
        for key in ("reward", "reward_score", "reward_value"):
            if key in action:
                try:
                    reward = float(action[key])
                    break
                except Exception:
                    pass
        for key in ("final_risk_score", "risk_score", "risk"):
            if key in action:
                try:
                    risk = float(action[key])
                    break
                except Exception:
                    pass
        if reward is None or risk is None:
            continue
        if reward > reward_threshold and risk > risk_threshold:
            misaligned.append(
                {
                    "index": idx,
                    "action_id": action.get("id") or action.get("action"),
                    "reward": reward,
                    "risk": risk,
                }
            )
    return misaligned


def generate_sanity_report(
    outliers: List[Tuple[int, float]],
    misalignments: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Write a JSON report containing detected issues."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.utcnow().isoformat()

    report = {
        "timestamp": now,
        "outliers": [
            {"index": idx, "reward": val, "suggested_action": "manual_review"}
            for idx, val in outliers
        ],
        "risk_reward_misalignments": [],
    }

    for rec in misalignments:
        action = "penalize" if rec.get("risk", 0) > (1.5 * 7.0) else "manual_review"
        report["risk_reward_misalignments"].append({**rec, "suggested_action": action})

    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


def is_reward_sane(
    reward: float,
    context: Dict[str, Any] | None = None,
    reward_threshold: float = 80.0,
    risk_threshold: float = 7.0,
) -> bool:
    """Return ``True`` if *reward* is considered sane in *context*."""

    if reward > reward_threshold and context:
        risk = None
        for key in ("final_risk_score", "risk_score", "risk"):
            if key in context:
                try:
                    risk = float(context[key])
                    break
                except Exception:
                    pass
        if risk is not None and risk > risk_threshold:
            return False
    return True


__all__ = [
    "load_recent_rewards",
    "detect_outliers",
    "check_risk_reward_alignment",
    "generate_sanity_report",
    "is_reward_sane",
]
