from __future__ import annotations

"""Generate sandbox environment presets for scenario testing."""

import random
import os
from typing import Any, Dict, List, TYPE_CHECKING
import json
import logging

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .roi_tracker import ROITracker


# failure modes injected by the sandbox to mimic unstable environments
_FAILURE_MODES = [
    None,
    "disk",
    "network",
    "cpu",
    "memory",
    "timeout",
    "disk_corruption",
    "network_partition",
    "cpu_spike",
]

# chance that multiple failure modes will be combined in one preset
_MULTI_FAILURE_CHANCE = 0.2

_DISK_LIMITS = ["512Mi", "1Gi", "2Gi", "4Gi", "8Gi"]
_LATENCIES = [10, 50, 100, 200, 500]  # milliseconds
_BANDWIDTHS = ["1Mbps", "5Mbps", "10Mbps", "50Mbps", "100Mbps"]
_CPU_LIMITS = ["0.5", "1", "2", "4", "8"]
_MEMORY_LIMITS = [f"{m}Mi" for m in [128, 256, 512, 1024, 2048, 4096]]
_PACKET_LOSS = [0.0, 0.01, 0.05, 0.1]
_JITTERS = [0, 5, 10, 20, 50]  # milliseconds
_PACKET_DUPLICATION = [0.0, 0.01, 0.05]
_SECURITY_LEVELS = [1, 2, 3, 4, 5]
_THREAT_INTENSITIES = [10, 30, 50, 70, 90]

logger = logging.getLogger(__name__)

# minimum ROI history length for the adaptive RL agent
_ADAPTIVE_THRESHOLD = 5

# optional hardware and platform settings
_GPU_LIMITS = ["0", "1", "2", "4"]
_OS_TYPES = ["linux", "windows", "macos"]
# OS flavours that require a VM when Docker is unavailable
_ALT_OS_TYPES = ["windows", "macos"]
# probability that a preset targets a non-Linux OS
_ALT_OS_PROB = 0.3


def _select_failures() -> list[str]:
    """Return a list of failure modes to apply."""

    if random.random() < _MULTI_FAILURE_CHANCE:
        count = random.randint(2, min(3, len(_FAILURE_MODES) - 1))
        return random.sample([m for m in _FAILURE_MODES if m], count)
    mode = random.choice(_FAILURE_MODES)
    return [mode] if mode else []


def generate_presets(
    count: int | None = None,
    *,
    agent: "AdaptivePresetAgent" | None = None,
    tracker: "ROITracker" | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of random environment presets.

    Each preset sets ``CPU_LIMIT`` and ``MEMORY_LIMIT`` and may include one or
    more ``FAILURE_MODES`` to stress specific subsystems. ``count`` defaults to
    ``3``
    and can be overridden to control the number of generated scenarios.
    """
    num = 3 if count is None else max(0, count)
    presets: List[Dict[str, Any]] = []
    for _ in range(num):
        cpu = random.choice(["0.5", "1", "2", "4", "8"])
        memory = f"{random.choice([128, 256, 512, 1024, 2048, 4096])}Mi"
        disk = random.choice(_DISK_LIMITS)
        jitter = random.choice(_JITTERS)
        bw_idx1 = random.randrange(len(_BANDWIDTHS))
        bw_idx2 = random.randrange(bw_idx1, len(_BANDWIDTHS))
        preset = {
            "CPU_LIMIT": cpu,
            "MEMORY_LIMIT": memory,
            "DISK_LIMIT": disk,
            "NETWORK_LATENCY_MS": random.choice(_LATENCIES),
            "NETWORK_JITTER_MS": jitter,
            "MIN_BANDWIDTH": _BANDWIDTHS[bw_idx1],
            "MAX_BANDWIDTH": _BANDWIDTHS[bw_idx2],
            "BANDWIDTH_LIMIT": random.choice(_BANDWIDTHS),
            "PACKET_LOSS": random.choice(_PACKET_LOSS),
            "PACKET_DUPLICATION": random.choice(_PACKET_DUPLICATION),
            "SECURITY_LEVEL": random.choice(_SECURITY_LEVELS),
            "THREAT_INTENSITY": random.choice(_THREAT_INTENSITIES),
        }
        if random.random() < 0.5:
            preset["GPU_LIMIT"] = random.choice(_GPU_LIMITS)
        if random.random() < _ALT_OS_PROB:
            os_type = random.choice(_ALT_OS_TYPES)
            preset["OS_TYPE"] = os_type
            preset["CONTAINER_IMAGE"] = f"python:3.11-{os_type}"
            preset["VM_SETTINGS"] = {
                f"{os_type}_image": f"{os_type}-base.qcow2",
                "memory": "4Gi",
            }
        failures = _select_failures()
        if failures:
            preset["FAILURE_MODES"] = failures[0] if len(failures) == 1 else failures
        presets.append(preset)

    if (
        agent
        and tracker
        and len(getattr(tracker, "roi_history", [])) >= _ADAPTIVE_THRESHOLD
    ):
        try:
            actions = agent.decide(tracker)

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            def _lvl(val: int, up: bool) -> int:
                levels = _THREAT_INTENSITIES
                if up:
                    for lvl in levels:
                        if lvl > val:
                            return lvl
                    return levels[-1]
                for lvl in reversed(levels):
                    if lvl < val:
                        return lvl
                return levels[0]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("bandwidth"):
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                    p["BANDWIDTH_LIMIT"] = _nxt(
                        _BANDWIDTHS, bw, actions["bandwidth"] > 0
                    )
                    p["MAX_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MAX_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                    p["MIN_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MIN_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            agent.save()
        except Exception:
            logger.exception("preset adaptation failed")

    return presets


__all__ = ["generate_presets", "AdaptivePresetAgent"]


class AdaptivePresetAgent:
    """Simple RL agent using ROI and synergy history to adjust presets.

    When ``strategy`` is set to ``"deep_q"`` a small neural network is used to
    estimate Q-values via :class:`DeepQLearningStrategy`. ``"double_dqn"``
    selects the :class:`DoubleDQNStrategy` if PyTorch is available.
    """

    ACTIONS: tuple[dict[str, int], ...] = (
        {"cpu": 1, "memory": 0, "threat": 0},
        {"cpu": -1, "memory": 0, "threat": 0},
        {"cpu": 0, "memory": 1, "threat": 0},
        {"cpu": 0, "memory": -1, "threat": 0},
        {"cpu": 0, "memory": 0, "threat": 1},
        {"cpu": 0, "memory": 0, "threat": -1},
        {"cpu": 0, "memory": 0, "threat": 0},
    )

    def __init__(self, path: str | None = None, *, strategy: str | None = None) -> None:
        from .self_improvement_policy import (
            SelfImprovementPolicy,
            strategy_factory,
            DoubleDQNStrategy,
            torch as _torch,
            nn as _nn,
        )

        if strategy is None:
            strategy = (
                os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY")
                or os.getenv("SANDBOX_PRESET_RL_STRATEGY")
                or "q_learning"
            )

        name = str(strategy).replace("-", "_")
        if name == "double_dqn" and _torch is not None and _nn is not None:
            rl_strategy = DoubleDQNStrategy()
        else:
            env_var = (
                "SANDBOX_ADAPTIVE_AGENT_STRATEGY"
                if os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY") is not None
                else "SANDBOX_PRESET_RL_STRATEGY"
            )
            rl_strategy = strategy_factory(strategy, env_var=env_var)

        self.policy = SelfImprovementPolicy(path=path, strategy=rl_strategy)
        self.state_file = f"{path}.state.json" if path else None
        self.prev_state: tuple[int, ...] | None = None
        self.prev_action: int | None = None
        self._load_state()

    # --------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load RL state: %s", exc)
            bak = f"{self.state_file}.bak"
            if os.path.exists(bak):
                try:
                    with open(bak) as fh:
                        data = json.load(fh)
                except Exception as exc2:
                    logger.warning("Failed to load backup RL state: %s", exc2)
                    return
            else:
                return
        st = data.get("state")
        self.prev_state = tuple(st) if st is not None else None
        self.prev_action = data.get("action")

    def _save_state(self) -> None:
        if not self.state_file:
            return
        tmp_file = f"{self.state_file}.tmp"
        bak_file = f"{self.state_file}.bak"
        try:
            with open(tmp_file, "w") as fh:
                json.dump({"state": self.prev_state, "action": self.prev_action}, fh)
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

    # --------------------------------------------------------------
    def _state(self, tracker: "ROITracker") -> tuple[int, ...]:
        """Return a compact representation of the current environment state.

        The state encodes ROI and synergy trends along with recent CPU usage,
        memory usage and threat intensity information obtained from
        :class:`ROITracker`.
        """

        roi_hist = tracker.roi_history
        syn_hist = tracker.metrics_history.get("synergy_roi", [])
        eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
        res_hist = tracker.metrics_history.get("synergy_resilience", [])
        cpu_hist = tracker.metrics_history.get("cpu_usage", [])
        mem_hist = tracker.metrics_history.get("memory_usage", [])
        threat_hist = tracker.metrics_history.get("threat_intensity", [])

        def _trend(vals: list[float]) -> float:
            if len(vals) >= 2:
                return float(vals[-1] - vals[-2])
            return float(vals[-1]) if vals else 0.0

        roi_trend = _trend(roi_hist)
        syn_trend = _trend(syn_hist)
        eff_trend = _trend(eff_hist)
        res_trend = _trend(res_hist)
        cpu_trend = _trend(cpu_hist)
        mem_trend = _trend(mem_hist)
        threat_trend = _trend(threat_hist)

        def _sign(val: float) -> int:
            return 1 if val > 0 else (-1 if val < 0 else 0)

        return (
            _sign(roi_trend),
            _sign(syn_trend),
            _sign(eff_trend),
            _sign(res_trend),
            _sign(cpu_trend),
            _sign(mem_trend),
            _sign(threat_trend),
        )

    def _reward(self, tracker: "ROITracker") -> float:
        """Return reward based on ROI, synergy and resource changes."""

        roi_hist = tracker.roi_history
        roi_delta = roi_hist[-1] - roi_hist[-2] if len(roi_hist) >= 2 else 0.0

        syn_hist = tracker.metrics_history.get("synergy_roi", [])
        eff_hist = tracker.metrics_history.get("synergy_efficiency", [])
        res_hist = tracker.metrics_history.get("synergy_resilience", [])
        syn_delta = 0.0
        if len(syn_hist) >= 2:
            syn_delta = syn_hist[-1] - syn_hist[-2]
        elif syn_hist:
            syn_delta = syn_hist[-1]
        eff_delta = (
            eff_hist[-1] - eff_hist[-2]
            if len(eff_hist) >= 2
            else (eff_hist[-1] if eff_hist else 0.0)
        )
        res_delta = (
            res_hist[-1] - res_hist[-2]
            if len(res_hist) >= 2
            else (res_hist[-1] if res_hist else 0.0)
        )

        cpu_hist = tracker.metrics_history.get("cpu_usage", [])
        mem_hist = tracker.metrics_history.get("memory_usage", [])
        threat_hist = tracker.metrics_history.get("threat_intensity", [])

        cpu_delta = cpu_hist[-1] - cpu_hist[-2] if len(cpu_hist) >= 2 else 0.0
        mem_delta = mem_hist[-1] - mem_hist[-2] if len(mem_hist) >= 2 else 0.0
        threat_delta = (
            threat_hist[-1] - threat_hist[-2] if len(threat_hist) >= 2 else 0.0
        )

        # Penalise increased resource usage and threat intensity
        return float(
            roi_delta
            + syn_delta
            + eff_delta
            + res_delta
            - cpu_delta
            - mem_delta
            - threat_delta
        )

    # --------------------------------------------------------------
    def decide(self, tracker: "ROITracker") -> Dict[str, int]:
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

    # --------------------------------------------------------------
    def export_policy(self) -> Dict[tuple[int, ...], Dict[int, float]]:
        """Return policy values for external analysis."""

        return {k: dict(v) for k, v in self.policy.values.items()}

    def import_policy(self, data: Dict[tuple[int, ...], Dict[int, float]]) -> None:
        """Load policy values from ``data``."""

        new_table: Dict[tuple[int, ...], Dict[int, float]] = {}
        for key, val in data.items():
            tup = tuple(key) if not isinstance(key, tuple) else key
            new_table[tup] = {int(a): float(q) for a, q in val.items()}
        self.policy.values = new_table


def adapt_presets(
    tracker: "ROITracker", presets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return ``presets`` adjusted based on ``tracker`` metrics.

    When the recent average ``security_score`` is high, the next higher
    ``THREAT_INTENSITY`` level is selected to further stress security
    mechanisms. Low scores decrease the intensity accordingly.
    """

    agent = None
    rl_path = os.getenv("SANDBOX_PRESET_RL_PATH")
    if not rl_path:
        rl_path = os.path.join("sandbox_data", "preset_policy.json")
    rl_strategy = os.getenv("SANDBOX_PRESET_RL_STRATEGY")
    if rl_path:
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)
        try:
            agent = getattr(adapt_presets, "_rl_agent", None)
            strat_name = (
                getattr(
                    getattr(agent, "policy", None), "strategy", None
                ).__class__.__name__.lower()
                if agent
                else None
            )
            if (
                agent is None
                or getattr(getattr(agent, "policy", None), "path", None) != rl_path
                or (rl_strategy and strat_name != rl_strategy.replace("-", "_").lower())
            ):
                agent = AdaptivePresetAgent(rl_path, strategy=rl_strategy)
                adapt_presets._rl_agent = agent
        except Exception:
            agent = None

    adapt_agent = getattr(adapt_presets, "_adaptive_agent", None)
    if adapt_agent is None:
        try:
            path = os.getenv("SANDBOX_ADAPTIVE_AGENT_PATH")
            adapt_strategy = os.getenv("SANDBOX_ADAPTIVE_AGENT_STRATEGY")
            adapt_agent = AdaptivePresetAgent(path, strategy=adapt_strategy)
            adapt_presets._adaptive_agent = adapt_agent
        except Exception:
            adapt_agent = None

    sec_vals = tracker.metrics_history.get("security_score", [])
    if not sec_vals:
        return presets

    recent = sec_vals[-3:]
    avg_sec = sum(recent) / len(recent)

    def _next_level(current: int, up: bool) -> int:
        levels = _THREAT_INTENSITIES
        if up:
            for lvl in levels:
                if lvl > current:
                    return lvl
            return levels[-1]
        for lvl in reversed(levels):
            if lvl < current:
                return lvl
        return levels[0]

    if avg_sec >= 80.0:
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            p["THREAT_INTENSITY"] = _next_level(cur, True)
    elif avg_sec < 50.0:
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            p["THREAT_INTENSITY"] = _next_level(cur, False)

    # --------------------------------------------------------------
    # ROI-driven resource scaling
    def _next_val(seq: List[Any], current: Any, up: bool) -> Any:
        lookup = [str(x) for x in seq]
        try:
            idx = lookup.index(str(current))
        except ValueError:
            idx = 0
        idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
        return seq[idx]

    roi_hist = getattr(tracker, "roi_history", [])
    if len(roi_hist) >= 2:
        deltas = [roi_hist[i] - roi_hist[i - 1] for i in range(1, len(roi_hist))]
        recent_deltas = deltas[-3:]
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        tol = getattr(tracker, "diminishing", lambda: 0.01)()

        if abs(avg_delta) <= tol:
            # ROI stagnates -> scale up resources
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
        elif avg_delta > tol:
            # ROI improving -> slowly tighten limits
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )

    # --------------------------------------------------------------
    # Use reinforcement learning when sufficient history is available
    if (
        agent
        and len(tracker.roi_history) >= 3
        and tracker.metrics_history.get("synergy_roi")
    ):
        try:
            actions = agent.decide(tracker)

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("bandwidth"):
                    bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                    p["BANDWIDTH_LIMIT"] = _nxt(
                        _BANDWIDTHS, bw, actions["bandwidth"] > 0
                    )
                    p["MAX_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MAX_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                    p["MIN_BANDWIDTH"] = _nxt(
                        _BANDWIDTHS,
                        str(p.get("MIN_BANDWIDTH", bw)),
                        actions["bandwidth"] > 0,
                    )
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))

                    def _lvl(val: int, up: bool) -> int:
                        levels = _THREAT_INTENSITIES
                        if up:
                            for lvl in levels:
                                if lvl > val:
                                    return lvl
                            return levels[-1]
                        for lvl in reversed(levels):
                            if lvl < val:
                                return lvl
                        return levels[0]

                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            agent.save()
            return presets
        except Exception:
            logger.exception("preset adaptation failed")

    elif adapt_agent and len(tracker.roi_history) >= _ADAPTIVE_THRESHOLD:
        try:
            actions = adapt_agent.decide(tracker)

            def _nxt(seq: List[Any], cur: Any, up: bool) -> Any:
                lookup = [str(x) for x in seq]
                try:
                    idx = lookup.index(str(cur))
                except ValueError:
                    idx = 0
                idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
                return seq[idx]

            for p in presets:
                if actions.get("cpu"):
                    p["CPU_LIMIT"] = _nxt(
                        _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), actions["cpu"] > 0
                    )
                if actions.get("memory"):
                    mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                    p["MEMORY_LIMIT"] = _nxt(_MEMORY_LIMITS, mem, actions["memory"] > 0)
                if actions.get("threat"):
                    cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))

                    def _lvl(val: int, up: bool) -> int:
                        levels = _THREAT_INTENSITIES
                        if up:
                            for lvl in levels:
                                if lvl > val:
                                    return lvl
                            return levels[-1]
                        for lvl in reversed(levels):
                            if lvl < val:
                                return lvl
                        return levels[0]

                    p["THREAT_INTENSITY"] = _lvl(cur, actions["threat"] > 0)
            adapt_agent.save()
            return presets
        except Exception:
            logger.exception("preset adaptation failed")

    # --------------------------------------------------------------
    # Synergy-driven adjustments with prediction support
    syn_roi_vals = tracker.metrics_history.get("synergy_roi", [])
    try:
        if hasattr(tracker, "forecast_synergy"):
            pred_synergy_roi = float(getattr(tracker, "forecast_synergy")()[0])
        else:
            pred_synergy_roi = float(getattr(tracker, "predict_synergy")())
    except Exception:
        pred_synergy_roi = 0.0
    vals = syn_roi_vals[-3:]
    if pred_synergy_roi:
        vals.append(pred_synergy_roi)
    if vals:
        avg_syn = sum(vals) / len(vals)
        if avg_syn > 0.1:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, True)
                lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, False)
                bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[0]))
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, True)
        elif avg_syn < -0.1:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, False)
                lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[-1])
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, True)
                bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[-1]))
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, False)

    syn_sec_vals = tracker.metrics_history.get("synergy_security_score", [])
    try:
        pred_syn_sec = float(
            getattr(tracker, "predict_synergy_metric")("security_score")
        )
    except Exception:
        pred_syn_sec = 0.0
    vals_sec = syn_sec_vals[-3:]
    if pred_syn_sec:
        vals_sec.append(pred_syn_sec)
    if vals_sec:
        avg_syn_sec = sum(vals_sec) / len(vals_sec)

        def _next_sec_level(current: int, up: bool) -> int:
            levels = _SECURITY_LEVELS
            if up:
                for lvl in levels:
                    if lvl > current:
                        return lvl
                return levels[-1]
            for lvl in reversed(levels):
                if lvl < current:
                    return lvl
            return levels[0]

        if avg_syn_sec > 5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, True)
        elif avg_syn_sec < -5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, False)

    syn_safe_vals = tracker.metrics_history.get("synergy_safety_rating", [])
    try:
        pred_syn_safe = float(
            getattr(tracker, "predict_synergy_metric")("safety_rating")
        )
    except Exception:
        pred_syn_safe = 0.0
    vals_safe = syn_safe_vals[-3:]
    if pred_syn_safe:
        vals_safe.append(pred_syn_safe)
    if vals_safe:
        avg_syn_safe = sum(vals_safe) / len(vals_safe)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_safe > 5.0:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_safe < -5.0:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_risk_vals = tracker.metrics_history.get("synergy_risk_index", [])
    try:
        pred_syn_risk = float(getattr(tracker, "predict_synergy_metric")("risk_index"))
    except Exception:
        pred_syn_risk = 0.0
    vals_risk = syn_risk_vals[-3:]
    if pred_syn_risk:
        vals_risk.append(pred_syn_risk)
    if vals_risk:
        avg_syn_risk = sum(vals_risk) / len(vals_risk)

        def _next_sec_level(current: int, up: bool) -> int:
            levels = _SECURITY_LEVELS
            if up:
                for lvl in levels:
                    if lvl > current:
                        return lvl
                return levels[-1]
            for lvl in reversed(levels):
                if lvl < current:
                    return lvl
            return levels[0]

        if avg_syn_risk > 5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, True)
        elif avg_syn_risk < -5.0:
            for p in presets:
                cur = int(p.get("SECURITY_LEVEL", _SECURITY_LEVELS[0]))
                p["SECURITY_LEVEL"] = _next_sec_level(cur, False)

    # ------------------------------------------------------------------
    # Synergy metrics for efficiency, antifragility and resilience
    syn_eff_vals = tracker.metrics_history.get("synergy_efficiency", [])
    try:
        pred_syn_eff = float(getattr(tracker, "predict_synergy_metric")("efficiency"))
    except Exception:
        pred_syn_eff = 0.0
    vals_eff = syn_eff_vals[-3:]
    if pred_syn_eff:
        vals_eff.append(pred_syn_eff)
    if vals_eff:
        avg_syn_eff = sum(vals_eff) / len(vals_eff)
        if avg_syn_eff > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_eff < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_adapt_vals = tracker.metrics_history.get("synergy_adaptability", [])
    try:
        pred_syn_adapt = float(
            getattr(tracker, "predict_synergy_metric")("adaptability")
        )
    except Exception:
        pred_syn_adapt = 0.0
    vals_adapt = syn_adapt_vals[-3:]
    if pred_syn_adapt:
        vals_adapt.append(pred_syn_adapt)
    if vals_adapt:
        avg_syn_adapt = sum(vals_adapt) / len(vals_adapt)
        if avg_syn_adapt > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_adapt < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_anti_vals = tracker.metrics_history.get("synergy_antifragility", [])
    try:
        pred_syn_af = float(getattr(tracker, "predict_synergy_metric")("antifragility"))
    except Exception:
        pred_syn_af = 0.0
    vals_af = syn_anti_vals[-3:]
    if pred_syn_af:
        vals_af.append(pred_syn_af)
    if vals_af:
        avg_syn_af = sum(vals_af) / len(vals_af)
        if avg_syn_af > 0.05:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, True)
        elif avg_syn_af < -0.05:
            for p in presets:
                cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_res_vals = tracker.metrics_history.get("synergy_resilience", [])
    try:
        pred_syn_res = float(getattr(tracker, "predict_synergy_metric")("resilience"))
    except Exception:
        pred_syn_res = 0.0
    vals_res = syn_res_vals[-3:]
    if pred_syn_res:
        vals_res.append(pred_syn_res)
    if vals_res:
        avg_syn_res = sum(vals_res) / len(vals_res)
        if avg_syn_res > 0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
        elif avg_syn_res < -0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[-1]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )

    syn_rec_vals = tracker.metrics_history.get("synergy_recovery_time", [])
    try:
        pred_syn_rec = float(getattr(tracker, "predict_synergy_recovery_time")())
    except Exception:
        pred_syn_rec = 0.0
    vals_rec = syn_rec_vals[-3:]
    if pred_syn_rec:
        vals_rec.append(pred_syn_rec)
    if vals_rec:
        avg_syn_rec = sum(vals_rec) / len(vals_rec)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_rec > 0.1:
                p["THREAT_INTENSITY"] = _next_level(cur, False)
            elif avg_syn_rec < -0.1:
                p["THREAT_INTENSITY"] = _next_level(cur, True)

    # ------------------------------------------------------------------
    # Additional synergy metrics controlling resources
    syn_ent_vals = tracker.metrics_history.get("synergy_shannon_entropy", [])
    try:
        pred_syn_ent = float(
            getattr(tracker, "predict_synergy_metric")("shannon_entropy")
        )
    except Exception:
        pred_syn_ent = 0.0
    vals_ent = syn_ent_vals[-3:]
    if pred_syn_ent:
        vals_ent.append(pred_syn_ent)
    if vals_ent:
        avg_syn_ent = sum(vals_ent) / len(vals_ent)
        if avg_syn_ent > 0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), True
                )
        elif avg_syn_ent < -0.05:
            for p in presets:
                p["CPU_LIMIT"] = _next_val(
                    _CPU_LIMITS, str(p.get("CPU_LIMIT", "1")), False
                )

    syn_flex_vals = tracker.metrics_history.get("synergy_flexibility", [])
    try:
        pred_syn_flex = float(getattr(tracker, "predict_synergy_metric")("flexibility"))
    except Exception:
        pred_syn_flex = 0.0
    vals_flex = syn_flex_vals[-3:]
    if pred_syn_flex:
        vals_flex.append(pred_syn_flex)
    if vals_flex:
        avg_syn_flex = sum(vals_flex) / len(vals_flex)
        if avg_syn_flex > 0.05:
            for p in presets:
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)
        elif avg_syn_flex < -0.05:
            for p in presets:
                mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)

    syn_energy_vals = tracker.metrics_history.get("synergy_energy_consumption", [])
    try:
        pred_syn_energy = float(
            getattr(tracker, "predict_synergy_metric")("energy_consumption")
        )
    except Exception:
        pred_syn_energy = 0.0
    vals_energy = syn_energy_vals[-3:]
    if pred_syn_energy:
        vals_energy.append(pred_syn_energy)
    if vals_energy:
        avg_syn_energy = sum(vals_energy) / len(vals_energy)
        if avg_syn_energy > 0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[-1]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, False)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), False
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )
        elif avg_syn_energy < -0.05:
            for p in presets:
                bw = str(p.get("BANDWIDTH_LIMIT", _BANDWIDTHS[0]))
                p["BANDWIDTH_LIMIT"] = _next_val(_BANDWIDTHS, bw, True)
                p["MAX_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MAX_BANDWIDTH", bw)), True
                )
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )

    syn_profit_vals = tracker.metrics_history.get("synergy_profitability", [])
    try:
        pred_syn_profit = float(
            getattr(tracker, "predict_synergy_metric")("profitability")
        )
    except Exception:
        pred_syn_profit = 0.0
    vals_profit = syn_profit_vals[-3:]
    if pred_syn_profit:
        vals_profit.append(pred_syn_profit)
    if vals_profit:
        avg_syn_profit = sum(vals_profit) / len(vals_profit)
        if avg_syn_profit > 0.05:
            for p in presets:
                cur = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, cur, True)
        elif avg_syn_profit < -0.05:
            for p in presets:
                cur = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, cur, False)

    syn_rev_vals = tracker.metrics_history.get("synergy_revenue", [])
    try:
        pred_syn_rev = float(getattr(tracker, "predict_synergy_metric")("revenue"))
    except Exception:
        pred_syn_rev = 0.0
    vals_rev = syn_rev_vals[-3:]
    if pred_syn_rev:
        vals_rev.append(pred_syn_rev)
    if vals_rev:
        avg_syn_rev = sum(vals_rev) / len(vals_rev)
        for p in presets:
            mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
            if avg_syn_rev > 0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
            elif avg_syn_rev < -0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)

    syn_lucr_vals = tracker.metrics_history.get("synergy_projected_lucrativity", [])
    try:
        pred_syn_lucr = float(
            getattr(tracker, "predict_synergy_metric")("projected_lucrativity")
        )
    except Exception:
        pred_syn_lucr = 0.0
    vals_lucr = syn_lucr_vals[-3:]
    if pred_syn_lucr:
        vals_lucr.append(pred_syn_lucr)
    if vals_lucr:
        avg_syn_lucr = sum(vals_lucr) / len(vals_lucr)
        if avg_syn_lucr > 0.05:
            for p in presets:
                cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, True)
        elif avg_syn_lucr < -0.05:
            for p in presets:
                cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, False)

    # ----------------------------------------------
    # New synergy metrics for maintainability, code quality
    syn_maint_vals = tracker.metrics_history.get("synergy_maintainability", [])
    try:
        pred_syn_maint = float(
            getattr(tracker, "predict_synergy_metric")("maintainability")
        )
    except Exception:
        pred_syn_maint = 0.0
    vals_maint = syn_maint_vals[-3:]
    if pred_syn_maint:
        vals_maint.append(pred_syn_maint)
    if vals_maint:
        avg_syn_maint = sum(vals_maint) / len(vals_maint)
        for p in presets:
            cpu = str(p.get("CPU_LIMIT", "1"))
            if avg_syn_maint > 0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cpu, False)
            elif avg_syn_maint < -0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cpu, True)

    syn_cq_vals = tracker.metrics_history.get("synergy_code_quality", [])
    try:
        pred_syn_cq = float(getattr(tracker, "predict_synergy_metric")("code_quality"))
    except Exception:
        pred_syn_cq = 0.0
    vals_cq = syn_cq_vals[-3:]
    if pred_syn_cq:
        vals_cq.append(pred_syn_cq)
    if vals_cq:
        avg_syn_cq = sum(vals_cq) / len(vals_cq)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_cq > 0.05:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_cq < -0.05:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_lat_vals = tracker.metrics_history.get("synergy_network_latency", [])
    try:
        pred_syn_lat = float(
            getattr(tracker, "predict_synergy_metric")("network_latency")
        )
    except Exception:
        pred_syn_lat = 0.0
    vals_lat = syn_lat_vals[-3:]
    if pred_syn_lat:
        vals_lat.append(pred_syn_lat)
    if vals_lat:
        avg_syn_lat = sum(vals_lat) / len(vals_lat)
        for p in presets:
            lat = p.get("NETWORK_LATENCY_MS", _LATENCIES[0])
            if avg_syn_lat > 1.0:
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, True)
            elif avg_syn_lat < -1.0:
                p["NETWORK_LATENCY_MS"] = _next_val(_LATENCIES, lat, False)

    syn_tp_vals = tracker.metrics_history.get("synergy_throughput", [])
    try:
        pred_syn_tp = float(getattr(tracker, "predict_synergy_metric")("throughput"))
    except Exception:
        pred_syn_tp = 0.0
    vals_tp = syn_tp_vals[-3:]
    if pred_syn_tp:
        vals_tp.append(pred_syn_tp)
    if vals_tp:
        avg_syn_tp = sum(vals_tp) / len(vals_tp)
        for p in presets:
            bw = str(p.get("MAX_BANDWIDTH", _BANDWIDTHS[0]))
            if avg_syn_tp > 5.0:
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, True)
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), True
                )
            elif avg_syn_tp < -5.0:
                p["MAX_BANDWIDTH"] = _next_val(_BANDWIDTHS, bw, False)
                p["MIN_BANDWIDTH"] = _next_val(
                    _BANDWIDTHS, str(p.get("MIN_BANDWIDTH", bw)), False
                )

    # ------------------------------------------------------------------
    # Additional synergy metrics reacting to usage and discrepancies
    syn_disc_vals = tracker.metrics_history.get("synergy_discrepancy_count", [])
    try:
        pred_syn_disc = float(
            getattr(tracker, "predict_synergy_metric")("discrepancy_count")
        )
    except Exception:
        pred_syn_disc = 0.0
    vals_disc = syn_disc_vals[-3:]
    if pred_syn_disc:
        vals_disc.append(pred_syn_disc)
    if vals_disc:
        avg_syn_disc = sum(vals_disc) / len(vals_disc)
        for p in presets:
            cur = int(p.get("THREAT_INTENSITY", _THREAT_INTENSITIES[0]))
            if avg_syn_disc > 1.0:
                p["THREAT_INTENSITY"] = _next_level(cur, True)
            elif avg_syn_disc < -1.0:
                p["THREAT_INTENSITY"] = _next_level(cur, False)

    syn_gpu_vals = tracker.metrics_history.get("synergy_gpu_usage", [])
    try:
        pred_syn_gpu = float(getattr(tracker, "predict_synergy_metric")("gpu_usage"))
    except Exception:
        pred_syn_gpu = 0.0
    vals_gpu = syn_gpu_vals[-3:]
    if pred_syn_gpu:
        vals_gpu.append(pred_syn_gpu)
    if vals_gpu:
        avg_syn_gpu = sum(vals_gpu) / len(vals_gpu)
        for p in presets:
            cur = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
            if avg_syn_gpu > 0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, True)
            elif avg_syn_gpu < -0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, cur, False)

    syn_cpu_vals = tracker.metrics_history.get("synergy_cpu_usage", [])
    try:
        pred_syn_cpu = float(getattr(tracker, "predict_synergy_metric")("cpu_usage"))
    except Exception:
        pred_syn_cpu = 0.0
    vals_cpu = syn_cpu_vals[-3:]
    if pred_syn_cpu:
        vals_cpu.append(pred_syn_cpu)
    if vals_cpu:
        avg_syn_cpu = sum(vals_cpu) / len(vals_cpu)
        for p in presets:
            cur = str(p.get("CPU_LIMIT", "1"))
            if avg_syn_cpu > 0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cur, True)
            elif avg_syn_cpu < -0.05:
                p["CPU_LIMIT"] = _next_val(_CPU_LIMITS, cur, False)

    syn_mem_vals = tracker.metrics_history.get("synergy_memory_usage", [])
    try:
        pred_syn_mem = float(getattr(tracker, "predict_synergy_metric")("memory_usage"))
    except Exception:
        pred_syn_mem = 0.0
    vals_mem = syn_mem_vals[-3:]
    if pred_syn_mem:
        vals_mem.append(pred_syn_mem)
    if vals_mem:
        avg_syn_mem = sum(vals_mem) / len(vals_mem)
        for p in presets:
            mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
            if avg_syn_mem > 0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, True)
            elif avg_syn_mem < -0.05:
                p["MEMORY_LIMIT"] = _next_val(_MEMORY_LIMITS, mem, False)

    syn_long_vals = tracker.metrics_history.get("synergy_long_term_lucrativity", [])
    try:
        pred_syn_long = float(
            getattr(tracker, "predict_synergy_metric")("long_term_lucrativity")
        )
    except Exception:
        pred_syn_long = 0.0
    vals_long = syn_long_vals[-3:]
    if pred_syn_long:
        vals_long.append(pred_syn_long)
    if vals_long:
        avg_syn_long = sum(vals_long) / len(vals_long)
        for p in presets:
            gpu = str(p.get("GPU_LIMIT", _GPU_LIMITS[0]))
            disk = str(p.get("DISK_LIMIT", _DISK_LIMITS[0]))
            if avg_syn_long > 0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, gpu, True)
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, disk, True)
            elif avg_syn_long < -0.05:
                p["GPU_LIMIT"] = _next_val(_GPU_LIMITS, gpu, False)
                p["DISK_LIMIT"] = _next_val(_DISK_LIMITS, disk, False)

    return presets


def export_preset_policy() -> Dict[tuple[int, ...], Dict[int, float]]:
    """Return policy data from the active RL agent."""

    agent = getattr(adapt_presets, "_rl_agent", None)
    return agent.export_policy() if agent else {}


def import_preset_policy(data: Dict[tuple[int, ...], Dict[int, float]]) -> None:
    """Load policy ``data`` into the active RL agent if available."""

    agent = getattr(adapt_presets, "_rl_agent", None)
    if agent:
        agent.import_policy(data)


def generate_presets_from_history(
    data_dir: str = "sandbox_data", count: int | None = None
) -> List[Dict[str, Any]]:
    """Return presets adapted using ROI/security history from ``data_dir``."""

    history = os.path.join(data_dir, "roi_history.json")
    tracker = None
    if os.path.exists(history):
        try:
            from .roi_tracker import ROITracker

            tracker = ROITracker()
            tracker.load_history(history)
        except Exception:
            tracker = None
    presets = generate_presets(count)
    if tracker and tracker.metrics_history.get("security_score"):
        try:
            presets = adapt_presets(tracker, presets)
        except Exception:
            logger.exception("preset evolution failed")
    return presets


__all__.append("adapt_presets")
__all__.extend([
    "export_preset_policy",
    "import_preset_policy",
    "generate_presets_from_history",
])
