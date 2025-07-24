from __future__ import annotations

"""Reinforcement-learning engine suggesting next actions."""

from typing import Callable, Optional, Tuple
import logging
import os
import json
from collections import deque
from datetime import datetime

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _NP:  # type: ignore
        inf = float("inf")
        float32 = float

        @staticmethod
        def asarray(x, dtype=None):
            return x

    np = _NP()  # type: ignore

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import DQN, PPO, A2C, SAC, TD3  # type: ignore
    import gym  # type: ignore
    from gym import spaces  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DQN = PPO = A2C = SAC = TD3 = None  # type: ignore
    gym = None  # type: ignore
    spaces = None  # type: ignore

from .neuroplasticity import PathwayDB, PathwayRecord
from .resource_allocation_optimizer import ROIDB
from .code_database import CodeDB
from .action_planner import _RLModel
from .unified_learning_engine import UnifiedLearningEngine


class _SB3Model:
    """Lightweight wrapper around stable-baselines3 models."""

    def __init__(
        self,
        algo: str = "dqn",
        *,
        train_steps: int | None = None,
        env_params: dict | None = None,
        **algo_kwargs,
    ) -> None:
        self.algo = algo
        self.algo_kwargs = algo_kwargs
        self.train_steps = train_steps
        self.env_params = env_params or {}
        self.transitions: list[tuple[tuple[float, ...], int, float]] = []
        self.state_map: dict[tuple[str, ...], int] = {}
        self.action_map: dict[str, int] = {}
        self.rev_action: dict[int, str] = {}
        self.model = None

    def _state_idx(self, state: tuple[str, ...]) -> int:
        if state not in self.state_map:
            self.state_map[state] = len(self.state_map)
        return self.state_map[state]

    def _action_idx(self, action: str) -> int:
        if action not in self.action_map:
            idx = len(self.action_map)
            self.action_map[action] = idx
            self.rev_action[idx] = action
        return self.action_map[action]

    # ------------------------------------------------------------------
    def update(
        self,
        state: tuple[str, ...],
        action: str,
        reward: float,
        obs: tuple[float, ...],
    ) -> None:
        """Store a transition with observation *obs* and *reward*."""
        self._state_idx(state)
        a = self._action_idx(action)
        self.transitions.append((obs, a, reward))

    def _build_env(self):
        assert gym is not None and spaces is not None  # pragma: no cover - sanity

        n_actions = max(self.action_map.values(), default=0) + 1
        obs_dim = len(self.transitions[0][0]) if self.transitions else 1

        class TransitionEnv(gym.Env):  # type: ignore
            def __init__(self, transitions: list[tuple[tuple[float, ...], int, float]]):
                self.transitions = transitions
                self.action_space = spaces.Discrete(n_actions)
                box_cls = getattr(spaces, "Box", None)
                if box_cls is None:
                    class box_cls:  # type: ignore
                        def __init__(self, *a, **k):
                            self.shape = k.get("shape")

                self.observation_space = box_cls(
                    -np.inf,
                    np.inf,
                    shape=(obs_dim,),
                    dtype=getattr(np, "float32", float),
                )
                self.idx = 0

            def reset(self):  # type: ignore[override]
                self.idx = (self.idx + 1) % len(self.transitions)
                obs, _, _ = self.transitions[self.idx]
                arr_fn = getattr(np, "asarray", lambda x, dtype=None: x)
                return arr_fn(obs, dtype=getattr(np, "float32", float))

            def step(self, action):  # type: ignore[override]
                obs, correct, reward = self.transitions[self.idx]
                r = reward if action == correct else -abs(reward)
                done = True
                info = {}
                self.idx = (self.idx + 1) % len(self.transitions)
                arr_fn = getattr(np, "asarray", lambda x, dtype=None: x)
                next_obs = arr_fn(self.transitions[self.idx][0], dtype=getattr(np, "float32", float))
                return next_obs, r, done, info

        return TransitionEnv(self.transitions)

    def train(self) -> None:
        if gym is None or all(m is None for m in (DQN, PPO, A2C, SAC, TD3)):
            return
        env = self._build_env()
        algo = self.algo.lower()
        if algo == "ppo" and PPO is not None:
            cls = PPO
        elif algo == "a2c" and A2C is not None:
            cls = A2C
        elif algo == "sac" and SAC is not None:
            cls = SAC
        elif algo == "td3" and TD3 is not None:
            cls = TD3
        else:
            cls = DQN
        self.model = cls("MlpPolicy", env, verbose=0, **self.algo_kwargs)
        steps = self.train_steps if self.train_steps is not None else max(len(self.transitions), 1)
        self.model.learn(total_timesteps=steps)

    def best_action(self, obs: tuple[float, ...]) -> Optional[str]:
        if self.model is None:
            return None
        arr_fn = getattr(np, "asarray", lambda x, dtype=None: x)
        obs_arr = arr_fn(obs, dtype=getattr(np, "float32", float))
        act, _ = self.model.predict(obs_arr)
        return self.rev_action.get(int(act))

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> None:
        if gym is None:
            return
        algo = self.algo.lower()
        if algo == "ppo" and PPO is not None:
            cls = PPO
        elif algo == "a2c" and A2C is not None:
            cls = A2C
        elif algo == "sac" and SAC is not None:
            cls = SAC
        elif algo == "td3" and TD3 is not None:
            cls = TD3
        else:
            cls = DQN
        self.model = cls.load(path)


class ActionLearningEngine:
    """Learn action transitions considering ROI and code complexity."""

    def __init__(
        self,
        pathway_db: PathwayDB,
        roi_db: ROIDB,
        code_db: CodeDB,
        learning_engine: UnifiedLearningEngine | None = None,
        *,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        success_weight: float = 1.0,
        reward_weights: dict | None = None,
        reward_weights_path: str | None = None,
        observation_features_fn: Callable[[str, PathwayRecord | None], tuple[float, ...]] | None = None,
        action_filter_fn: Callable[[str], bool] | None = None,
        transition_log_size: int = 1000,
        env_params: dict | None = None,
        algo: str | None = None,
        algo_kwargs: dict | None = None,
        train_steps: int | None = None,
    ) -> None:
        self.pathway_db = pathway_db
        self.roi_db = roi_db
        self.code_db = code_db
        self.learning_engine = learning_engine
        self.logger = logging.getLogger("ActionLearningEngine")
        self.success_weight = success_weight
        self.reward_weights = {
            "roi": 1.0,
            "future_roi": 1.0,
            "trend": 0.5,
            "prediction": self.success_weight,
            "accuracy": 1.0,
            "complexity": 1.0,
        }
        # reward weights may come from env, file, or arg
        env_cfg = os.environ.get("ACTION_LEARNING_REWARD_WEIGHTS")
        if env_cfg:
            try:
                self.reward_weights.update(json.loads(env_cfg))
            except Exception as exc:
                self.logger.exception("invalid reward weights from env: %s", exc)
        if reward_weights_path:
            try:
                with open(reward_weights_path) as fh:
                    self.reward_weights.update(json.load(fh))
            except Exception as exc:
                self.logger.exception("failed to load reward weights file: %s", exc)
        if reward_weights:
            self.reward_weights.update(reward_weights)
        self.env_params = env_params or {}
        self.observation_features_fn = observation_features_fn
        self.action_filter_fn = action_filter_fn
        self.transition_log: deque[tuple[tuple[str, ...], str, float, tuple[float, ...]]] = deque(maxlen=transition_log_size)
        if algo and gym is not None and any(m is not None for m in (DQN, PPO, A2C, SAC, TD3)):
            self.model: object = _SB3Model(
                algo,
                train_steps=train_steps,
                env_params=self.env_params,
                **(algo_kwargs or {}),
            )
        else:
            self.model = _RLModel(alpha, epsilon=epsilon)
        self.meta: dict = {}

    # --------------------------------------------------------------
    def _code_complexity(self, action: str) -> float:
        try:
            recs = self.code_db.search(action)
        except Exception:
            return 0.0
        if not recs:
            return 0.0
        return float(
            sum(r.get("complexity", r.get("complexity_score", 0.0)) for r in recs)
            / len(recs)
        )

    def _roi(self, action: str) -> float:
        try:
            df = self.roi_db.history(action, limit=5)
            if getattr(df, "empty", False):
                return 0.0
            return float(df["revenue"].mean() - df["api_cost"].mean())
        except Exception:
            return 0.0

    def _roi_trend(self, action: str) -> float:
        """Return simple ROI trend for *action* based on history."""
        try:
            df = self.roi_db.history(action, limit=self.env_params.get("trend_window", 5))
            rows = getattr(df, "rows", None)
            if rows is None:
                if getattr(df, "empty", False) or len(df) < 2:
                    return 0.0
                rows = [row for _, row in df.iterrows()]  # type: ignore[attr-defined]
            if len(rows) < 2:
                return 0.0
            rois = [
                (r["revenue"] - r["api_cost"]) / (r.get("cpu_seconds", 1.0) or 1.0) * r.get("success_rate", 1.0)
                for r in rows
            ]
            return float(rois[-1] - rois[0])
        except Exception:
            return 0.0

    def _predict_success(self, action: str, rec: PathwayRecord | None = None) -> float:
        if not self.learning_engine:
            return 0.0
        try:
            if rec:
                return float(
                    self.learning_engine.predict_success(
                        1.0,
                        rec.exec_time,
                        rec.roi,
                        1.0,
                        action,
                    )
                )
            return float(
                self.learning_engine.predict_success(
                    1.0,
                    1.0,
                    self._roi(action),
                    1.0,
                    action,
                )
            )
        except Exception:
            return 0.0

    def _observation_features(
        self, action: str, rec: PathwayRecord | None = None
    ) -> tuple[float, ...]:
        """Return observation features for *action*."""
        if self.observation_features_fn:
            try:
                return tuple(self.observation_features_fn(action, rec))
            except Exception as exc:
                self.logger.exception("observation feature function failed: %s", exc)
        return (
            self._roi_trend(action),
            self._predict_success(action, rec),
            self._code_complexity(action),
        )

    def _reward(self, action: str, rec: PathwayRecord) -> float:
        roi = self._roi(action)
        future = 0.0
        try:
            future = self.roi_db.future_roi(action)
        except Exception:
            future = 0.0
        trend = self._roi_trend(action)
        pred = self._predict_success(action, rec)
        cx = self._code_complexity(action)
        actual = 1.0 if str(rec.outcome).upper().startswith("SUCCESS") else 0.0
        accuracy = 1.0 - abs(actual - pred)
        w = self.reward_weights
        return (
            w["roi"] * roi
            + w["future_roi"] * future
            + w["trend"] * trend
            + w["prediction"] * pred
            + w["accuracy"] * accuracy
            - w["complexity"] * cx
        )

    def _update_from_record(self, rec: PathwayRecord) -> None:
        steps = [s.strip() for s in rec.actions.split("->") if s.strip()]
        if len(steps) < 2:
            return
        for i in range(len(steps) - 1):
            state = tuple(steps[: i + 1])
            next_action = steps[i + 1]
            if self.action_filter_fn and not self.action_filter_fn(next_action):
                continue
            reward = self._reward(next_action, rec)
            obs = self._observation_features(state[-1], rec)
            if isinstance(self.model, _SB3Model):
                self.model.update(state, next_action, reward, obs)
            else:
                self.model.update(state, next_action, reward)
            self.transition_log.append((state, next_action, reward, obs))

    def _load_history(self, batch_size: int = 1000) -> None:
        cur = self.pathway_db.conn.execute(
            "SELECT actions, exec_time, roi FROM pathways"
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for actions, exec_time, roi in rows:
                rec = PathwayRecord(
                    actions=actions,
                    inputs="",
                    outputs="",
                    exec_time=float(exec_time),
                    resources="",
                    outcome="SUCCESS",
                    roi=float(roi),
                )
                self._update_from_record(rec)

    # --------------------------------------------------------------
    def train(self) -> bool:
        if self.learning_engine:
            try:
                self.learning_engine.train()
            except Exception as exc:
                self.logger.exception("learning_engine.train failed: %s", exc)
        self._load_history()
        if hasattr(self.model, "train"):
            try:
                self.model.train()  # type: ignore[attr-defined]
            except Exception as exc:
                self.logger.exception("model training failed: %s", exc)
        return True

    def next_action(self, current_sequence: str) -> Optional[str]:
        steps = [s.strip() for s in current_sequence.split("->") if s.strip()]
        if not steps:
            return None
        state: Tuple[str, ...] = tuple(steps)
        if isinstance(self.model, _SB3Model):
            obs = self._observation_features(state[-1])
            act = self.model.best_action(obs)
        else:
            act = self.model.best_action(state)
        if act and self.action_filter_fn and not self.action_filter_fn(act):
            return None
        return act

    def partial_train(self, record: PathwayRecord) -> bool:
        """Incrementally update the RL model with a single pathway."""
        try:
            self._update_from_record(record)
            if self.learning_engine:
                try:
                    self.learning_engine.partial_train(record)
                except Exception as exc:
                    self.logger.exception(
                        "learning_engine.partial_train failed: %s", exc
                    )
            if hasattr(self.model, "train"):
                try:
                    self.model.train()  # type: ignore[attr-defined]
                except Exception as exc:
                    self.logger.exception("model training failed: %s", exc)
            return True
        except Exception:
            return False

    # --------------------------------------------------------------
    def save_policy(self, path: str) -> bool:
        try:
            if hasattr(self.model, "save"):
                self.model.save(path)  # type: ignore[attr-defined]
            else:
                import pickle
                with open(path, "wb") as fh:
                    pickle.dump(self.model, fh)
            meta = {
                "algo": getattr(self.model, "algo", "custom"),
                "train_steps": getattr(self.model, "train_steps", None),
                "save_time": datetime.utcnow().isoformat(),
            }
            try:
                with open(f"{path}.meta.json", "w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
            except Exception as exc:
                self.logger.exception("failed to write policy metadata: %s", exc)
            self.meta = meta
            return True
        except Exception:
            return False

    def load_policy(self, path: str) -> bool:
        try:
            if hasattr(self.model, "load"):
                self.model.load(path)  # type: ignore[attr-defined]
            else:
                import pickle
                with open(path, "rb") as fh:
                    self.model = pickle.load(fh)
            meta_path = f"{path}.meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        self.meta = json.load(fh)
                except Exception:
                    self.meta = {}
            return True
        except Exception:
            return False

    def get_transition_log(self) -> list[tuple[tuple[str, ...], str, float, tuple[float, ...]]]:
        """Return a list of recently seen transitions."""
        return list(self.transition_log)


__all__ = ["ActionLearningEngine"]
