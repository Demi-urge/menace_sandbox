"""Incremental policy learner using PathwayDB sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, Iterable, Optional
import pickle
import os
import random
import logging

try:
    from .logging_utils import log_record
except Exception:  # pragma: no cover - fallback for optional dependency
    try:
        from logging_utils import log_record  # type: ignore
    except Exception:  # pragma: no cover - last resort
        def log_record(**fields: object) -> dict[str, object]:  # type: ignore
            return fields

from .neuroplasticity import PathwayDB, PathwayRecord
try:
    from .resource_allocation_optimizer import ROIDB
except Exception:  # pragma: no cover - optional dependency
    class ROIDB:  # type: ignore
        def history(self, bot: str | None = None, limit: int = 50):
            import pandas as pd
            return pd.DataFrame()
from .unified_event_bus import UnifiedEventBus
from .growth_utils import growth_score
from .adaptive_roi_predictor import AdaptiveROIPredictor
from .roi_tracker import ROITracker
from sandbox_settings import SandboxSettings

logger = logging.getLogger(__name__)


class _RLModel:
    """Tiny reinforcement learning model updating action weights."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        epsilon: float = 0.1,
        path: Optional[str] = None,
    ) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.path = path
        self.q: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(dict)
        if self.path:
            self.load(self.path)

    def update(self, state: Tuple[Any, ...], action: str, reward: float) -> None:
        values = self.q[state]
        prev = values.get(action, 0.0)
        values[action] = prev + self.alpha * (reward - prev)
        self.q[state] = values
        if self.path:
            self.save(self.path)

    def best_action(self, state: Tuple[Any, ...]) -> Optional[str]:
        values = self.q.get(state)
        if not values:
            return None
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(list(values.keys()))
        return max(values, key=values.get)

    def save(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp:
            return
        try:
            with open(fp, "wb") as fh:
                pickle.dump(dict(self.q), fh)
            logger.info("Saved RL model to %s", fp)
        except Exception:
            logger.exception("Failed to save RL model to %s", fp)
            raise

    def load(self, path: Optional[str] = None) -> None:
        fp = path or self.path
        if not fp or not os.path.exists(fp):
            return
        try:
            with open(fp, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                self.q = defaultdict(dict, data)
            logger.info("Loaded RL model from %s", fp)
        except Exception:
            logger.exception("Failed to load RL model from %s", fp)
            raise

    def get_q_table(self) -> Dict[Tuple[Any, ...], Dict[str, float]]:
        """Return a copy of the internal Q-table."""
        return {state: dict(values) for state, values in self.q.items()}

    def summary(self) -> str:
        """Return string summary of the Q-table."""
        import json

        return json.dumps(self.get_q_table(), indent=2)


class ActionPlanner:
    """Learn action transitions from historical pathways."""

    def __init__(
        self,
        pathway_db: PathwayDB,
        roi_db: ROIDB,
        *,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        state_length: int = 1,
        reward_fn: Callable[[str, PathwayRecord], float] | None = None,
        feature_fn: Callable[[str], Iterable[float]] | None = None,
        multi_step: bool = False,
        discount: float = 0.9,
        model_path: Optional[str] = None,
        event_bus: Optional[UnifiedEventBus] = None,
        roi_predictor: AdaptiveROIPredictor | None = None,
        roi_tracker: ROITracker | None = None,
        use_adaptive_roi: bool | None = None,
        growth_weighting: bool | None = None,
        growth_multipliers: Dict[str, float] | None = None,
    ) -> None:
        self.pathway_db = pathway_db
        self.roi_db = roi_db
        self.model = _RLModel(alpha, epsilon=epsilon, path=model_path)
        self.state_length = max(1, int(state_length))
        self.reward_fn = reward_fn
        self.feature_fn = feature_fn
        self.multi_step = multi_step
        self.discount = discount
        self.event_bus = event_bus
        if use_adaptive_roi is None:
            try:
                use_adaptive_roi = SandboxSettings().adaptive_roi_prioritization
            except Exception:
                use_adaptive_roi = True
        self.use_adaptive_roi = bool(use_adaptive_roi)
        if self.use_adaptive_roi:
            self.roi_predictor = roi_predictor or AdaptiveROIPredictor()
            self.roi_tracker = roi_tracker or ROITracker()
        else:
            self.roi_predictor = None
            self.roi_tracker = None
        if growth_weighting is None:
            try:
                growth_weighting = SandboxSettings().roi_growth_weighting
            except Exception:
                growth_weighting = True
        self.growth_weighting = bool(growth_weighting)
        if growth_multipliers is None:
            try:
                s = SandboxSettings()
                growth_multipliers = {
                    "exponential": s.growth_multiplier_exponential,
                    "linear": s.growth_multiplier_linear,
                    "marginal": s.growth_multiplier_marginal,
                }
            except Exception:
                growth_multipliers = {
                    cat: 1.0 + growth_score(cat)
                    for cat in ("exponential", "linear", "marginal")
                }
        self.growth_multipliers = growth_multipliers
        self.priority_weights: Dict[str, float] = {}
        if self.event_bus:
            try:
                self.event_bus.subscribe("pathway:new", self._on_new_pathway)
            except Exception:
                logger.exception("Event bus subscription failed")
                raise
        self._load_history()

    # ------------------------------------------------------------------
    def update_priorities(self, weights: Dict[str, float]) -> None:
        """Update internal priority weights for actions.

        Each provided base weight is scaled by the predicted ROI and a
        growth-type multiplier derived from :class:`AdaptiveROIPredictor`.
        Missing predictors or prediction errors fall back to the supplied
        weight without scaling.
        """

        for action, base in weights.items():
            try:
                weight = float(base)
            except Exception:
                logger.exception("priority weight casting failed")
                continue
            if (
                self.use_adaptive_roi
                and self.roi_predictor
                and self.feature_fn
            ):
                try:
                    feats = [list(self.feature_fn(action))]
                    try:
                        seq, growth, _, _ = self.roi_predictor.predict(
                            feats, horizon=len(feats)
                        )
                    except TypeError:
                        val, growth, _, _ = self.roi_predictor.predict(feats)
                        seq = [[float(val)]]
                    if seq and isinstance(seq[0], (list, tuple)):
                        roi_seq = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in seq]
                    else:
                        roi_seq = [float(x) for x in seq]
                    roi_est = roi_seq[-1] if roi_seq else 1.0
                    mult = (
                        self.growth_multipliers.get(growth, 1.0)
                        if self.growth_weighting
                        else 1.0
                    )
                    weight *= roi_est * mult
                except Exception:
                    logger.exception("roi prediction failed for %s", action)
            self.priority_weights[action] = weight

    def get_priority_queue(self) -> list[str]:
        """Return actions ordered by priority weight."""
        return [
            a
            for a, _ in sorted(
                self.priority_weights.items(), key=lambda x: x[1], reverse=True
            )
        ]

    # ------------------------------------------------------------------
    def _roi(self, action: str) -> float:
        """Return average ROI for *action* from ROIDB."""
        try:
            df = self.roi_db.history(action, limit=5)
            if df.empty:
                return 0.0
            rev = float(df["revenue"].mean())
            cost = float(df["api_cost"].mean())
            cpu = float(df["cpu_seconds"].mean()) or 1.0
            return (rev - cost) / cpu
        except Exception:
            return 0.0

    def _predict_growth(
        self, action: str
    ) -> tuple[list[float], str, float | None]:
        """Return predicted ROI sequence, growth category and confidence."""

        if not self.roi_predictor or not self.feature_fn:
            return [], "marginal", None
        feats = [list(self.feature_fn(action))]
        try:
            result = self.roi_predictor.predict(feats, horizon=len(feats))
        except TypeError:
            result = self.roi_predictor.predict(feats)

        seq: list[float] | float
        category: str
        growth_conf: float | None

        if isinstance(result, tuple):
            if len(result) == 4:
                seq, category, _, growth_conf = result
            elif len(result) == 3:
                seq, category, growth_conf = result
            elif len(result) == 2:
                seq, category = result
                growth_conf = None
            else:
                seq = result[0]
                category = "marginal"
                growth_conf = None
        else:
            seq = result
            category = "marginal"
            growth_conf = None

        if isinstance(seq, (list, tuple)):
            seq = [float(x) for x in seq]
        else:
            seq = [float(seq)]

        # Update priority weights based on predicted growth category.
        # This ensures actions with exponential forecasts receive higher
        # priority while downgraded predictions decay their influence.
        try:
            mult = self.growth_multipliers.get(category, 1.0)
            self.priority_weights[action] = mult
        except Exception:
            logger.exception("priority weight update from growth failed")

        return seq, category, growth_conf

    def _reward(self, action: str, rec: PathwayRecord) -> float:
        """Return reward for taking *action* in *rec*."""
        if self.reward_fn:
            try:
                reward = float(self.reward_fn(action, rec))
            except Exception as exc:
                logger.exception("reward_fn failed: %s", exc)
                reward = 0.0
        else:
            reward = self._roi(action) or rec.roi
        success = str(rec.outcome).upper().startswith("SUCCESS")
        if (
            self.growth_weighting
            and self.use_adaptive_roi
            and self.roi_predictor
            and self.feature_fn
        ):
            try:
                seq, category, growth_conf = self._predict_growth(action)
                roi_est = float(seq[-1]) if seq else 0.0
                conf = 1.0 if growth_conf is None else float(growth_conf)
                # Growth multiplier already applied via priority weights;
                # only scale by confidence here.
                reward *= conf
                if self.roi_tracker:
                    try:
                        self.roi_tracker.record_prediction(
                            [roi_est],
                            [rec.roi],
                            predicted_class=category,
                            actual_class=None,
                            confidence=growth_conf,
                        )
                        logger.info(
                            "roi prediction",
                            extra=log_record(
                                action=action,
                                predicted_roi=roi_est,
                                actual_roi=rec.roi,
                                growth=category,
                            ),
                        )
                    except Exception:
                        logger.exception("roi tracker record failed")
            except Exception:
                logger.exception("growth weighting failed")
        if action in self.priority_weights:
            try:
                reward *= float(self.priority_weights[action])
            except Exception:
                logger.exception("priority weight application failed")
        if not success:
            reward = -abs(reward)
        return reward

    def _state_key(self, steps: Iterable[str]) -> Tuple[Any, ...]:
        """Return state key from *steps* applying feature_fn if set."""
        if not self.feature_fn:
            return tuple(steps)
        feats: list[float] = []
        for s in steps:
            try:
                fvals = self.feature_fn(s)
                feats.extend(float(x) for x in fvals)
            except Exception as exc:
                logger.exception("feature_fn failed: %s", exc)
                feats.append(float(hash(s) % 1000))
        return tuple(feats)

    def _update_from_record(self, rec: PathwayRecord) -> None:
        steps = [s.strip() for s in rec.actions.split("->") if s.strip()]
        if len(steps) < 2:
            return
        for i in range(len(steps) - 1):
            next_action = steps[i + 1]
            reward = self._reward(next_action, rec)
            for k in range(1, self.state_length + 1):
                if i - k + 1 < 0:
                    break
                state_steps = steps[i - k + 1 : i + 1]
                state = self._state_key(state_steps)
                r = reward * (self.discount ** (k - 1)) if self.multi_step else reward
                self.model.update(state, next_action, r)

    def _load_history(self) -> None:
        cur = self.pathway_db.conn.execute(
            "SELECT actions, outcome, roi FROM pathways"
        )
        rows = cur.fetchall()
        for actions, outcome, roi in rows:
            rec = PathwayRecord(
                actions=actions,
                inputs="",
                outputs="",
                exec_time=0.0,
                resources="",
                outcome=outcome or "SUCCESS",
                roi=float(roi),
            )
            self._update_from_record(rec)

    # ------------------------------------------------------------------
    def predict_next_action(self, current: str) -> Optional[str]:
        steps = [s.strip() for s in current.split("->") if s.strip()]
        if not steps:
            return None
        state_steps = steps[-self.state_length :]
        state = self._state_key(state_steps)
        values = self.model.q.get(state)
        if not values:
            return None
        if self.roi_predictor and self.use_adaptive_roi and self.feature_fn:
            scored: list[tuple[str, float, str, float]] = []
            roi_vals: list[float] = []
            for action, val in values.items():
                try:
                    seq, category, _ = self._predict_growth(action)
                    roi_est = float(seq[-1]) if seq else 0.0
                except Exception:
                    category = "marginal"
                    roi_est = 0.0
                val *= self.priority_weights.get(action, 1.0)
                scored.append((action, val, category, float(roi_est)))
                roi_vals.append(float(roi_est))
            if roi_vals:
                max_roi = max(roi_vals)
                min_roi = min(roi_vals)
            else:
                max_roi = min_roi = 0.0

            def _norm(r: float) -> float:
                if max_roi == min_roi:
                    return 0.0
                return (r - min_roi) / (max_roi - min_roi)

            scored.sort(
                key=lambda x: (
                    -(
                        growth_score(x[2])
                        + _norm(x[3])
                    ),
                    -x[1],
                )
            )
            return scored[0][0]
        return max(values, key=values.get)

    # ------------------------------------------------------------------
    def plan_actions(
        self, current: str, candidates: Iterable[str]
    ) -> list[str]:
        """Rank *candidates* for the next action from *current* state.

        Adaptive ROI predictions are used to adjust an improvement score
        for each candidate based on expected growth category.  The
        resulting list is ordered from highest to lowest score.
        """

        steps = [s.strip() for s in current.split("->") if s.strip()]
        state_steps = steps[-self.state_length :] if steps else []
        state = self._state_key(state_steps)
        values = self.model.q.get(state, {})
        scored: list[tuple[str, float, float, str]] = []
        for action in candidates:
            base = float(values.get(action, 0.0))
            score = base
            roi_est = 0.0
            category = "marginal"
            if (
                self.roi_predictor
                and self.use_adaptive_roi
                and self.feature_fn
            ):
                try:
                    seq, category, _ = self._predict_growth(action)
                    roi_est = float(seq[-1]) if seq else 0.0
                except Exception:
                    roi_est = 0.0
                    category = "marginal"
            mult = self.growth_multipliers.get(category, 1.0)
            score += roi_est * mult
            score *= self.priority_weights.get(action, 1.0)
            scored.append((action, score, roi_est, category))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and self.roi_tracker:
            top_action, _, pred_roi, cat = scored[0]
            try:
                actual_roi = self._roi(top_action)
                self.roi_tracker.record_roi_prediction(
                    [pred_roi], [actual_roi], predicted_class=cat
                )
                logger.info(
                    "action prediction",
                    extra=log_record(
                        action=top_action,
                        predicted_roi=pred_roi,
                        actual_roi=actual_roi,
                        growth=cat,
                    ),
                )
            except Exception:
                logger.exception("roi tracker record failed")
        return [a for a, _, _, _ in scored]

    # ------------------------------------------------------------------
    def _on_new_pathway(self, topic: str, payload: object) -> None:
        if isinstance(payload, dict):
            try:
                rec = PathwayRecord(
                    actions=payload.get("actions", ""),
                    inputs="",
                    outputs="",
                    exec_time=0.0,
                    resources="",
                    outcome=payload.get("outcome", "SUCCESS"),
                    roi=float(payload.get("roi", 0.0)),
                )
                self._update_from_record(rec)
            except Exception:
                logger.exception("Failed to process pathway event")
                raise


__all__ = ["ActionPlanner"]
