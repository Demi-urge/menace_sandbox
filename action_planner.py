"""Incremental policy learner using PathwayDB sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, Iterable, Optional
import pickle
import os
import random
import logging
import json
import numpy as np

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
from vector_service import CognitionLayer
from vector_service.context_builder import ContextBuilder
from .governance import check_veto, load_rules

logger = logging.getLogger(__name__)

RULES = load_rules()


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
        context_builder: ContextBuilder,
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
        cognition_layer: CognitionLayer | None = None,
    ) -> None:
        self.pathway_db = pathway_db
        self.roi_db = roi_db
        self.context_builder = context_builder
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
        if cognition_layer is None:
            try:
                cognition_layer = CognitionLayer(
                    context_builder=context_builder, roi_tracker=self.roi_tracker
                )
            except Exception:  # pragma: no cover - optional dependency
                cognition_layer = None
        self.cognition_layer = cognition_layer
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
                    base_roi, raroi, _ = (
                        self.roi_tracker.calculate_raroi(
                            roi_est, workflow_type="standard", metrics={}
                        )
                        if self.roi_tracker
                        else (roi_est, roi_est, [])
                    )
                    if self.roi_tracker:
                        final_score, needs_review, confidence = self.roi_tracker.score_workflow(
                            action, raroi
                        )
                    else:
                        final_score, needs_review, confidence = raroi, False, 1.0
                    if not needs_review:
                        weight *= final_score * mult
                    logger.debug(
                        "priority roi scaled",
                        extra=log_record(
                            action=action,
                            base_roi=base_roi,
                            raroi=raroi,
                            final_score=final_score,
                            confidence=confidence,
                        ),
                    )
                    if needs_review:
                        logger.info(
                            "priority weighting deferred for review",
                            extra=log_record(
                                action=action,
                                confidence=confidence,
                                threshold=getattr(self.roi_tracker, "confidence_threshold", 0.5),
                                human_review=True,
                            ),
                        )
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
    def _roi(self, action: str) -> tuple[float, float]:
        """Return ``(base_roi, raroi)`` for *action* from ROIDB."""
        try:
            df = self.roi_db.history(action, limit=5)
            if df.empty:
                return 0.0, 0.0
            rev = float(df["revenue"].mean())
            cost = float(df["api_cost"].mean())
            cpu = float(df["cpu_seconds"].mean()) or 1.0
            base = (rev - cost) / cpu
            if self.roi_tracker:
                try:
                    _base, raroi, _ = self.roi_tracker.calculate_raroi(
                        base, workflow_type="standard", metrics={}
                    )
                except Exception:
                    raroi = base
            else:
                raroi = base
            logger.debug(
                "roi lookup",
                extra=log_record(action=action, base_roi=base, raroi=raroi),
            )
            return base, raroi
        except Exception:
            return 0.0, 0.0

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
        success = True
        if self.reward_fn:
            try:
                reward = float(self.reward_fn(action, rec))
            except Exception as exc:
                logger.exception("reward_fn failed: %s", exc)
                reward = 0.0
        else:
            base_roi, raroi = self._roi(action)
            if self.roi_tracker:
                final_score, needs_review, confidence = self.roi_tracker.score_workflow(
                    action, raroi
                )
                reward = final_score
            else:
                reward, needs_review, confidence = raroi, False, 1.0
            if reward == 0.0 and self.roi_tracker:
                try:
                    _, raroi, _ = self.roi_tracker.calculate_raroi(
                        rec.roi, workflow_type="standard", metrics={}
                    )
                    reward, needs_review, confidence = self.roi_tracker.score_workflow(
                        action, raroi
                    )
                except Exception:
                    reward = rec.roi
                    needs_review = False
                    confidence = 1.0
            elif reward == 0.0:
                reward = rec.roi
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
                        pred_base, pred_raroi, _ = self.roi_tracker.calculate_raroi(
                            roi_est, workflow_type="standard", metrics={}
                        )
                        actual_base, actual_raroi, _ = self.roi_tracker.calculate_raroi(
                            rec.roi, workflow_type="standard", metrics={}
                        )
                        self.roi_tracker.record_prediction(
                            [pred_raroi],
                            [actual_raroi],
                            predicted_class=category,
                            actual_class=None,
                            confidence=growth_conf,
                        )
                        logger.info(
                            "roi prediction",
                            extra=log_record(
                                action=action,
                                predicted_roi=pred_raroi,
                                actual_roi=actual_raroi,
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
    def _context_metric(self, action: str, ctx: Dict[str, Any]) -> float:
        """Return a metric boost for *action* from context data."""

        for bucket in ("bots", "workflows", "errors", "discrepancies", "code"):
            for entry in ctx.get(bucket, []):
                name = entry.get("name") or entry.get("title") or ""
                if isinstance(name, str) and name.lower() == action.lower():
                    try:
                        return float(entry.get("metric", 0.0))
                    except Exception:
                        return 0.0
        return 0.0

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
        ranked = self.plan_actions(current, values.keys())
        return ranked[0] if ranked else None

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
        ctx: Dict[str, Any] = {}
        if self.cognition_layer:
            try:
                result, _sid = self.cognition_layer.query(current)
                ctx = json.loads(result) if result else {}
            except Exception:
                ctx = {}
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
            raroi_est = roi_est
            final_score = raroi_est
            if self.roi_tracker:
                try:
                    _, raroi_est, _ = self.roi_tracker.calculate_raroi(
                        roi_est, workflow_type="standard", metrics={}
                    )
                    final_score, _needs_review, _conf = self.roi_tracker.score_workflow(
                        action, raroi_est
                    )
                except Exception:
                    raroi_est = roi_est
                    final_score = raroi_est
            score += final_score * mult
            score *= self.priority_weights.get(action, 1.0)
            if ctx:
                score += self._context_metric(action, ctx)
            scorecard = {
                "decision": action,
                "alignment": ctx.get("alignment", "pass"),
                "raroi_increase": ctx.get("raroi_increase", 0),
            }
            veto = check_veto(scorecard, RULES)
            if veto:
                logger.info(
                    "action vetoed", extra=log_record(action=action, veto=";".join(veto))
                )
                continue
            scored.append((action, score, raroi_est, category))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and self.roi_tracker:
            top_action, _, pred_roi, cat = scored[0]
            try:
                _base, actual_raroi = self._roi(top_action)
                ta = self.roi_tracker.truth_adapter
                pred_cal, _ = ta.predict(
                    np.array([[float(pred_roi)]], dtype=np.float64)
                )
                act_cal, _ = ta.predict(
                    np.array([[float(actual_raroi)]], dtype=np.float64)
                )
                pred_roi = float(pred_cal[0])
                actual_raroi = float(act_cal[0])
                self.roi_tracker.record_roi_prediction(
                    [pred_roi],
                    [actual_raroi],
                    predicted_class=cat,
                    workflow_id=str(top_action),
                )
                logger.info(
                    "action prediction",
                    extra=log_record(
                        action=top_action,
                        predicted_roi=pred_roi,
                        actual_roi=actual_raroi,
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
