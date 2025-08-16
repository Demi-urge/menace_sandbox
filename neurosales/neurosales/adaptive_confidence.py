from __future__ import annotations

import time
from typing import Dict, Optional


class AdaptiveConfidenceScorer:
    """Combine submodule confidences and adapt thresholds per user.

    This scorer also maintains per-topic confidence curves that decay over
    time. Thresholds are adjusted using explicit user corrections and implicit
    cues like negative emojis or shortened responses.
    """

    def __init__(
        self,
        *,
        base_threshold: float = 0.5,
        decay_rate: float = 0.99,
    ) -> None:
        self.base_threshold = base_threshold
        self.decay_rate = decay_rate
        self.user_thresholds: Dict[str, float] = {}
        self.topic_confidence: Dict[str, float] = {}
        self.topic_last: Dict[str, float] = {}

    def _aggregate(self, scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    def overall_confidence(self, scores: Dict[str, float], *, topic: Optional[str] = None) -> float:
        """Return average confidence optionally adjusted by topic."""
        base = self._aggregate(scores)
        if topic is not None:
            self.decay_topics()
            topic_conf = self.topic_confidence.get(topic, 0.5)
            return (base + topic_conf) / 2
        return base

    def get_threshold(self, user_id: str) -> float:
        return self.user_thresholds.get(user_id, self.base_threshold)

    def should_clarify(
        self, user_id: str, scores: Dict[str, float], *, topic: Optional[str] = None
    ) -> bool:
        conf = self.overall_confidence(scores, topic=topic)
        return conf < self.get_threshold(user_id)

    def update_with_feedback(
        self,
        user_id: str,
        *,
        corrected: bool,
        response_length_drop: float = 0.0,
        negative_emojis: int = 0,
        latency_increase: float = 0.0,
    ) -> None:
        """Adjust the user's threshold based on feedback signals."""
        thr = self.get_threshold(user_id)
        if corrected:
            thr = min(0.9, thr + 0.05)
        else:
            thr = max(0.1, thr - 0.03)
        thr += 0.02 * response_length_drop
        thr += 0.01 * negative_emojis
        thr += 0.02 * latency_increase
        thr = max(0.1, min(0.9, thr))
        self.user_thresholds[user_id] = thr

    # ---------------------- topic confidence utils ----------------------
    def record_topic_result(self, topic: str, *, success: bool) -> None:
        conf = self.topic_confidence.get(topic, 0.5)
        conf = conf * (1.1 if success else 0.9)
        self.topic_confidence[topic] = max(0.0, min(1.0, conf))
        self.topic_last[topic] = time.time()

    def decay_topics(self) -> None:
        now = time.time()
        for topic, last in list(self.topic_last.items()):
            if now - last > 3600:  # one hour of inactivity
                self.topic_confidence[topic] = self.topic_confidence.get(topic, 0.5) * self.decay_rate
                self.topic_last[topic] = now

    # ---------------------- response modulation ----------------------
    def modulate_voice(self, scores: Dict[str, float], *, user_id: Optional[str] = None) -> str:
        """Return a style keyword based on overall confidence and user threshold."""
        conf = self.overall_confidence(scores)
        thr = self.get_threshold(user_id) if user_id is not None else self.base_threshold
        if conf >= thr + 0.15:
            return "cta"
        if conf >= thr - 0.05:
            return "hedge"
        return "clarify"

    def stamp_response(self, user_id: str, metrics: Dict[str, float], *, topic: Optional[str] = None) -> tuple[float, str]:
        """Return confidence and style label for a response."""
        conf = self.overall_confidence(metrics, topic=topic)
        style = self.modulate_voice(metrics, user_id=user_id)
        return conf, style

