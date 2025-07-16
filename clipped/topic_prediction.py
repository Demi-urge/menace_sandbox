"""Very small topic prediction engine for tests."""

from __future__ import annotations

from typing import List, Dict, Any


class TopicPredictionEngine:
    """Suggest trending topics not already present."""

    _CANDIDATES = [
        "AI",
        "Machine Learning",
        "Data Science",
        "Blockchain",
        "Cybersecurity",
        "Robotics",
    ]

    def suggest_topics(self, existing: List[str]) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        for topic in self._CANDIDATES:
            if topic not in existing:
                suggestions.append(
                    {
                        "name": topic,
                        "trend_velocity": 1.0,
                        "similarity": 0.5,
                        "projected_profit_density": 1.0,
                    }
                )
            if len(suggestions) >= 3:
                break
        return suggestions

__all__ = ["TopicPredictionEngine"]
