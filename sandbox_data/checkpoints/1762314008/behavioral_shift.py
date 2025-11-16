from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    import numpy as np
except Exception:  # pragma: no cover - optional heavy dep
    IsolationForest = None  # type: ignore
    MLPRegressor = None  # type: ignore
    np = None  # type: ignore

from .anomaly_detection import AnomalyDetector


class BehavioralShiftDetector:
    """Detect major behavioural shifts using ML models when available."""

    SWEAR_WORDS = {"damn", "shit", "fuck", "hell"}
    FORMAL_WORDS = {"sir", "madam", "dear", "sincerely"}

    def __init__(self, *, contamination: float = 0.1) -> None:
        self.contamination = contamination
        self._forest: Optional[IsolationForest] = None
        self._ae: Optional[MLPRegressor] = None
        self._trained = False
        self._last_time: Dict[str, float] = {}
        self._fallback = AnomalyDetector()

    # ------------------------------------------------------------------
    def _vector(self, text: str, timestamp: float, prev_time: float) -> List[float]:
        length = len(text)
        punct = text.count("!") + text.count("?")
        letters = sum(1 for c in text if c.isalpha())
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, letters)
        time_gap = timestamp - prev_time if prev_time else 0.0
        has_swear = int(any(w in text.lower() for w in self.SWEAR_WORDS))
        has_formal = int(any(w in text.lower() for w in self.FORMAL_WORDS))
        return [length, punct, upper_ratio, time_gap, has_swear, has_formal]

    # ------------------------------------------------------------------
    def fit(self, texts: List[str], *, timestamps: Optional[List[float]] = None) -> None:
        if IsolationForest is None or MLPRegressor is None or np is None:
            # fallback to heuristic baseline
            for i, t in enumerate(texts):
                ts = timestamps[i] if timestamps else i
                self._fallback.detect("_train", t, timestamp=ts)
            self._trained = True
            return

        timestamps = timestamps or list(range(len(texts)))
        prev = timestamps[0]
        X = []
        for text, ts in zip(texts, timestamps):
            vec = self._vector(text, ts, prev)
            prev = ts
            X.append(vec)
        X_arr = np.array(X)
        self._forest = IsolationForest(contamination=self.contamination, random_state=0)
        self._forest.fit(X_arr)
        self._ae = MLPRegressor(hidden_layer_sizes=(8,), max_iter=200, random_state=0)
        self._ae.fit(X_arr, X_arr)
        self._trained = True

    # ------------------------------------------------------------------
    def detect(self, user_id: str, text: str, *, timestamp: Optional[float] = None) -> float:
        ts = time.time() if timestamp is None else timestamp
        prev = self._last_time.get(user_id, ts)
        self._last_time[user_id] = ts

        if not self._trained:
            # Nothing learned yet, rely on simple heuristic
            return self._fallback.detect(user_id, text, timestamp=ts)

        vec = self._vector(text, ts, prev)

        if self._forest is None or self._ae is None or np is None:
            return self._fallback.detect(user_id, text, timestamp=ts)

        X = np.array([vec])
        forest_score = -self._forest.score_samples(X)[0]
        recon = self._ae.predict(X)[0]
        mse = float(np.mean((recon - X[0]) ** 2))
        return (forest_score + mse) / 2

    # ------------------------------------------------------------------
    def session_anomaly_score(self, texts: List[str]) -> float:
        ts = 0.0
        scores = []
        for text in texts:
            score = self.detect("_sess", text, timestamp=ts)
            scores.append(score)
            ts += 1.0
        return sum(scores) / len(scores) if scores else 0.0
