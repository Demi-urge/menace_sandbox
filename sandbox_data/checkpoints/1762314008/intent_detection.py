from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

try:
    from bertopic import BERTopic  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    BERTopic = None

from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class IntentDetector:
    """Unsupervised intent detection with optional clustering backends."""

    def __init__(
        self,
        *,
        num_clusters: int = 5,
        threshold: float = 0.5,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 2,
    ) -> None:
        self.num_clusters = num_clusters
        self.threshold = threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.cluster_labels: List[str] = []
        self.classifier: OneVsRestClassifier | None = None
        self.correlation: Dict[Tuple[str, str], int] = defaultdict(int)
        self.messages: List[str] = []

    def _cluster_kmeans(self, X) -> List[int]:
        model = KMeans(n_clusters=self.num_clusters, n_init="auto")
        return model.fit_predict(X)

    def _cluster_dbscan(self, X) -> List[int]:
        model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        return model.fit_predict(X)

    def _cluster_bertopic(self, messages: List[str]) -> List[int]:
        if BERTopic is None:
            raise RuntimeError("BERTopic is not installed")
        topic_model = BERTopic(verbose=False)
        labels, _ = topic_model.fit_transform(messages)
        return labels

    def _generate_label(
        self, indices: List[int], feature_names: List[str], X
    ) -> str:
        counts = X[indices].sum(axis=0).A1
        top_idx = counts.argsort()[-3:][::-1]
        words = [feature_names[i] for i in top_idx if counts[i] > 0]
        return " ".join(words)

    def fit(self, messages: Iterable[str]) -> None:
        self.messages = list(messages)
        X = self.vectorizer.fit_transform(self.messages)
        try:
            labels = self._cluster_bertopic(self.messages)
        except Exception:
            try:
                labels = self._cluster_kmeans(X)
            except Exception:
                labels = self._cluster_dbscan(X)
        unique = sorted(set(labels))
        feature_names = self.vectorizer.get_feature_names_out()
        label_names: Dict[int, str] = {}
        for u in unique:
            indices = [i for i, l in enumerate(labels) if l == u]
            name = self._generate_label(indices, feature_names, X)
            label_names[u] = name or f"cluster_{u}"
        self.cluster_labels = [label_names[u] for u in unique]
        y = label_binarize(labels, classes=unique)
        self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.classifier.fit(X, y)

    def predict(self, text: str) -> Dict[str, float]:
        if not self.classifier:
            raise RuntimeError("Model not fitted")
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        return {label: float(p) for label, p in zip(self.cluster_labels, probs)}

    def detect_intents(self, text: str) -> List[Tuple[str, float]]:
        scores = self.predict(text)
        intents = [
            (label, score) for label, score in scores.items() if score >= self.threshold
        ]
        if intents:
            for i, _ in intents:
                for j, _ in intents:
                    if i < j:
                        self.correlation[(i, j)] += 1
        else:
            intents.append(("clarify", 1.0))
        return intents

    def adjust_threshold(self, feedback_accuracy: float) -> None:
        if feedback_accuracy < 0.5:
            self.threshold = max(0.1, self.threshold * 0.9)
        else:
            self.threshold = min(0.9, self.threshold * 1.05)

    def partial_fit(self, new_messages: Iterable[str]) -> None:
        self.fit(self.messages + list(new_messages))
