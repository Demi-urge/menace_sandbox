from __future__ import annotations

import random
from collections import Counter, deque
from typing import Deque, Dict, Iterable, List

try:
    from bertopic import BERTopic  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    BERTopic = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    hdbscan = None

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional deps
    KMeans = None  # type: ignore
    TfidfVectorizer = None  # type: ignore

from .few_shot_learning import FewShotZeroShotClassifier


class TriggerPromptCache:
    """Rolling cache of exemplar prompts per persuasion trigger."""

    def __init__(self, max_prompts: int = 5) -> None:
        self.max_prompts = max_prompts
        self.cache: Dict[str, Deque[str]] = {}

    def add(self, trigger: str, prompt: str) -> None:
        q = self.cache.setdefault(trigger, deque(maxlen=self.max_prompts))
        if prompt not in q:
            q.append(prompt)

    def get(self, trigger: str) -> List[str]:
        return list(self.cache.get(trigger, []))

    def all_prompts(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self.cache.items()}


class FewShotPromptEngine:
    """Generate new prompts and seed novel triggers via clustering."""

    def __init__(self, threshold: float = 0.5, max_prompts: int = 5) -> None:
        self.cache = TriggerPromptCache(max_prompts=max_prompts)
        self.classifier = FewShotZeroShotClassifier(threshold=threshold)
        self.vectorizer = TfidfVectorizer(stop_words="english") if TfidfVectorizer else None
        self._cluster_counter = 0

    # ---------------------------- prompt utilities ----------------------------
    def ingest(self, trigger: str, prompt: str) -> None:
        """Add a prompt to the cache and training set."""
        self.cache.add(trigger, prompt)
        self.classifier.add_examples(trigger, [prompt])

    def generate_variants(self, trigger: str, jargon_terms: Iterable[str]) -> List[str]:
        """Return simple extrapolated variants using stored exemplars."""
        examples = self.cache.get(trigger)
        variants: List[str] = []
        if not examples:
            return variants
        for ex in examples:
            for j in jargon_terms:
                variants.append(f"{j} {ex}")
        return variants

    # ---------------------------- clustering helpers ----------------------------
    def _cluster_hdbscan(self, X) -> List[int]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        return clusterer.fit_predict(X)

    def _cluster_kmeans(self, X, k: int) -> List[int]:
        model = KMeans(n_clusters=k, n_init="auto")
        return model.fit_predict(X)

    def cluster_texts(self, texts: List[str]) -> Dict[str, List[str]]:
        """Cluster raw text and return mapping cluster_name -> texts."""
        if not texts:
            return {}
        if BERTopic is not None:
            model = BERTopic(verbose=False)
            labels, _ = model.fit_transform(texts)
        else:
            if self.vectorizer is None or KMeans is None:
                return {f"cluster_0": texts}
            X = self.vectorizer.fit_transform(texts)
            if hdbscan is not None:
                try:
                    labels = self._cluster_hdbscan(X.toarray())
                except Exception:
                    labels = self._cluster_kmeans(X, min(2, len(texts)))
            else:
                labels = self._cluster_kmeans(X, min(2, len(texts)))
        clusters: Dict[int, List[str]] = {}
        for t, l in zip(texts, labels):
            clusters.setdefault(int(l), []).append(t)
        named: Dict[str, List[str]] = {}
        for lbl, items in clusters.items():
            name = self._name_cluster(items)
            named[name] = items
        return named

    def _name_cluster(self, texts: List[str]) -> str:
        """Name a cluster by its two most common words."""
        words = [w.strip(".,!?\"'").lower() for t in texts for w in t.split()]
        common = [w for w, _ in Counter(words).most_common(2)]
        if not common:
            self._cluster_counter += 1
            return f"cluster_{self._cluster_counter}"
        return "_".join(common)

    def seed_new_slots(self, texts: List[str]) -> List[str]:
        """Cluster texts and seed new categories."""
        clusters = self.cluster_texts(texts)
        new_labels: List[str] = []
        for label, examples in clusters.items():
            new_labels.append(label)
            self.classifier.add_examples(label, examples[:3])
            for ex in examples[:3]:
                self.cache.add(label, ex)
        return new_labels

    def contrastive_fine_tune(self, label: str, examples: Iterable[str]) -> None:
        """Fine-tune classifier with tiny synthetic examples."""
        self.classifier.add_examples(label, list(examples))

