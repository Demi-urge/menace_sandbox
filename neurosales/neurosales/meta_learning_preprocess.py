from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .few_shot_learning import FewShotZeroShotClassifier


@dataclass
class TrainingSample:
    category: str
    text: str
    score: float


class MetaLearningPreprocessor:
    """Simplified meta-learning preprocessing layer."""

    def __init__(self, base_threshold: float = 0.5) -> None:
        self.base_threshold = base_threshold
        self.category_thresholds: Dict[str, float] = {}
        self.samples: Dict[str, List[TrainingSample]] = {}
        self.seen: set[str] = set()
        self.refinement_log: List[str] = []
        self.classifier = FewShotZeroShotClassifier()

    # --------------------- scoring helpers ---------------------
    def data_utility_score(self, text: str) -> float:
        tokens = [t.strip(".,!?\"'`").lower() for t in text.split() if t]
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        noise_ratio = sum(1 for t in tokens if not t.isalpha()) / len(tokens)
        duplicate_penalty = 1.0 if text in self.seen else 0.0
        return unique_ratio * (1 - noise_ratio) - duplicate_penalty

    # --------------------- sample ingestion ---------------------
    def add_sample(self, category: str, text: str) -> bool:
        score = self.data_utility_score(text)
        threshold = self.category_thresholds.get(category, self.base_threshold)
        if score < threshold:
            return False
        sample = TrainingSample(category, text, score)
        self.samples.setdefault(category, []).append(sample)
        self.seen.add(text)
        self.classifier.add_examples(category, [text])
        self.category_thresholds.setdefault(category, self.base_threshold)
        return True

    def rebalance(self) -> None:
        if not self.samples:
            return
        counts = {c: len(s) for c, s in self.samples.items()}
        max_count = max(counts.values())
        for cat, items in self.samples.items():
            while len(items) < max_count and items:
                items.append(items[-1])

    # --------------------- feedback and evaluation ---------------------
    def update_feedback(self, category: str, confirmed: bool) -> None:
        thr = self.category_thresholds.get(category, self.base_threshold)
        if confirmed:
            thr = min(1.0, thr + 0.05)
        else:
            thr = max(0.0, thr - 0.05)
        self.category_thresholds[category] = thr

    def self_evaluate(self) -> None:
        for cat, samples in self.samples.items():
            for s in samples:
                pred, _ = self.classifier.classify(s.text)
                if pred != cat:
                    self.classifier.log_feedback(s.text, pred, cat)
                    self.refinement_log.append(s.text)

    def classify(self, text: str) -> Tuple[str, float]:
        return self.classifier.classify(text)

    def log_refinement(self, text: str) -> None:
        self.refinement_log.append(text)
