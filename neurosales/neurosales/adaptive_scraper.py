from __future__ import annotations

import math
from typing import Dict, List


class AdaptiveWebScraper:
    """Adapt crawl keywords and depth based on entropy spikes and RL feedback."""

    def __init__(self, keywords: List[str], *, crawl_depth: int = 1) -> None:
        self.keywords = list(keywords)
        self.crawl_depth = crawl_depth
        self.entropy = 0.0
        self.source_priority: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t.strip(".,()\"'`").lower() for t in text.split() if t]

    def _entropy(self, tokens: List[str]) -> float:
        total = len(tokens) or 1
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return -sum((c / total) * math.log(c / total) for c in freq.values())

    # ------------------------------------------------------------------
    def nightly_update(
        self, source: str, abstracts: List[str], rl_scores: List[float]
    ) -> None:
        tokens = []
        for abs_txt in abstracts:
            tokens.extend(self._tokenize(abs_txt))
        new_entropy = self._entropy(tokens)
        if new_entropy > self.entropy * 1.1 and tokens:
            # detect novel terminology burst
            new_terms = [t for t in tokens if t not in self.keywords]
            self.keywords.extend(new_terms[:3])
            self.crawl_depth += 1
        self.entropy = max(self.entropy, new_entropy)

        avg_score = sum(rl_scores) / (len(rl_scores) or 1)
        base = self.source_priority.get(source, 1.0)
        if avg_score > 0.1:
            self.source_priority[source] = base + 0.5
        else:
            self.source_priority[source] = max(0.1, base * 0.5)
