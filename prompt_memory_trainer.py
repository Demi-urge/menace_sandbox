import json
import re
from collections import defaultdict
from typing import Any, Dict, Mapping

from gpt_memory import GPTMemoryManager
from code_database import PatchHistoryDB


class PromptMemoryTrainer:
    """Analyse historical prompts and patch outcomes to learn formatting.

    The trainer reads prompts from :class:`GPTMemoryManager` and correlates
    them with patch outcomes stored in :class:`PatchHistoryDB`.  Regex based
    heuristics extract style features – headers, example order and tone – and
    additional cues such as code blocks, bullet lists and explicit ``System:``
    / ``User:`` sections.  Success rates are aggregated per style using ROI or
    patch complexity improvement as weights, exposing the resulting metrics via
    :attr:`style_weights`.
    """

    def __init__(
        self,
        *,
        memory: GPTMemoryManager | None = None,
        patch_db: PatchHistoryDB | None = None,
    ) -> None:
        self.memory = memory or GPTMemoryManager(db_path=":memory:")
        self.patch_db = patch_db or PatchHistoryDB(":memory:")
        self.style_weights: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    def _extract_style(self, prompt: str) -> Dict[str, Any]:
        """Return formatting features for ``prompt``.

        Besides the original header, example ordering and tone heuristics the
        method now records whether the prompt contains fenced code blocks,
        bullet lists and ``System``/``User`` sections.
        """

        headers = re.findall(r"^#+\s*(.+)$", prompt, flags=re.MULTILINE)
        example_order = re.findall(r"Example\s*([\w-]+)", prompt, flags=re.IGNORECASE)
        tone = (
            "polite"
            if re.search(r"\b(?:please|kindly)\b", prompt, re.IGNORECASE)
            else "direct"
        )
        has_code = bool(re.search(r"```.+?```", prompt, flags=re.DOTALL))
        has_bullets = bool(
            re.search(r"^\s*(?:[-*]|\d+\.)\s+", prompt, flags=re.MULTILINE)
        )
        sections = [
            m.group(1).lower()
            for m in re.finditer(r"^(System|User):", prompt, flags=re.MULTILINE)
        ]
        return {
            "headers": headers,
            "example_order": example_order,
            "tone": tone,
            "has_code": has_code,
            "has_bullets": has_bullets,
            "sections": sections,
        }

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Dict[str, float]]:
        """Compute success rates for observed prompt styles.

        Each observation is weighted by either ROI improvement or, when ROI is
        unchanged, the reduction in patch complexity.  This emphasises styles
        associated with more impactful patches.
        """

        stats: Dict[str, Dict[str, Mapping[str, float]]] = {
            "headers": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "example_order": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "tone": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "has_code": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "has_bullets": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "sections": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
        }

        cur = self.memory.conn.execute("SELECT prompt FROM interactions")
        for (text,) in cur.fetchall():
            match = re.search(r"PATCH:(\d+)", text)
            if not match:
                continue
            patch_id = int(match.group(1))
            row = self.patch_db.conn.execute(
                "SELECT outcome, roi_before, roi_after, complexity_before, complexity_after FROM patch_history WHERE id=?",
                (patch_id,),
            ).fetchone()
            if not row:
                continue
            outcome, roi_before, roi_after, c_before, c_after = row
            roi_before = roi_before or 0.0
            roi_after = roi_after or 0.0
            c_before = c_before or 0.0
            c_after = c_after or 0.0

            roi_improvement = roi_after - roi_before
            complexity_improvement = c_before - c_after
            weight = roi_improvement if roi_improvement > 0 else complexity_improvement
            if weight <= 0:
                weight = 1.0

            success = (outcome or "").upper() == "SUCCESS"
            feats = self._extract_style(text)
            for key, val in feats.items():
                val_key = json.dumps(val) if isinstance(val, list) else str(val)
                d = stats[key][val_key]
                d["weight"] += weight
                if success:
                    d["success"] += weight

        self.style_weights = {
            feat: {k: (v["success"] / v["weight"]) if v["weight"] else 0.0 for k, v in m.items()}
            for feat, m in stats.items()
        }
        return self.style_weights

    # ------------------------------------------------------------------
    def suggest_style(self) -> Dict[str, Any]:
        """Return the highest scoring formatting style."""

        if not self.style_weights:
            self.train()
        suggestion: Dict[str, Any] = {}
        for feat, mapping in self.style_weights.items():
            if not mapping:
                continue
            best_val, _ = max(mapping.items(), key=lambda kv: kv[1])
            if feat in {"headers", "example_order"}:
                try:
                    suggestion[feat] = json.loads(best_val)
                except Exception:
                    suggestion[feat] = [best_val]
            else:
                suggestion[feat] = best_val
        return suggestion

    # ------------------------------------------------------------------
    def record(self, **_: Any) -> None:  # pragma: no cover - compatibility shim
        """Existing callers may invoke :meth:`record`; it is a no-op."""

        return None


__all__ = ["PromptMemoryTrainer"]
