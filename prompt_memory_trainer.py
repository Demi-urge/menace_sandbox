import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping

from gpt_memory import GPTMemoryManager
from code_database import PatchHistoryDB


class PromptMemoryTrainer:
    """Analyse historical prompts and patch outcomes to learn formatting.

    The trainer reads prompts from :class:`GPTMemoryManager` and correlates
    them with patch outcomes stored in :class:`PatchHistoryDB`.  Simple regex
    heuristics extract style features (headers, example order and tone) from
    each prompt.  Success rates are aggregated per style and exposed as
    weighted metrics.
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
        """Return formatting features for ``prompt``."""

        headers = re.findall(r"^#+\s*(.+)$", prompt, flags=re.MULTILINE)
        example_order = re.findall(r"Example\s*([\w-]+)", prompt, flags=re.IGNORECASE)
        tone = "polite" if re.search(r"\b(?:please|kindly)\b", prompt, re.IGNORECASE) else "direct"
        return {"headers": headers, "example_order": example_order, "tone": tone}

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Dict[str, float]]:
        """Compute success rates for observed prompt styles."""

        stats: Dict[str, Dict[str, Mapping[str, int]]] = {
            "headers": defaultdict(lambda: {"success": 0, "count": 0}),
            "example_order": defaultdict(lambda: {"success": 0, "count": 0}),
            "tone": defaultdict(lambda: {"success": 0, "count": 0}),
        }

        cur = self.memory.conn.execute("SELECT prompt FROM interactions")
        for (text,) in cur.fetchall():
            match = re.search(r"PATCH:(\d+)", text)
            if not match:
                continue
            patch_id = int(match.group(1))
            row = self.patch_db.conn.execute(
                "SELECT outcome FROM patch_history WHERE id=?", (patch_id,)
            ).fetchone()
            if not row:
                continue
            outcome = row[0] or ""
            success = outcome.upper() == "SUCCESS"
            feats = self._extract_style(text)
            for key, val in feats.items():
                val_key = json.dumps(val) if isinstance(val, list) else str(val)
                d = stats[key][val_key]
                d["count"] += 1
                if success:
                    d["success"] += 1

        self.style_weights = {
            feat: {k: (v["success"] / v["count"]) if v["count"] else 0.0 for k, v in m.items()}
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
