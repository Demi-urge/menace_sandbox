from __future__ import annotations

import logging
from typing import Iterator, List, Tuple

try:  # pragma: no cover - allow package/flat imports
    from .code_database import CodeDB, PatchHistoryDB
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import CodeDB, PatchHistoryDB  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .chatgpt_enhancement_bot import EnhancementDB
except Exception:  # pragma: no cover - fallback for flat layout
    from chatgpt_enhancement_bot import EnhancementDB  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .patch_suggestion_db import SuggestionRecord
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import SuggestionRecord  # type: ignore

logger = logging.getLogger(__name__)


class EnhancementClassifier:
    """Classify modules that may benefit from further enhancement.

    The classifier inspects historical patch and enhancement metadata to flag
    modules that show signs of inefficiency such as repeated refactors with
    low or negative ROI or steadily increasing complexity.
    """

    def __init__(
        self,
        *,
        code_db: CodeDB | None = None,
        patch_db: PatchHistoryDB | None = None,
        enhancement_db: EnhancementDB | None = None,
    ) -> None:
        self.code_db = code_db or CodeDB()
        self.patch_db = patch_db or PatchHistoryDB()
        self.enhancement_db = enhancement_db or EnhancementDB()

    # ------------------------------------------------------------------
    def _module_stats(
        self, code_id: int
    ) -> tuple[str, float, float, float, int] | None:
        """Return (filename, avg_roi, avg_complexity_delta, avg_outcome, count).

        ``None`` is returned when no patch history exists for ``code_id``.
        """

        with self.patch_db._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                "SELECT filename, roi_delta, complexity_delta "
                "FROM patch_history WHERE code_id=?",
                (code_id,),
            ).fetchall()
        if not rows:
            return None
        filename = rows[0][0] or f"id:{code_id}"
        patch_count = len(rows)
        avg_roi = sum((r[1] or 0.0) for r in rows) / patch_count
        avg_complexity_delta = sum((r[2] or 0.0) for r in rows) / patch_count

        # Enhancement metadata lookup
        with self.code_db._connect() as conn:  # type: ignore[attr-defined]
            enh_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT enhancement_id FROM code_enhancements WHERE code_id=?",
                    (code_id,),
                ).fetchall()
            ]
        outcome_scores: List[float] = []
        if enh_ids:
            e_conn = self.enhancement_db._connect()
            for eid in enh_ids:
                row = e_conn.execute(
                    "SELECT outcome_score FROM enhancements WHERE id=?",
                    (eid,),
                ).fetchone()
                if row:
                    outcome_scores.append(float(row[0] or 0.0))
        avg_outcome = sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0
        return filename, avg_roi, avg_complexity_delta, avg_outcome, patch_count

    # ------------------------------------------------------------------
    def scan_repo(self) -> Iterator[SuggestionRecord]:
        """Yield :class:`SuggestionRecord` instances for suspicious modules."""

        with self.code_db._connect() as conn:  # type: ignore[attr-defined]
            code_ids = [row[0] for row in conn.execute("SELECT id FROM code").fetchall()]

        for cid in code_ids:
            stats = self._module_stats(cid)
            if not stats:
                continue
            filename, avg_roi, avg_comp, avg_outcome, patch_count = stats
            # score: frequent patches or complexity growth penalised by ROI gains
            score = patch_count + max(0.0, avg_comp) - avg_roi - avg_outcome
            if score <= 0:
                continue
            rationale = (
                f"{patch_count} patches, avg ROI delta {avg_roi:.2f}, "
                f"avg complexity delta {avg_comp:.2f}, "
                f"avg enhancement outcome {avg_outcome:.2f}"
            )
            description = f"score={score:.2f} - {rationale}"
            logger.debug("suggestion", extra={"module": filename, "score": score})
            yield SuggestionRecord(module=filename, description=description)


__all__ = ["EnhancementClassifier"]
