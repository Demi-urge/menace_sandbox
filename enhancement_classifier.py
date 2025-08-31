"""Analyse patch history to suggest potential code enhancements.

This module scans the repository using ``PatchHistoryDB`` and ``CodeDB`` and
looks for modules that exhibit signs of inefficiency.  It evaluates simple
heuristics (high refactor frequency, negative ROI deltas and complexity
spikes) and assigns a weighted score based on metrics defined in a YAML/JSON
configuration file.  Results are returned as :class:`EnhancementSuggestion`
objects.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - used when PyYAML isn't installed
    yaml = None  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .code_database import CodeDB, PatchHistoryDB
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import CodeDB, PatchHistoryDB  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class EnhancementSuggestion:
    """Outcome of a repository scan for potential improvements."""

    path: str
    score: float
    rationale: str


# ---------------------------------------------------------------------------
class EnhancementClassifier:
    """Detect modules likely to benefit from further enhancement."""

    def __init__(
        self,
        *,
        code_db: Optional[CodeDB] = None,
        patch_db: Optional[PatchHistoryDB] = None,
        config_path: str | None = None,
    ) -> None:
        self.code_db = code_db or CodeDB()
        self.patch_db = patch_db or PatchHistoryDB()
        self.weights = self._load_weights(config_path)

    # ------------------------------------------------------------------
    def _load_weights(self, config_path: str | None) -> dict[str, float]:
        """Load metric weights from a JSON or YAML configuration file."""

        path = config_path or os.getenv(
            "ENHANCEMENT_CLASSIFIER_CONFIG", "enhancement_classifier_config.json"
        )
        weights = {"frequency": 1.0, "roi": 1.0, "errors": 1.0, "anti": 1.0}
        try:
            text = Path(path).read_text()
            if path.endswith((".yaml", ".yml")) and yaml is not None:
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            data = data.get("weights", data)
            for key in weights:
                if key in data:
                    weights[key] = float(data[key])
        except Exception:  # pragma: no cover - fall back to defaults
            logger.debug("using default enhancement classifier weights", exc_info=True)
        return weights

    # ------------------------------------------------------------------
    def _gather_metrics(
        self, code_id: int
    ) -> tuple[str, int, float, float, float] | None:
        """Return metrics for ``code_id`` or ``None`` when absent."""

        with self.patch_db._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                """
                SELECT filename, roi_delta, errors_before, errors_after, complexity_delta
                FROM patch_history WHERE code_id=?
                """,
                (code_id,),
            ).fetchall()
        if not rows:
            return None
        filename = rows[0][0] or f"id:{code_id}"
        patch_count = len(rows)
        avg_roi = sum((r[1] or 0.0) for r in rows) / patch_count
        avg_errors = sum(((r[3] or 0) - (r[2] or 0)) for r in rows) / patch_count
        avg_complexity = sum((r[4] or 0.0) for r in rows) / patch_count
        return filename, patch_count, avg_roi, avg_errors, avg_complexity

    # ------------------------------------------------------------------
    def _detect_anti_patterns(self, code_id: int) -> tuple[float, list[str]]:
        """Analyse source code for basic anti-patterns.

        Returns a tuple of ``(score_penalty, notes)`` where ``score_penalty`` is a
        positive number added to the overall score when issues are found and
        ``notes`` contains human-readable descriptions of the problems.
        """

        try:
            with self.code_db._connect() as conn:  # type: ignore[attr-defined]
                row = conn.execute(
                    "SELECT code FROM code WHERE id=?", (code_id,)
                ).fetchone()
        except Exception:  # pragma: no cover - tolerant to schema mismatch
            logger.debug("anti-pattern scan skipped", exc_info=True)
            return 0.0, []
        if not row or not row[0]:
            return 0.0, []
        source = row[0]
        try:
            tree = ast.parse(source)
        except Exception:  # pragma: no cover - invalid source
            return 0.0, []

        issues: list[str] = []
        penalty = 0.0

        def complexity(node: ast.AST) -> int:
            count = 1
            for child in ast.walk(node):
                if isinstance(
                    child,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.Try,
                        ast.With,
                        ast.IfExp,
                        ast.ExceptHandler,
                        ast.BoolOp,
                    ),
                ):
                    if isinstance(child, ast.BoolOp):
                        count += max(0, len(child.values) - 1)
                    else:
                        count += 1
            return count

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cplx = complexity(node)
                if cplx > 10:
                    issues.append(
                        f"function {node.name} exhibits high cyclomatic complexity ({cplx})"
                    )
                    penalty += cplx - 10

        return penalty, issues

    # ------------------------------------------------------------------
    def scan_repo(self) -> Iterator[EnhancementSuggestion]:
        """Yield scored suggestions for modules that warrant attention."""

        with self.code_db._connect() as conn:  # type: ignore[attr-defined]
            code_ids = [row[0] for row in conn.execute("SELECT id FROM code").fetchall()]

        for cid in code_ids:
            metrics = self._gather_metrics(cid)
            if not metrics:
                continue
            filename, patches, avg_roi, avg_errors, avg_complexity = metrics
            anti_score, anti_notes = self._detect_anti_patterns(cid)

            # Heuristics: flag if any inefficiency indicator is observed
            if not (
                patches >= 3 or avg_roi < 0 or avg_complexity > 0 or anti_score > 0
            ):
                continue

            score = (
                self.weights["frequency"] * patches
                + self.weights["roi"] * (-avg_roi)
                + self.weights["errors"] * avg_errors
                + self.weights["anti"] * anti_score
            )

            rationale = (
                f"{patches} patches, avg ROI delta {avg_roi:.2f}, "
                f"avg error delta {avg_errors:.2f}, avg complexity delta {avg_complexity:.2f}"
            )
            if anti_notes:
                rationale += "; " + "; ".join(anti_notes)

            yield EnhancementSuggestion(path=filename, score=score, rationale=rationale)


__all__ = ["EnhancementClassifier", "EnhancementSuggestion"]

