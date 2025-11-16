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
import statistics
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - used when PyYAML isn't installed
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency for style checks
    import pycodestyle  # type: ignore
except Exception:  # pragma: no cover - pycodestyle may be missing
    pycodestyle = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover - radon may be missing
    cc_visit = None  # type: ignore

try:  # pragma: no cover - allow package/flat imports
    from .code_database import CodeDB, PatchHistoryDB
except Exception:  # pragma: no cover - fallback for flat layout
    from code_database import CodeDB, PatchHistoryDB  # type: ignore

try:  # pragma: no cover - allow patch suggestion imports
    from .patch_suggestion_db import PatchSuggestionDB, SuggestionRecord
except Exception:  # pragma: no cover - fallback for flat layout
    from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class EnhancementSuggestion:
    """Outcome of a repository scan for potential improvements."""

    path: str
    score: float
    rationale: str
    raroi: float = 0.0


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
        self.weights, self.thresholds = self._load_weights(config_path)

    # ------------------------------------------------------------------
    def _load_weights(
        self, config_path: str | None
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Load metric weights and scan thresholds from a configuration file."""

        path = config_path or os.getenv(
            "ENHANCEMENT_CLASSIFIER_CONFIG", "enhancement_classifier_config.json"
        )
        weights = {
            "frequency": 1.0,
            "roi": 1.0,
            "raroi": 1.0,
            "errors": 1.0,
            "complexity": 1.0,
            "cyclomatic": 1.0,
            "duplication": 1.0,
            "length": 1.0,
            "anti": 1.0,
            "history": 1.0,
            "churn": 1.0,
            "style": 1.0,
            "refactor": 1.0,
            "anti_log": 1.0,
        }
        thresholds = {
            "min_patches": 3.0,
            "roi_cutoff": 0.0,
            "complexity_delta": 0.0,
            "churn": 0.0,
            "style": 0.0,
            "refactor": 0.0,
            "anti_pattern": 0.0,
        }
        try:
            text = Path(path).read_text()
            if path.endswith((".yaml", ".yml")) and yaml is not None:
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            weight_data = data.get("weights", data)
            for key in weights:
                if key in weight_data:
                    weights[key] = float(weight_data[key])
            threshold_data = data.get("thresholds", {})
            for key in thresholds:
                if key in threshold_data:
                    thresholds[key] = float(threshold_data[key])
        except Exception:  # pragma: no cover - fall back to defaults
            logger.debug(
                "using default enhancement classifier weights and thresholds",
                exc_info=True,
            )
        for key in weights:
            env = os.getenv(f"ENHANCEMENT_WEIGHT_{key.upper()}")
            if env is not None:
                try:
                    weights[key] = float(env)
                except ValueError:  # pragma: no cover - ignore bad overrides
                    pass
        for key in thresholds:
            env = os.getenv(f"ENHANCEMENT_THRESHOLD_{key.upper()}")
            if env is not None:
                try:
                    thresholds[key] = float(env)
                except ValueError:  # pragma: no cover - ignore bad overrides
                    pass
        return weights, thresholds

    # ------------------------------------------------------------------
    def _gather_metrics(
        self, code_id: int
    ) -> tuple[
        str,
        int,
        float,
        float,
        float,
        float,
        float,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        list[str],
        float,
        float,
    ] | None:
        """Return metrics for ``code_id`` or ``None`` when absent."""

        with self.patch_db._connect() as conn:  # type: ignore[attr-defined]
            try:
                rows = conn.execute(
                    """
                    SELECT filename, roi_delta, errors_before, errors_after, complexity_delta, description
                    FROM patch_history WHERE code_id=?
                    """,
                    (code_id,),
                ).fetchall()
            except Exception:
                base_rows = conn.execute(
                    """
                    SELECT filename, roi_delta, errors_before, errors_after, complexity_delta
                    FROM patch_history WHERE code_id=?
                    """,
                    (code_id,),
                ).fetchall()
                rows = [r + (None,) for r in base_rows]
        if not rows:
            return None
        filename = rows[0][0] or f"id:{code_id}"
        patch_count = len(rows)
        roi_deltas = [r[1] or 0.0 for r in rows]
        error_deltas = [(r[3] or 0) - (r[2] or 0) for r in rows]
        avg_roi = sum(roi_deltas) / patch_count
        avg_errors = sum(error_deltas) / patch_count
        roi_volatility = (
            statistics.pstdev(roi_deltas) if len(roi_deltas) > 1 else 0.0
        )
        raroi = avg_roi / (1.0 + roi_volatility + max(0.0, avg_errors))
        avg_complexity = sum((r[4] or 0.0) for r in rows) / patch_count
        neg_roi_ratio = sum(1 for r in rows if (r[1] or 0.0) < 0) / patch_count
        error_prone_ratio = sum(
            1 for r in rows if (r[3] or 0) > (r[2] or 0)
        ) / patch_count

        avg_cc = 0.0
        dup_count = 0
        total_funcs = 0
        long_funcs = 0
        notes: list[str] = []
        style_violations = 0

        try:
            with self.code_db._connect() as conn:  # type: ignore[attr-defined]
                row = conn.execute("SELECT code FROM code WHERE id=?", (code_id,)).fetchone()
        except Exception:  # pragma: no cover - tolerant to schema mismatch
            row = None

        source = row[0] if row and row[0] else None
        if source:
            tree: ast.AST | None = None
            if cc_visit is not None:
                try:
                    blocks = cc_visit(source)  # type: ignore[call-arg]
                    if blocks:
                        avg_cc = sum(b.complexity for b in blocks) / len(blocks)
                except Exception:  # pragma: no cover - radon may fail on edge cases
                    tree = ast.parse(source)
            if avg_cc == 0.0:
                tree = tree or ast.parse(source)

                def _complexity(node: ast.AST) -> int:
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

                complexities = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexities.append(_complexity(node))
                if complexities:
                    avg_cc = sum(complexities) / len(complexities)

            tree = tree or ast.parse(source)
            hashes: dict[str, int] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    body_repr = ast.dump(
                        ast.Module(body=node.body, type_ignores=[]),
                        include_attributes=False,
                    )
                    hashes[body_repr] = hashes.get(body_repr, 0) + 1
                    end = getattr(node, "end_lineno", node.lineno) or node.lineno
                    length = end - node.lineno + 1
                    if length > 50:
                        long_funcs += 1
                        notes.append(
                            f"function {node.name} is long ({length} lines)"
                        )
            dup_count = sum(v - 1 for v in hashes.values() if v > 1)
            dup_ratio = dup_count / total_funcs if total_funcs else 0.0
            if dup_count:
                notes.append(f"{dup_count} duplicated function(s)")
            if pycodestyle is not None:
                try:
                    checker = pycodestyle.Checker(lines=source.splitlines())  # type: ignore[arg-type]
                    style_violations = checker.check_all()
                except Exception:
                    style_violations = 0
        else:
            dup_ratio = 0.0
            total_funcs = total_funcs or 1

        func_churn = patch_count / max(total_funcs, 1)

        refactor_count = sum(
            1 for r in rows if r[5] and "refactor" in str(r[5]).lower()
        )
        if refactor_count == 0:
            refactor_count = patch_count

        anti_pattern_hits = 0
        try:
            error_db_path = os.getenv("ERROR_DB_PATH", "errors.db")
            with sqlite3.connect(error_db_path) as econn:
                cur = econn.execute(
                    "SELECT COUNT(*) FROM errors WHERE category='anti-pattern' AND message LIKE ?",
                    (f"%{filename}%",),
                )
                anti_pattern_hits = int(cur.fetchone()[0])
        except Exception:
            anti_pattern_hits = 0

        if refactor_count:
            notes.append(f"{refactor_count} refactor(s)")
        if anti_pattern_hits:
            notes.append(f"{anti_pattern_hits} anti-pattern log(s)")

        return (
            filename,
            patch_count,
            avg_roi,
            avg_errors,
            avg_complexity,
            avg_cc,
            dup_ratio,
            long_funcs,
            neg_roi_ratio,
            error_prone_ratio,
            func_churn,
            style_violations,
            refactor_count,
            anti_pattern_hits,
            notes,
            roi_volatility,
            raroi,
        )

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
        issues: list[str] = []
        penalty = 0.0

        if cc_visit is not None:
            try:
                blocks = cc_visit(source)  # type: ignore[call-arg]
                for block in blocks:
                    if block.complexity > 10:
                        issues.append(
                            f"function {block.name} exhibits high cyclomatic complexity "
                            f"({block.complexity})"
                        )
                        penalty += block.complexity - 10
                return penalty, issues
            except Exception:  # pragma: no cover - radon may fail on malformed input
                issues = []
                penalty = 0.0

        try:
            tree = ast.parse(source)
        except Exception:  # pragma: no cover - invalid source
            return 0.0, []

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
        suggestion_db = PatchSuggestionDB(Path("suggestions.db").resolve())
        with self.code_db._connect() as conn:  # type: ignore[attr-defined]
            code_ids = [row[0] for row in conn.execute("SELECT id FROM code").fetchall()]

        for cid in code_ids:
            metrics = self._gather_metrics(cid)
            if not metrics:
                continue
            (
                filename,
                patches,
                avg_roi,
                avg_errors,
                avg_complexity,
                avg_cc,
                dup_ratio,
                long_funcs,
                neg_roi_ratio,
                error_prone_ratio,
                func_churn,
                style_violations,
                refactor_count,
                anti_pattern_hits,
                notes,
                roi_volatility,
                raroi,
            ) = metrics
            anti_score, anti_notes = self._detect_anti_patterns(cid)
            anti_score += anti_pattern_hits
            if anti_pattern_hits:
                anti_notes.append("anti-pattern logged")

            # Heuristics: flag if any inefficiency indicator is observed
            if not (
                patches >= self.thresholds["min_patches"]
                or avg_roi < self.thresholds["roi_cutoff"]
                or avg_complexity > self.thresholds["complexity_delta"]
                or avg_cc > 0
                or dup_ratio > 0
                or long_funcs > 0
                or anti_score > 0
                or neg_roi_ratio > 0
                or error_prone_ratio > 0
                or func_churn > self.thresholds["churn"]
                or style_violations > self.thresholds["style"]
                or refactor_count > self.thresholds["refactor"]
                or anti_pattern_hits > self.thresholds["anti_pattern"]
            ):
                continue

            history_corr = (neg_roi_ratio + error_prone_ratio) * (avg_cc + dup_ratio)
            score = (
                self.weights["frequency"] * patches
                + self.weights["roi"] * (-avg_roi)
                + self.weights["raroi"] * (-raroi)
                + self.weights["errors"] * avg_errors
                + self.weights["complexity"] * avg_complexity
                + self.weights["cyclomatic"] * avg_cc
                + self.weights["duplication"] * dup_ratio
                + self.weights["length"] * long_funcs
                + self.weights["anti"] * anti_score
                + self.weights["history"] * history_corr
                + self.weights["churn"] * func_churn
                + self.weights["style"] * style_violations
                + self.weights["refactor"] * refactor_count
                + self.weights["anti_log"] * anti_pattern_hits
            )

            roi_phrase = "dropped" if avg_roi < 0 else "improved"
            if avg_errors > 0:
                err_phrase = f"errors increased by {avg_errors:.2f}"
            elif avg_errors < 0:
                err_phrase = f"errors decreased by {abs(avg_errors):.2f}"
            else:
                err_phrase = "errors unchanged"
            rationale = (
                f"module {filename} refactored {patches} times; "
                f"ROI {roi_phrase} {abs(avg_roi):.2f}%; "
                f"{err_phrase}; "
                f"RAROI {raroi:.2f} (vol {roi_volatility:.2f})"
            )
            triggers: list[str] = []
            if refactor_count > self.thresholds["refactor"]:
                triggers.append(f"{refactor_count} refactors")
            if avg_complexity > self.thresholds["complexity_delta"]:
                triggers.append(f"complexity +{avg_complexity:.2f}%")
            if func_churn > self.thresholds["churn"]:
                triggers.append(f"churn {func_churn:.2f}")
            if style_violations > self.thresholds["style"]:
                triggers.append(f"{style_violations} style violations")
            if anti_pattern_hits > self.thresholds["anti_pattern"]:
                triggers.append(f"{anti_pattern_hits} anti-pattern errors")
            all_notes = notes + anti_notes
            if triggers:
                rationale += "; triggers: " + ", ".join(triggers)
            if all_notes:
                rationale += "; " + "; ".join(all_notes)

            suggestion = EnhancementSuggestion(
                path=filename, score=score, rationale=rationale, raroi=raroi
            )
            try:
                rec = SuggestionRecord(
                    module=filename,
                    description=rationale,
                    score=score,
                    rationale=rationale,
                    patch_count=patches,
                    module_id=str(cid),
                    raroi=raroi,
                )
                suggestion_db.add(rec)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed recording suggestion for %s", filename)
            yield suggestion


__all__ = ["EnhancementClassifier", "EnhancementSuggestion"]
