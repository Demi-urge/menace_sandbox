"""CLI to backfill enhancement scores for existing patches."""

from __future__ import annotations

import logging
from enhancement_score import EnhancementMetrics
from vector_service.patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from code_database import PatchHistoryDB


def backfill() -> None:
    pdb = PatchHistoryDB()
    vm = VectorMetricsDB()
    logger = PatchLogger(patch_db=pdb, vector_metrics=vm)
    conn = pdb.router.get_connection("patch_history")
    cur = conn.execute(
        "SELECT id, lines_changed, context_tokens, time_to_completion, "
        "tests_passed, error_trace_count, effort_estimate, roi_tag FROM patch_history"
    )
    for (
        patch_id,
        lines_changed,
        context_tokens,
        time_to_completion,
        tests_passed,
        error_trace_count,
        effort_estimate,
        roi_tag,
    ) in cur.fetchall():
        metrics = EnhancementMetrics(
            lines_changed=lines_changed or 0,
            context_tokens=context_tokens or 0,
            time_to_completion=time_to_completion or 0.0,
            tests_passed=1 if tests_passed else 0,
            tests_failed=0 if tests_passed else 1 if tests_passed is not None else 0,
            error_traces=error_trace_count or 0,
            effort_estimate=effort_estimate or 0.0,
        )
        try:
            logger.recompute_enhancement_score(
                patch_id,
                metrics,
                roi_tag=roi_tag,
                error_trace_count=error_trace_count,
            )
        except Exception:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).exception(
                "failed to backfill enhancement score for patch %s", patch_id
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill()
