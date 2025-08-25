from __future__ import annotations

"""Analyse pathway data to propose new workflow sequences."""

from dataclasses import dataclass
from typing import Dict, Iterable, List
import logging

from .neuroplasticity import PathwayDB
from . import mutation_logger as MutationLogger
try:  # pragma: no cover - allow flat imports
    from .intent_clusterer import IntentClusterer
    from .universal_retriever import UniversalRetriever
except Exception:  # pragma: no cover - fallback for flat layout
    from intent_clusterer import IntentClusterer  # type: ignore
    from universal_retriever import UniversalRetriever  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSuggestion:
    sequence: str
    expected_roi: float


class WorkflowEvolutionBot:
    """Suggest workflow improvements from PathwayDB statistics."""

    def __init__(
        self,
        pathway_db: PathwayDB | None = None,
        intent_clusterer: IntentClusterer | None = None,
    ) -> None:
        self.db = pathway_db or PathwayDB()
        self.intent_clusterer = intent_clusterer or IntentClusterer(UniversalRetriever())
        # Track mutation events for rearranged sequences so benchmarking
        # results can be fed back once available.
        self._rearranged_events: Dict[str, int] = {}

    def analyse(self, limit: int = 5) -> List[WorkflowSuggestion]:
        seqs = self.db.top_sequences(3, limit=limit)
        suggestions: List[WorkflowSuggestion] = []
        for seq, _weight in seqs:
            ids = [int(p) for p in seq.split("-") if p.isdigit()]
            rois: List[float] = []
            for pid in ids:
                row = self.db.conn.execute(
                    "SELECT avg_roi FROM metadata WHERE pathway_id=?", (pid,)
                ).fetchone()
                rois.append(float(row[0] or 0.0) if row else 0.0)
            expected = sum(rois) / len(rois) if rois else 0.0
            suggestions.append(WorkflowSuggestion(sequence=seq, expected_roi=expected))
        return suggestions

    def propose_rearrangements(
        self,
        limit: int = 5,
        *,
        workflow_id: int = 0,
        parent_event_id: int | None = None,
    ) -> Iterable[str]:
        """Yield rearranged sequences and log mutation events.

        Each rearranged sequence is recorded via :mod:`mutation_logger` with a
        placeholder performance value.  The returned strings can later be
        associated with benchmarking results via :meth:`record_benchmark`.
        """
        for suggestion in self.analyse(limit):
            parts = suggestion.sequence.split("-")
            seq = "-".join(reversed(parts))
            if self.intent_clusterer:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(seq)
                    paths = [
                        m.get("path")
                        for m in matches
                        if isinstance(m, dict) and m.get("path")
                    ]
                    clusters = [
                        m.get("cluster_id")
                        for m in matches
                        if isinstance(m, dict) and m.get("cluster_id") is not None
                    ]
                    if paths:
                        logger.info("intent matches for %s: %s", seq, paths)
                    if clusters:
                        logger.info("intent clusters for %s: %s", seq, clusters)
                except Exception as exc:
                    logger.error("intent cluster search failed: %s", exc)
            yield seq
            event_id = MutationLogger.log_mutation(
                change=seq,
                reason="rearrangement",
                trigger="workflow_evolution_bot",
                performance=0.0,
                workflow_id=workflow_id,
                parent_id=parent_event_id,
            )
            self._rearranged_events[seq] = event_id

    def record_benchmark(
        self, sequence: str, *, after_metric: float, roi: float, performance: float
    ) -> None:
        """Update performance metrics for a previously proposed sequence."""
        event_id = self._rearranged_events.get(sequence)
        if event_id is not None:
            MutationLogger._history_db.record_outcome(
                event_id,
                after_metric=after_metric,
                roi=roi,
                performance=performance,
            )


__all__ = ["WorkflowEvolutionBot", "WorkflowSuggestion"]
