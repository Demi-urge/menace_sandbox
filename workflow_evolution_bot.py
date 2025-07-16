from __future__ import annotations

"""Analyse pathway data to propose new workflow sequences."""

from dataclasses import dataclass
from typing import Iterable, List

from .neuroplasticity import PathwayDB


@dataclass
class WorkflowSuggestion:
    sequence: str
    expected_roi: float


class WorkflowEvolutionBot:
    """Suggest workflow improvements from PathwayDB statistics."""

    def __init__(self, pathway_db: PathwayDB | None = None) -> None:
        self.db = pathway_db or PathwayDB()

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

    def propose_rearrangements(self, limit: int = 5) -> Iterable[str]:
        """Return alternative sequences with reversed order."""
        for suggestion in self.analyse(limit):
            parts = suggestion.sequence.split("-")
            yield "-".join(reversed(parts))


__all__ = ["WorkflowEvolutionBot", "WorkflowSuggestion"]
