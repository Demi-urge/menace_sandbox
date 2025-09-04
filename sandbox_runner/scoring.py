from __future__ import annotations

"""Lightweight scoring façade for sandbox runs.

The module aggregates per-module ROI deltas, entropy deltas, coverage
percentages and pass/fail counts. Results can be queried or persisted for
later analysis.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
import json


@dataclass
class ModuleScore:
    """Accumulated metrics for a single module."""

    roi_delta: float = 0.0
    entropy_delta: float = 0.0
    coverage: float = 0.0
    passes: int = 0
    failures: int = 0


class ScoringFacade:
    """Aggregate and persist module scores."""

    def __init__(self) -> None:
        self._scores: Dict[str, ModuleScore] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        name: str,
        *,
        roi_delta: float = 0.0,
        entropy_delta: float = 0.0,
        coverage: float = 0.0,
        success: bool = True,
    ) -> None:
        entry = self._scores.setdefault(name, ModuleScore())
        entry.roi_delta += roi_delta
        entry.entropy_delta += entropy_delta
        entry.coverage = max(entry.coverage, coverage)
        if success:
            entry.passes += 1
        else:
            entry.failures += 1

    # ------------------------------------------------------------------
    def scores(self) -> Dict[str, ModuleScore]:
        """Return a mapping of module names to their accumulated scores."""

        return self._scores

    # ------------------------------------------------------------------
    def persist(self, path: Path) -> None:
        """Write the current scores to ``path`` as JSON."""

        data = {name: asdict(score) for name, score in self._scores.items()}
        path.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "ScoringFacade":
        """Load scores from ``path`` and return a facade instance."""

        facade = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for name, vals in data.items():
                    facade._scores[name] = ModuleScore(**vals)
            except Exception:
                pass
        return facade


__all__ = ["ModuleScore", "ScoringFacade"]
