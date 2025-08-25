from __future__ import annotations

"""Lightweight workflow synthesizer utilities.

This module exposes :class:`WorkflowSynthesizer` which combines structural
signals from :class:`~module_synergy_grapher.ModuleSynergyGrapher` with
semantic intent matches provided by :class:`~intent_clusterer.IntentClusterer`.

The synthesizer is intentionally small and focuses on expanding an initial set
of modules either by following the synergy graph around a starting module or by
searching for modules related to a textual problem description.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

try:  # Optional imports; fall back to stubs in tests
    from module_synergy_grapher import ModuleSynergyGrapher
except Exception:  # pragma: no cover - graceful degradation
    ModuleSynergyGrapher = None  # type: ignore[misc]

try:  # Optional dependency
    from intent_clusterer import IntentClusterer
except Exception:  # pragma: no cover - graceful degradation
    IntentClusterer = None  # type: ignore[misc]


@dataclass
class WorkflowSynthesizer:
    """Suggest modules for building a workflow.

    Parameters
    ----------
    module_synergy_grapher:
        Helper used to expand modules via structural relationships.
    intent_clusterer:
        Component used to search modules based on natural language problems.
    synergy_graph_path:
        Location of the persisted synergy graph JSON file.
    intent_db_path:
        Optional location of the intent vector database.  The synthesizer does
        not interact with the database directly but exposes this path so that
        callers can initialise :class:`IntentClusterer` with it if desired.
    """

    module_synergy_grapher: ModuleSynergyGrapher | None = None
    intent_clusterer: IntentClusterer | None = None
    synergy_graph_path: Path = Path("sandbox_data/module_synergy_graph.json")
    intent_db_path: Path | None = None

    # ------------------------------------------------------------------
    def synthesize(
        self,
        start_module: str | None = None,
        problem: str | None = None,
        limit: int = 10,
    ) -> List[str]:
        """Return a list of module names relevant to ``start_module`` and ``problem``.

        The method first loads the synergy graph from ``synergy_graph_path`` and,
        if ``start_module`` is provided, expands the cluster around that module.
        When a textual ``problem`` is supplied, semantic matches from
        :class:`IntentClusterer` are merged with the synergy set.
        """

        modules: Set[str] = set()

        # ----- expand via synergy graph
        if start_module and self.module_synergy_grapher is not None:
            try:
                if hasattr(self.module_synergy_grapher, "load"):
                    # Ensure the grapher has the latest graph loaded
                    self.module_synergy_grapher.load(self.synergy_graph_path)
                cluster = self.module_synergy_grapher.get_synergy_cluster(start_module)
                modules.update(cluster)
            except Exception:  # pragma: no cover - best effort
                modules.add(start_module)

        # ----- expand via intent search
        if problem and self.intent_clusterer is not None:
            try:
                matches = self.intent_clusterer.find_modules_related_to(
                    problem, top_k=limit
                )
                for match in matches:
                    path = getattr(match, "path", None)
                    if path:
                        modules.add(Path(path).stem)
            except Exception:  # pragma: no cover - ignore search failures
                pass

        return sorted(modules)[:limit]
