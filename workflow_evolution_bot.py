"""Analyse pathway data to propose new workflow sequences."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass
from typing import Dict, Iterable, List
import logging

from .neuroplasticity import PathwayDB
from . import mutation_logger as MutationLogger
from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator
try:  # pragma: no cover - allow flat imports
    from .intent_clusterer import IntentClusterer
    from .universal_retriever import UniversalRetriever
except Exception:  # pragma: no cover - fallback for flat layout
    from intent_clusterer import IntentClusterer  # type: ignore
    from universal_retriever import UniversalRetriever  # type: ignore

try:  # Optional helpers for variant generation
    from .module_synergy_grapher import get_synergy_cluster
except Exception:  # pragma: no cover - flat layout fallback
    try:
        from module_synergy_grapher import get_synergy_cluster  # type: ignore
    except Exception:  # pragma: no cover - best effort
        get_synergy_cluster = None  # type: ignore[misc]

try:  # ``WorkflowGraph`` may not always be available
    from .workflow_graph import WorkflowGraph
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from workflow_graph import WorkflowGraph  # type: ignore
    except Exception:  # pragma: no cover - best effort
        WorkflowGraph = None  # type: ignore[misc]

try:  # ``ModuleIOAnalyzer`` for structural validation
    from .workflow_synthesizer import ModuleIOAnalyzer
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from workflow_synthesizer import ModuleIOAnalyzer  # type: ignore
    except Exception:  # pragma: no cover - best effort
        ModuleIOAnalyzer = None  # type: ignore[misc]

try:  # ``analysis.get_io_signature`` for lightweight compatibility checks
    from analysis import get_io_signature
except Exception:  # pragma: no cover - fallback for flat layout
    try:
        from .analysis import get_io_signature  # type: ignore
    except Exception:  # pragma: no cover - best effort
        get_io_signature = None  # type: ignore[misc]

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = ContextBuilder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator: EvolutionOrchestrator | None = None
_th = get_thresholds("WorkflowEvolutionBot")
persist_sc_thresholds(
    "WorkflowEvolutionBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
)
manager = internalize_coding_bot(
    "WorkflowEvolutionBot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=evolution_orchestrator,
    threshold_service=ThresholdService(),
    roi_threshold=_th.roi_drop,
    error_threshold=_th.error_increase,
)


@dataclass
class WorkflowSuggestion:
    sequence: str
    expected_roi: float


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class WorkflowEvolutionBot:
    """Suggest workflow improvements from PathwayDB statistics."""

    def __init__(
        self,
        *,
        pathway_db: PathwayDB | None = None,
        intent_clusterer: IntentClusterer | None = None,
        manager: SelfCodingManager | None = None,
    ) -> None:
        self.db = pathway_db or PathwayDB()
        self.intent_clusterer = intent_clusterer or IntentClusterer(UniversalRetriever())
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot
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

    # ------------------------------------------------------------------
    def _validate_sequence(self, modules: List[str]) -> bool:
        """Return ``True`` if ``modules`` appears structurally sound."""

        # Use ``WorkflowGraph`` if available to avoid violating dependencies.
        if WorkflowGraph is not None:
            try:
                graph_obj = WorkflowGraph()
                graph = getattr(graph_obj, "graph", None)
                if graph is not None:
                    for a, b in zip(modules, modules[1:]):
                        if hasattr(graph, "has_edge") and graph.has_edge(str(b), str(a)):
                            return False
                        elif isinstance(graph, dict):
                            edges = graph.get("edges", {})
                            if str(b) in edges and str(a) in edges[str(b)]:
                                return False
            except Exception:
                pass

        if ModuleIOAnalyzer is not None:
            try:
                analyzer = ModuleIOAnalyzer()
                for a, b in zip(modules, modules[1:]):
                    sig_a = analyzer.analyze(a)
                    sig_b = analyzer.analyze(b)
                    if sig_a.outputs and sig_b.inputs and not (
                        set(sig_a.outputs) & set(sig_b.inputs)
                    ):
                        return False
            except Exception:
                pass
        elif get_io_signature is not None:
            try:
                for a, b in zip(modules, modules[1:]):
                    sig_a = get_io_signature(a)
                    sig_b = get_io_signature(b)
                    if sig_a.globals and sig_b.globals and not (
                        sig_a.globals & sig_b.globals
                    ):
                        return False
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    def generate_variants(
        self,
        limit: int = 5,
        *,
        workflow_id: int = 0,
        parent_event_id: int | None = None,
    ) -> Iterable[str]:
        """Yield variant workflow sequences and log mutation events.

        Variants are generated by swapping steps with synergistic modules and
        injecting intent-related modules.  Each proposed sequence is validated
        structurally and logged via :mod:`mutation_logger` for later
        benchmarking.
        """

        if self.manager and not self.manager.should_refactor():
            return

        emitted: set[str] = set()
        for suggestion in self.analyse(limit):
            parts = suggestion.sequence.split("-")

            # Basic reversed ordering as a starting variant
            reversed_seq = list(reversed(parts))
            if reversed_seq != parts and self._validate_sequence(reversed_seq):
                seq = "-".join(reversed_seq)
                if seq not in emitted:
                    emitted.add(seq)
                    yield seq
                    event_id = MutationLogger.log_mutation(
                        change=seq,
                        reason="variant",
                        trigger="workflow_evolution_bot",
                        performance=0.0,
                        workflow_id=workflow_id,
                        parent_id=parent_event_id,
                    )
                    self._rearranged_events[seq] = event_id

            # Swap each step with synergistic alternatives
            for idx, mod in enumerate(parts):
                cluster: set[str] = {mod}
                if get_synergy_cluster is not None:
                    try:
                        cluster = get_synergy_cluster(str(mod)) or {mod}
                    except Exception:
                        cluster = {mod}
                for cand in cluster:
                    if cand == mod:
                        continue
                    swapped = parts.copy()
                    swapped[idx] = str(cand)
                    if not self._validate_sequence(swapped):
                        continue
                    seq = "-".join(swapped)
                    if seq in emitted:
                        continue
                    emitted.add(seq)
                    yield seq
                    event_id = MutationLogger.log_mutation(
                        change=seq,
                        reason="variant",
                        trigger="workflow_evolution_bot",
                        performance=0.0,
                        workflow_id=workflow_id,
                        parent_id=parent_event_id,
                    )
                    self._rearranged_events[seq] = event_id

            # Inject intent-related modules at the end of the sequence
            if self.intent_clusterer:
                try:
                    matches = self.intent_clusterer.find_modules_related_to(
                        suggestion.sequence
                    )
                    for match in matches:
                        path = getattr(match, "path", None)
                        if not path:
                            continue
                        candidate = parts + [path]
                        if not self._validate_sequence(candidate):
                            continue
                        seq = "-".join(candidate)
                        if seq in emitted:
                            continue
                        emitted.add(seq)
                        yield seq
                        event_id = MutationLogger.log_mutation(
                            change=seq,
                            reason="variant",
                            trigger="workflow_evolution_bot",
                            performance=0.0,
                            workflow_id=workflow_id,
                            parent_id=parent_event_id,
                        )
                        self._rearranged_events[seq] = event_id
                except Exception as exc:
                    logger.error("intent cluster search failed: %s", exc)

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


def generate_variants(
    workflow: Iterable[str] | str,
    n: int,
    *,
    workflow_id: int = 0,
    parent_event_id: int | None = None,
) -> List[str]:
    """Return up to ``n`` valid workflow variants derived from ``workflow``.

    The function performs three mutation strategies:
    swapping modules with synergistic alternatives, generating step
    reorderings and injecting intent-related modules.  Each candidate is
    validated using :func:`analysis.get_io_signature` or the more advanced
    ``ModuleIOAnalyzer`` when available, and recorded via
    :func:`mutation_logger.log_mutation`.
    """

    bot = WorkflowEvolutionBot()
    parts = list(workflow.split("-") if isinstance(workflow, str) else workflow)
    emitted: set[str] = set()
    variants: List[str] = []

    def _emit(mods: List[str]) -> None:
        if len(variants) >= n:
            return
        if not bot._validate_sequence(mods):
            return
        seq = "-".join(str(m) for m in mods)
        if seq in emitted:
            return
        emitted.add(seq)
        MutationLogger.log_mutation(
            change=seq,
            reason="variant",
            trigger="workflow_evolution_bot",
            performance=0.0,
            workflow_id=workflow_id,
            parent_id=parent_event_id,
        )
        variants.append(seq)

    # ---- swap modules with synergy alternatives
    for idx, mod in enumerate(parts):
        cluster: set[str] = {str(mod)}
        if get_synergy_cluster is not None:
            try:
                cluster = get_synergy_cluster(str(mod)) or {str(mod)}
            except Exception:
                cluster = {str(mod)}
        for cand in cluster:
            if cand == mod:
                continue
            swapped = parts.copy()
            swapped[idx] = str(cand)
            _emit(swapped)
            if len(variants) >= n:
                return variants

    # ---- reorder steps (pairwise swaps)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            swapped = parts.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            _emit(swapped)
            if len(variants) >= n:
                return variants

    # ---- inject intent-related modules
    try:
        matches = bot.intent_clusterer.find_modules_related_to("-".join(parts))
    except Exception:
        matches = []
    for match in matches:
        path = getattr(match, "path", None)
        if not path:
            continue
        candidate = parts + [str(path)]
        _emit(candidate)
        if len(variants) >= n:
            return variants

    return variants


__all__ = ["WorkflowEvolutionBot", "WorkflowSuggestion", "generate_variants"]
