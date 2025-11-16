"""Analyse pathway data to propose new workflow sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Callable, TYPE_CHECKING, cast
import logging
from importlib import import_module
from pathlib import Path
import sys

_HAS_PACKAGE = bool(__package__)

logger = logging.getLogger(__name__)


def _flat_import(module: str) -> object:
    """Import ``module`` from the top-level package when executed flat."""

    package_root = Path(__file__).resolve().parent
    package_name = package_root.name
    qualified = f"{package_name}.{module}"
    try:
        return import_module(qualified)
    except ModuleNotFoundError:
        parent = str(package_root.parent)
        if parent not in sys.path:
            sys.path.append(parent)
        return import_module(qualified)

if _HAS_PACKAGE:
    from .coding_bot_interface import self_coding_managed
else:  # pragma: no cover - execution as a script
    self_coding_managed = _flat_import("coding_bot_interface").self_coding_managed  # type: ignore[attr-defined]

if _HAS_PACKAGE:
    from .neuroplasticity import PathwayDB
else:  # pragma: no cover - execution as a script
    PathwayDB = _flat_import("neuroplasticity").PathwayDB  # type: ignore[attr-defined]

try:  # pragma: no cover - prefer package relative import
    from . import mutation_logger as MutationLogger
except ImportError:  # pragma: no cover - allow execution as a script
    MutationLogger = _flat_import("mutation_logger")  # type: ignore[assignment]

if _HAS_PACKAGE:
    from .bot_registry import BotRegistry
else:  # pragma: no cover - execution as a script
    BotRegistry = _flat_import("bot_registry").BotRegistry  # type: ignore[attr-defined]

if _HAS_PACKAGE:
    from .data_bot import DataBot, persist_sc_thresholds
else:  # pragma: no cover - execution as a script
    _data_bot = _flat_import("data_bot")
    DataBot = _data_bot.DataBot  # type: ignore[attr-defined]
    persist_sc_thresholds = _data_bot.persist_sc_thresholds  # type: ignore[attr-defined]

if _HAS_PACKAGE:
    from .safe_repr import basic_repr
else:  # pragma: no cover - execution as a script
    basic_repr = _flat_import("safe_repr").basic_repr  # type: ignore[attr-defined]

try:
    if _HAS_PACKAGE:
        from .self_coding_manager import SelfCodingManager, internalize_coding_bot
    else:  # pragma: no cover - execution as a script
        _scm = _flat_import("self_coding_manager")
        SelfCodingManager = _scm.SelfCodingManager  # type: ignore[attr-defined]
        internalize_coding_bot = _scm.internalize_coding_bot  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning(
        "SelfCodingManager unavailable for WorkflowEvolutionBot: %s",
        exc,
    )

    class _FallbackManager:  # pragma: no cover - simplified stub
        engine: Any | None = None

    SelfCodingManager = _FallbackManager  # type: ignore[misc, assignment]

    def internalize_coding_bot(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        return None

try:
    if _HAS_PACKAGE:
        from .self_coding_engine import SelfCodingEngine
    else:  # pragma: no cover - execution as a script
        SelfCodingEngine = _flat_import("self_coding_engine").SelfCodingEngine  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning(
        "SelfCodingEngine unavailable for WorkflowEvolutionBot: %s",
        exc,
    )
    SelfCodingEngine = type("_FallbackEngine", (), {})  # type: ignore[misc, assignment]


def _load_model_automation_pipeline() -> type | None:
    """Return the concrete ``ModelAutomationPipeline`` class if available."""

    candidates: list[tuple[str, Callable[[], object]]] = []
    if _HAS_PACKAGE:
        dotted = f"{__package__}.model_automation_pipeline"
        candidates.append((dotted, lambda d=dotted: import_module(d)))
    candidates.extend(
        [
            (
                "menace_sandbox.model_automation_pipeline",
                lambda: import_module("menace_sandbox.model_automation_pipeline"),
            ),
            (
                "model_automation_pipeline",
                lambda: import_module("model_automation_pipeline"),
            ),
        ]
    )
    if not _HAS_PACKAGE:
        candidates.append(("flat:model_automation_pipeline", lambda: _flat_import("model_automation_pipeline")))

    for label, loader in candidates:
        try:
            module = loader()
        except ModuleNotFoundError:  # pragma: no cover - dependency missing in environment
            continue
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            logger.debug(
                "Deferred ModelAutomationPipeline import via %s failed: %s",
                label,
                exc,
                exc_info=True,
            )
            continue
        pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
        if isinstance(pipeline_cls, type):
            return pipeline_cls
    return None


if TYPE_CHECKING:  # pragma: no cover - typing only import
    from .model_automation_pipeline import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime alias for typing convenience
    _ModelAutomationPipeline = Any  # type: ignore[misc, assignment]


def _create_model_automation_pipeline(
    *, context_builder: Any
) -> _ModelAutomationPipeline | None:
    """Instantiate ``ModelAutomationPipeline`` while avoiding circular imports."""

    pipeline_cls = _load_model_automation_pipeline()
    if pipeline_cls is None:
        return None
    try:
        return cast("_ModelAutomationPipeline", pipeline_cls)(
            context_builder=context_builder
        )
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        logger.debug(
            "Failed to initialise ModelAutomationPipeline: %s",
            exc,
            exc_info=True,
        )
        return None

try:
    if _HAS_PACKAGE:
        from .threshold_service import ThresholdService
    else:  # pragma: no cover - execution as a script
        ThresholdService = _flat_import("threshold_service").ThresholdService  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning(
        "ThresholdService unavailable for WorkflowEvolutionBot: %s",
        exc,
    )

    class ThresholdService:  # type: ignore[no-redef]
        pass

try:
    if _HAS_PACKAGE:
        from .code_database import CodeDB
    else:  # pragma: no cover - execution as a script
        CodeDB = _flat_import("code_database").CodeDB  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning("CodeDB unavailable for WorkflowEvolutionBot: %s", exc)
    CodeDB = type("_FallbackCodeDB", (), {})  # type: ignore[misc, assignment]

try:
    if _HAS_PACKAGE:
        from .gpt_memory import GPTMemoryManager
    else:  # pragma: no cover - execution as a script
        GPTMemoryManager = _flat_import("gpt_memory").GPTMemoryManager  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning("GPTMemoryManager unavailable for WorkflowEvolutionBot: %s", exc)
    GPTMemoryManager = type("_FallbackGPTMemory", (), {})  # type: ignore[misc, assignment]

try:
    if _HAS_PACKAGE:
        from .self_coding_thresholds import get_thresholds
    else:  # pragma: no cover - execution as a script
        get_thresholds = _flat_import("self_coding_thresholds").get_thresholds  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning("get_thresholds unavailable for WorkflowEvolutionBot: %s", exc)

    def get_thresholds(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        raise RuntimeError("thresholds unavailable")

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder  # type: ignore
except Exception:  # pragma: no cover - allow running without builder
    ContextBuilder = None  # type: ignore[misc]

try:
    if _HAS_PACKAGE:
        from .shared_evolution_orchestrator import get_orchestrator
    else:  # pragma: no cover - execution as a script
        get_orchestrator = _flat_import("shared_evolution_orchestrator").get_orchestrator  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning(
        "get_orchestrator unavailable for WorkflowEvolutionBot: %s",
        exc,
    )

    def get_orchestrator(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[override]
        return None

if _HAS_PACKAGE:
    from .context_builder_util import create_context_builder
else:  # pragma: no cover - execution as a script
    create_context_builder = _flat_import("context_builder_util").create_context_builder  # type: ignore[attr-defined]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator

try:
    if _HAS_PACKAGE:
        from .intent_clusterer import IntentClusterer
    else:  # pragma: no cover - execution as a script
        IntentClusterer = _flat_import("intent_clusterer").IntentClusterer  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning("IntentClusterer unavailable for WorkflowEvolutionBot: %s", exc)

    class IntentClusterer:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def cluster(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

try:
    if _HAS_PACKAGE:
        from .universal_retriever import UniversalRetriever
    else:  # pragma: no cover - execution as a script
        UniversalRetriever = _flat_import("universal_retriever").UniversalRetriever  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - degraded bootstrap
    logger.warning(
        "UniversalRetriever unavailable for WorkflowEvolutionBot: %s",
        exc,
    )

    class UniversalRetriever:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def retrieve(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

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

_RUNTIME_CACHE: dict[str, Any] | None = None


def _ensure_runtime_dependencies() -> dict[str, Any]:
    """Instantiate expensive runtime dependencies on demand."""

    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    runtime: dict[str, Any] = {}

    try:
        registry_local = BotRegistry()
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        logger.warning("BotRegistry unavailable for WorkflowEvolutionBot: %s", exc)

        class _FallbackBotRegistry:
            modules: dict[str, str] = {}
            graph = type("_Graph", (), {"nodes": {}})()

            def register_bot(self, *args: Any, **kwargs: Any) -> None:
                return None

            def update_bot(self, *args: Any, **kwargs: Any) -> None:
                return None

            def hot_swap_active(self) -> bool:
                return False

        registry_local = _FallbackBotRegistry()  # type: ignore[assignment]

    runtime["registry"] = registry_local

    try:
        data_bot_local = DataBot(start_server=False)
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        logger.warning("DataBot unavailable for WorkflowEvolutionBot: %s", exc)

        class _FallbackDataBot:
            def reload_thresholds(self, *_args: Any, **_kwargs: Any) -> Any:
                return None

        data_bot_local = _FallbackDataBot()  # type: ignore[assignment]

    runtime["data_bot"] = data_bot_local

    try:
        context_builder = create_context_builder()
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        logger.warning(
            "Context builder unavailable for WorkflowEvolutionBot: %s",
            exc,
        )
        context_builder = None

    runtime["context_builder"] = context_builder

    if context_builder is not None:
        try:
            engine = SelfCodingEngine(
                CodeDB(),
                GPTMemoryManager(),
                context_builder=context_builder,
            )
        except Exception as exc:  # pragma: no cover - degraded bootstrap
            logger.warning(
                "SelfCodingEngine unavailable; WorkflowEvolutionBot will run without self-coding manager: %s",
                exc,
            )
            engine = None  # type: ignore[assignment]
    else:
        engine = None  # type: ignore[assignment]

    runtime["engine"] = engine

    if context_builder is not None and engine is not None:
        pipeline = _create_model_automation_pipeline(context_builder=context_builder)
        if pipeline is None:
            logger.warning(
                "ModelAutomationPipeline unavailable for WorkflowEvolutionBot",
            )
    else:
        pipeline = None

    runtime["pipeline"] = pipeline

    try:
        evolution_orchestrator = (
            get_orchestrator("WorkflowEvolutionBot", data_bot_local, engine)
            if engine is not None
            else None
        )
    except Exception as exc:  # pragma: no cover - orchestrator optional
        logger.warning(
            "EvolutionOrchestrator unavailable for WorkflowEvolutionBot: %s",
            exc,
        )
        evolution_orchestrator = None

    runtime["evolution_orchestrator"] = evolution_orchestrator

    try:
        thresholds = get_thresholds("WorkflowEvolutionBot")
    except Exception as exc:  # pragma: no cover - thresholds optional
        logger.warning("Threshold lookup failed for WorkflowEvolutionBot: %s", exc)
        thresholds = None

    runtime["thresholds"] = thresholds

    if thresholds is not None:
        try:
            persist_sc_thresholds(
                "WorkflowEvolutionBot",
                roi_drop=thresholds.roi_drop,
                error_increase=thresholds.error_increase,
                test_failure_increase=thresholds.test_failure_increase,
            )
        except Exception as exc:  # pragma: no cover - best effort persistence
            logger.warning("failed to persist WorkflowEvolutionBot thresholds: %s", exc)

    if engine is not None and pipeline is not None and thresholds is not None:
        try:
            manager_local = internalize_coding_bot(
                "WorkflowEvolutionBot",
                engine,
                pipeline,
                data_bot=data_bot_local,
                bot_registry=registry_local,
                evolution_orchestrator=evolution_orchestrator,
                threshold_service=ThresholdService(),
                roi_threshold=thresholds.roi_drop,
                error_threshold=thresholds.error_increase,
                test_failure_threshold=thresholds.test_failure_increase,
            )
        except Exception as exc:  # pragma: no cover - degraded bootstrap
            logger.warning(
                "failed to initialise self-coding manager for WorkflowEvolutionBot: %s",
                exc,
            )
            manager_local = None
    else:
        manager_local = None

    runtime["manager"] = manager_local

    _RUNTIME_CACHE = runtime
    return runtime


def _ensure_self_coding_bootstrap(cls: type) -> None:
    """Apply ``self_coding_managed`` lazily when runtime is needed."""

    if getattr(cls, "_self_coding_bootstrapped", False):
        return
    runtime = _ensure_runtime_dependencies()
    manager = runtime.get("manager")
    decorated = self_coding_managed(
        bot_registry=runtime["registry"],
        data_bot=runtime["data_bot"],
        manager=manager,
    )(cls)
    # ``self_coding_managed`` decorates the class in place, but return value is
    # used defensively to make intent explicit.
    assert decorated is cls
    cls._self_coding_bootstrapped = True


class _LazySelfCodingMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any):  # type: ignore[override]
        _ensure_self_coding_bootstrap(cls)
        return super().__call__(*args, **kwargs)


@dataclass
class WorkflowSuggestion:
    sequence: str
    expected_roi: float


class WorkflowEvolutionBot(metaclass=_LazySelfCodingMeta):
    """Suggest workflow improvements from PathwayDB statistics."""

    def __init__(
        self,
        *,
        pathway_db: PathwayDB | None = None,
        intent_clusterer: IntentClusterer | None = None,
        manager: SelfCodingManager | None = None,
    ) -> None:
        runtime: dict[str, Any] | None = None
        try:
            runtime = _ensure_runtime_dependencies()
        except Exception as exc:  # pragma: no cover - defensive bootstrap
            logger.debug(
                "WorkflowEvolutionBot runtime dependencies unavailable during init: %s",
                exc,
                exc_info=True,
            )

        self.db = pathway_db or PathwayDB()
        self.intent_clusterer = intent_clusterer or IntentClusterer(UniversalRetriever())
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = runtime.get("data_bot") if runtime is not None else None
        if manager is None and runtime is not None:
            manager = runtime.get("manager")
        if manager is not None:
            self.manager = manager
        # Track mutation events for rearranged sequences so benchmarking
        # results can be fed back once available.
        self._rearranged_events: Dict[str, int] = {}

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return basic_repr(
            self,
            attrs={
                "db": self.db,
                "intent_clusterer": self.intent_clusterer,
                "manager": getattr(self, "manager", None),
            },
        )

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
