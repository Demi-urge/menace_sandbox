from __future__ import annotations

"""Automatically propose fixes for recurring errors."""

from pathlib import Path
import logging
import subprocess
import shutil
import tempfile
import json
from typing import Tuple, Iterable, Dict, Any
from gpt_memory_interface import GPTMemoryInterface

try:  # pragma: no cover - optional dependency
    from .error_cluster_predictor import ErrorClusterPredictor
except Exception:  # pragma: no cover - optional dependency
    ErrorClusterPredictor = object  # type: ignore

from .error_bot import ErrorDB
from .self_coding_manager import SelfCodingManager
from .knowledge_graph import KnowledgeGraph
from .context_builder import ContextBuilder
try:  # pragma: no cover - optional dependency
    from .human_alignment_flagger import _collect_diff_data
except Exception:  # pragma: no cover - fallback for tests
    def _collect_diff_data(*a, **k):
        return {}
try:  # pragma: no cover - optional dependency
    from .human_alignment_agent import HumanAlignmentAgent
except Exception:  # pragma: no cover - fallback for tests
    HumanAlignmentAgent = object  # type: ignore
try:  # pragma: no cover - optional dependency
    from .violation_logger import log_violation
except Exception:  # pragma: no cover - fallback for tests
    def log_violation(*a, **k):
        return None

try:  # pragma: no cover - optional dependency
    from .universal_retriever import UniversalRetriever
except Exception:  # pragma: no cover - missing dependency
    UniversalRetriever = None  # type: ignore

try:  # pragma: no cover - tag constants
    from .log_tags import ERROR_FIX
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import ERROR_FIX  # type: ignore

def generate_patch(
    module: str,
    engine: "SelfCodingEngine" | None = None,
    context_builder: ContextBuilder | None = None,
    memory_context: GPTMemoryInterface | None = None,
) -> int | None:
    """Attempt a quick patch for *module* and return the patch id.

    Parameters
    ----------
    module:
        Target module path or module name without ``.py``.
    engine:
        Optional :class:`~self_coding_engine.SelfCodingEngine` instance.  If not
        provided, a minimal engine is instantiated on demand.  The function
        tolerates missing dependencies and simply returns ``None`` on failure.
    """

    logger = logging.getLogger("QuickFixEngine")
    path = Path(module)
    if path.suffix == "":
        path = path.with_suffix(".py")
    if not path.exists():
        logger.error("module not found: %s", module)
        return None

    context_meta: Dict[str, Any] = {"module": str(path), "reason": "preemptive_fix"}
    builder = context_builder or ContextBuilder()
    description = f"preemptive fix for {path.name}"
    try:
        context_block = builder.build_context(description)
    except Exception:
        context_block = {}
    if context_block:
        description += "\n\n" + json.dumps(context_block, indent=2)

    if engine is None:
        try:  # pragma: no cover - heavy dependencies
            from .self_coding_engine import SelfCodingEngine
            from .code_database import CodeDB
            from .menace_memory_manager import MenaceMemoryManager

            engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager())
        except Exception as exc:  # pragma: no cover - optional deps
            logger.error("self coding engine unavailable: %s", exc)
            return None

    if memory_context is None and getattr(engine, "gpt_memory", None) is not None:
        try:
            base_mem = engine.gpt_memory  # type: ignore[attr-defined]
            tagged = base_mem.search_context("", tags=[ERROR_FIX])

            class _TaggedMemory:
                def __init__(self, base: GPTMemoryInterface, entries: list[Any]) -> None:
                    self.base = base
                    self.entries = entries

                def search_context(self, query: str, limit: int = 5):  # pragma: no cover - tiny wrapper
                    extra: list[Any]
                    try:
                        extra = self.base.search_context(query, limit=limit, tags=[ERROR_FIX])
                    except Exception:
                        extra = []
                    return (self.entries + extra)[:limit]

            memory_context = _TaggedMemory(base_mem, tagged)
        except Exception:
            memory_context = getattr(engine, "gpt_memory", None)

    try:
        patch_id: int | None
        with tempfile.TemporaryDirectory() as before_dir, tempfile.TemporaryDirectory() as after_dir:
            rel = path.name if path.is_absolute() else path
            before_target = Path(before_dir) / rel
            before_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, before_target)
            try:
                patch_id, _, _ = engine.apply_patch(
                    path,
                    description,
                    reason="preemptive_fix",
                    trigger="quick_fix_engine",
                    context_meta=context_meta,
                    memory_context=memory_context,
                )
            except AttributeError:
                engine.patch_file(
                    path, "preemptive_fix", context_meta=context_meta, memory_context=memory_context
                )
                patch_id = None
            after_target = Path(after_dir) / rel
            after_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, after_target)
            diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
            workflow_changes = [
                {"file": f, "code": "\n".join(d["added"])}
                for f, d in diff_data.items()
                if d["added"]
            ]
            if workflow_changes:
                agent = HumanAlignmentAgent()
                warnings = agent.evaluate_changes(workflow_changes, None, [])
                if any(warnings.values()):
                    log_violation(
                        str(patch_id) if patch_id is not None else str(path),
                        "alignment_warning",
                        1,
                        {"warnings": warnings},
                        alignment_warning=True,
                    )
            return patch_id
    except Exception as exc:  # pragma: no cover - runtime issues
        logger.error("quick fix generation failed for %s: %s", module, exc)
        return None


class QuickFixEngine:
    """Analyse frequent errors and trigger small patches."""

    def __init__(
        self,
        error_db: ErrorDB,
        manager: SelfCodingManager,
        *,
        threshold: int = 3,
        graph: KnowledgeGraph | None = None,
        risk_threshold: float = 0.5,
        predictor: ErrorClusterPredictor | None = None,
        retriever: "UniversalRetriever | None" = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.db = error_db
        self.manager = manager
        self.threshold = threshold
        self.graph = graph or KnowledgeGraph()
        self.risk_threshold = risk_threshold
        self.predictor = predictor
        self.retriever = retriever
        self.context_builder = context_builder
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _top_error(
        self, bot: str
    ) -> Tuple[str, str, dict[str, int], int] | None:
        try:
            info = self.db.top_error_module(bot)
        except Exception:
            return None
        if not info:
            return None
        etype, module, mods, count, _ = info
        return etype, module, mods, count

    def run(self, bot: str) -> None:
        """Attempt a quick patch for the most frequent error of ``bot``."""
        if self.predictor is not None:
            try:
                modules = self.predictor.predict_high_risk_modules()
            except Exception as exc:
                self.logger.error("high risk prediction failed: %s", exc)
                modules = []
            if modules:
                if isinstance(modules[0], str):
                    modules_with_scores = [(m, 1.0) for m in modules]
                else:
                    modules_with_scores = modules  # type: ignore[assignment]
                self.preemptive_patch_modules(modules_with_scores)

        info = self._top_error(bot)
        if not info:
            return
        etype, module, mods, count = info
        if count < self.threshold:
            return
        path = Path(f"{module}.py")
        if not path.exists():
            return
        context_meta = {"error_type": etype, "module": module, "bot": bot}
        builder = self.context_builder
        if builder is None:
            try:
                builder = ContextBuilder()
            except Exception:
                builder = None
            self.context_builder = builder
        ctx_block = {}
        if builder is not None:
            try:
                query = f"{etype} in {module}"
                ctx_block = builder.build_context(query)
            except Exception:
                ctx_block = {}
        desc = f"quick fix {etype}"
        if ctx_block:
            desc += "\n\n" + json.dumps(ctx_block, indent=2)
        session_id = ""
        vectors = []
        if self.retriever is not None:
            try:
                _, metrics = self.retriever.retrieve_with_confidence(
                    module, top_k=1, return_metrics=True
                )
                if metrics:
                    session_id = metrics[0].get("session_id", "")
                    vectors = [
                        (m["origin_db"], m["record_id"]) for m in metrics if m.get("hit")
                    ]
            except Exception:
                self.logger.debug("retriever lookup failed", exc_info=True)
        if session_id:
            context_meta["retrieval_session_id"] = session_id
            context_meta["retrieval_vectors"] = vectors
        patch_id = None
        try:
            try:
                result = self.manager.run_patch(path, desc, context_meta=context_meta)
            except TypeError:
                result = self.manager.run_patch(path, desc)
            patch_id = getattr(result, "patch_id", None)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.error("quick fix failed for %s: %s", bot, exc)
        tests_ok = True
        try:
            subprocess.run(["pytest", "-q"], check=True)
        except Exception as exc:
            tests_ok = False
            self.logger.error("quick fix validation failed: %s", exc)
        try:
            self.graph.add_telemetry_event(
                bot, etype, module, mods, patch_id=patch_id, resolved=tests_ok
            )
            self.graph.update_error_stats(self.db)
        except Exception as exc:
            self.logger.exception("telemetry update failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    def preemptive_patch_modules(
        self,
        modules: Iterable[tuple[str, float]] | Iterable[str],
        *,
        risk_threshold: float | None = None,
    ) -> None:
        """Proactively patch ``modules`` that exceed ``risk_threshold``.

        Parameters
        ----------
        modules:
            Iterable of ``(module, risk_score)`` pairs as produced by
            :meth:`~error_cluster_predictor.ErrorClusterPredictor.predict_high_risk_modules`.
        risk_threshold:
            Minimum risk score required to trigger a patch.  Defaults to the
            instance's ``risk_threshold`` attribute.
        """

        thresh = self.risk_threshold if risk_threshold is None else risk_threshold
        for item in modules:
            if isinstance(item, tuple):
                module, risk = item
            else:
                module, risk = item, 1.0
            if risk < thresh:
                continue
            path = Path(f"{module}.py")
            if not path.exists():
                continue
            meta = {"module": module, "reason": "preemptive_patch"}
            builder = self.context_builder
            if builder is None:
                try:
                    builder = ContextBuilder()
                except Exception:
                    builder = None
                self.context_builder = builder
            ctx = {}
            if builder is not None:
                try:
                    query = f"preemptive patch {module}"
                    ctx = builder.build_context(query)
                except Exception:
                    ctx = {}
            desc = "preemptive_patch"
            if ctx:
                desc += "\n\n" + json.dumps(ctx, indent=2)
            session_id = ""
            vectors = []
            if self.retriever is not None:
                try:
                    _, metrics = self.retriever.retrieve_with_confidence(
                        module, top_k=1, return_metrics=True
                    )
                    if metrics:
                        session_id = metrics[0].get("session_id", "")
                        vectors = [
                            (m["origin_db"], m["record_id"]) for m in metrics if m.get("hit")
                        ]
                except Exception:
                    self.logger.debug("retriever lookup failed", exc_info=True)
            if session_id:
                meta["retrieval_session_id"] = session_id
                meta["retrieval_vectors"] = vectors
            patch_id = None
            try:
                try:
                    result = self.manager.run_patch(path, desc, context_meta=meta)
                except TypeError:
                    result = self.manager.run_patch(path, desc)
                patch_id = getattr(result, "patch_id", None)
            except Exception as exc:  # pragma: no cover - runtime issues
                self.logger.error("preemptive patch failed for %s: %s", module, exc)
                try:
                    try:
                        patch_id = generate_patch(
                            module,
                            getattr(self.manager, "engine", None),
                            context_builder=self.context_builder,
                        )
                    except TypeError:
                        patch_id = generate_patch(
                            module, getattr(self.manager, "engine", None)
                        )
                except Exception:
                    patch_id = None
            try:
                self.graph.add_telemetry_event(
                    "predictor",
                    "predicted_high_risk",
                    module,
                    None,
                    patch_id=patch_id,
                )
            except Exception as exc:  # pragma: no cover - graph issues
                self.logger.error(
                    "failed to record telemetry for %s: %s", module, exc
                )
            try:
                self.db.log_preemptive_patch(module, risk, patch_id)
            except Exception as exc:  # pragma: no cover - db issues
                self.logger.error("failed to record preemptive patch for %s: %s", module, exc)

    # ------------------------------------------------------------------
    def run_and_validate(self, bot: str) -> None:
        """Run :meth:`run` then execute the test suite."""
        self.run(bot)


__all__ = ["QuickFixEngine", "generate_patch"]
