from __future__ import annotations
# flake8: noqa

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
from .coding_bot_interface import (
    _bootstrap_dependency_broker,
    _current_bootstrap_context,
    advertise_bootstrap_placeholder,
    get_active_bootstrap_pipeline,
    prepare_pipeline_for_bootstrap,
    read_bootstrap_heartbeat,
    self_coding_managed,
)
from typing import TYPE_CHECKING, Callable, Dict, Any
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder
from context_builder import handle_failure, PromptBuildError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator

_active_pipeline, _active_manager = get_active_bootstrap_pipeline()
_BOOTSTRAP_PLACEHOLDER = advertise_bootstrap_placeholder(
    dependency_broker=_bootstrap_dependency_broker(),
    pipeline=_active_pipeline,
    manager=_active_manager,
)


class _Runtime:
    def __init__(
        self,
        registry: BotRegistry,
        data_bot: DataBot,
        context_builder: ContextBuilder,
        engine: SelfCodingEngine,
        pipeline: ModelAutomationPipeline,
        promoter: Callable[[SelfCodingManager | None], None] | None,
        evolution_orchestrator: "EvolutionOrchestrator",
        manager: SelfCodingManager,
    ) -> None:
        self.registry = registry
        self.data_bot = data_bot
        self.context_builder = context_builder
        self.engine = engine
        self.pipeline = pipeline
        self.promoter = promoter
        self.evolution_orchestrator = evolution_orchestrator
        self.manager = manager


_runtime: _Runtime | None = None


def _build_runtime() -> _Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime

    dependency_broker = _bootstrap_dependency_broker()
    broker_pipeline = getattr(dependency_broker, "active_pipeline", None)
    broker_manager = getattr(dependency_broker, "active_sentinel", None)
    bootstrap_pipeline, bootstrap_manager = get_active_bootstrap_pipeline()
    bootstrap_context = _current_bootstrap_context()

    pipeline_candidate = bootstrap_pipeline or broker_pipeline
    manager_candidate = bootstrap_manager or broker_manager

    placeholder_pipeline: object | None = None
    placeholder_manager: object | None = None
    sentinel_fallback = manager_candidate or getattr(bootstrap_context, "manager", None)
    try:
        placeholder_pipeline, placeholder_manager = advertise_bootstrap_placeholder(
            dependency_broker=dependency_broker,
            pipeline=pipeline_candidate,
            manager=sentinel_fallback,
        )
    except Exception:  # pragma: no cover - best effort placeholder
        placeholder_pipeline = pipeline_candidate
        placeholder_manager = sentinel_fallback

    if pipeline_candidate is None and read_bootstrap_heartbeat():
        pipeline_candidate = placeholder_pipeline
        manager_candidate = manager_candidate or placeholder_manager

    sentinel_candidate = manager_candidate or placeholder_manager
    try:
        dependency_broker.advertise(
            pipeline=pipeline_candidate or placeholder_pipeline,
            sentinel=sentinel_candidate,
        )
    except Exception:  # pragma: no cover - best effort broker advertisement
        sentinel_candidate = sentinel_candidate

    if pipeline_candidate is None and bootstrap_context is not None:
        pipeline_candidate = getattr(bootstrap_context, "pipeline", None)
        manager_candidate = manager_candidate or getattr(bootstrap_context, "manager", None)

    registry = BotRegistry()
    data_bot = DataBot(start_server=False)
    context_builder = create_context_builder()
    engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=context_builder)

    pipeline_promoter: Callable[[SelfCodingManager | None], None] | None = None
    if pipeline_candidate is None:
        pipeline_candidate, pipeline_promoter = prepare_pipeline_for_bootstrap(
            pipeline_cls=ModelAutomationPipeline,
            context_builder=context_builder,
            bot_registry=registry,
            data_bot=data_bot,
        )

    evolution_orchestrator = get_orchestrator("EnhancementBot", data_bot, engine)
    thresholds = get_thresholds("EnhancementBot")
    persist_sc_thresholds(
        "EnhancementBot",
        roi_drop=thresholds.roi_drop,
        error_increase=thresholds.error_increase,
        test_failure_increase=thresholds.test_failure_increase,
    )
    manager: SelfCodingManager | None = manager_candidate
    if manager is None:
        manager = internalize_coding_bot(
            "EnhancementBot",
            engine,
            pipeline_candidate,
            data_bot=data_bot,
            bot_registry=registry,
            evolution_orchestrator=evolution_orchestrator,
            threshold_service=ThresholdService(),
            roi_threshold=thresholds.roi_drop,
            error_threshold=thresholds.error_increase,
            test_failure_threshold=thresholds.test_failure_increase,
        )

    if pipeline_promoter is not None:
        pipeline_promoter(manager)

    _runtime = _Runtime(
        registry=registry,
        data_bot=data_bot,
        context_builder=context_builder,
        engine=engine,
        pipeline=pipeline_candidate,
        promoter=pipeline_promoter,
        evolution_orchestrator=evolution_orchestrator,
        manager=manager,
    )
    return _runtime


def get_runtime() -> _Runtime:
    return _build_runtime()


runtime = get_runtime()
registry = runtime.registry
data_bot = runtime.data_bot
_context_builder = runtime.context_builder
engine = runtime.engine
pipeline = runtime.pipeline
evolution_orchestrator = runtime.evolution_orchestrator
manager = runtime.manager

"""Automatically validate and merge Codex refactors.

The enhancement workflow depends on :class:`vector_service.ContextBuilder` for
supplementary code context.  If the ``vector_service`` package is unavailable an
informative :class:`ImportError` is raised at import time.
"""

import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import ast
from difflib import SequenceMatcher

from .code_database import CodeDB
from .chatgpt_enhancement_bot import (
    EnhancementDB,
    EnhancementHistory,
    Enhancement,
)
from .micro_models.diff_summarizer import summarize_diff
from snippet_compressor import compress_snippets

from billing.prompt_notice import prepend_payment_notice
from llm_interface import LLMClient, Prompt

try:
    from vector_service.context_builder import ContextBuilder
except Exception as exc:  # pragma: no cover - fail fast when dependency missing
    raise ImportError(
        "enhancement_bot requires vector_service.ContextBuilder; install the"
        " vector_service package to enable context retrieval"
    ) from exc

try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class RefactorProposal:
    """Proposed code improvement from Codex."""

    file_path: Path
    new_code: str
    author_bot: str = "codex"


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class EnhancementBot:
    """Automatically validate and merge Codex refactors."""

    def __init__(
        self,
        code_db: CodeDB | None = None,
        enhancement_db: EnhancementDB | None = None,
        cadence: int = 86_400,
        confidence_override: float = 1.0,
        *,
        context_builder: ContextBuilder,
        llm_client: LLMClient | None = None,
        manager: SelfCodingManager | None = None,
    ) -> None:
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.code_db = code_db or CodeDB()
        self.enh_db = enhancement_db or EnhancementDB()
        self.cadence = cadence
        self.confidence_override = confidence_override
        self.context_builder = context_builder
        self.db_weights = self.context_builder.refresh_db_weights()
        self.llm_client = llm_client
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = data_bot

    # ------------------------------------------------------------------
    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def _logic_preserved(self, before: str, after: str) -> bool:
        try:
            a_before = ast.parse(before)
            a_after = ast.parse(after)
        except Exception:
            return False
        names_before = [n.name for n in a_before.body if isinstance(n, ast.FunctionDef)]
        names_after = [n.name for n in a_after.body if isinstance(n, ast.FunctionDef)]
        if names_before != names_after:
            return False
        diff_ratio = SequenceMatcher(None, before, after).ratio()
        return diff_ratio < 1.0

    def _benchmark(self, code: str) -> float:
        """Execute ``code`` in a restricted namespace and measure runtime."""

        env: dict[str, object] = {}
        try:
            exec(compile(code, "<benchmark>", "exec"), {"__builtins__": {}}, env)
        except Exception:
            return 0.0

        fn = env.get("run") or env.get("main")
        if not callable(fn):
            return 0.0
        start = time.perf_counter()
        for _ in range(1000):
            try:
                fn(1, 1)
            except Exception:
                fn()
        return time.perf_counter() - start

    # ------------------------------------------------------------------
    def _codex_summarize(
        self,
        before: str,
        after: str,
        *,
        hint: str = "",
        confidence: float = 0.0,
        context_builder: ContextBuilder,
    ) -> str:
        """Summarise the code change using the internal LLM interface.

        ``hint`` is a short diff description generated by micro models. When a
        ``ContextBuilder`` is available the description is passed to
        :meth:`ContextBuilder.build_prompt` to gather additional historical or
        code context which is appended to the prompt before invoking the LLM
        client.
        ``confidence`` weights the retrieval scope so more confident summaries
        fetch a broader context window.
        """

        context = ""
        if confidence > 0.0:
            top_k = max(1, int(5 * confidence))
            retrieval_prompt: Prompt | None = None
            try:
                retrieval_prompt = context_builder.build_prompt(
                    hint,
                    top_k=top_k,
                )
            except Exception as exc:
                if isinstance(exc, PromptBuildError):
                    raise
                handle_failure(
                    "ContextBuilder.build_prompt failed for codex context retrieval",
                    exc,
                    logger=logger,
                )
            if retrieval_prompt is not None:
                examples = [
                    str(example)
                    for example in getattr(retrieval_prompt, "examples", [])
                    if str(example).strip()
                ]
                if examples:
                    joined = "\n\n".join(examples)
                    context = compress_snippets({"snippet": joined}).get("snippet", "")

        intent_meta: Dict[str, Any] = {
            "before_hash": self._hash(before),
            "after_hash": self._hash(after),
        }
        if hint:
            intent_meta["refactor_summary"] = hint
        if context:
            intent_meta["retrieved_context"] = context
        intent_payload = dict(intent_meta)
        intent_payload.setdefault("top_k", 0)
        try:
            prompt = engine.build_enriched_prompt(
                "Summarize the code change.",
                intent=intent_payload,
                context_builder=context_builder,
            )
        except Exception as exc:
            if isinstance(exc, PromptBuildError):
                raise
            handle_failure(
                "build_enriched_prompt failed for codex summary prompt",
                exc,
                logger=logger,
            )
        notice = prepend_payment_notice([])[0]["content"]
        prompt.system = notice

        if not self.llm_client:
            return ""

        try:
            result = self.llm_client.generate(prompt, context_builder=context_builder)
            return result.text.strip()
        except TypeError as exc:
            raise RuntimeError(
                "llm_client.generate missing context_builder support"
            ) from exc
        except Exception:  # pragma: no cover - llm failures are non fatal
            logger.exception("LLM generation failed")
            return ""

    # ------------------------------------------------------------------
    def evaluate(self, proposal: RefactorProposal) -> bool:
        file_path = resolve_path(proposal.file_path)
        orig_text = file_path.read_text()
        if not self._logic_preserved(orig_text, proposal.new_code):
            return False

        baseline = self._benchmark(orig_text)
        improved = self._benchmark(proposal.new_code)
        if baseline == 0:
            return False
        delta = (baseline - improved) / baseline
        if delta < 0.05 * self.confidence_override:
            return False

        orig_hash = self._hash(orig_text)
        new_hash = self._hash(proposal.new_code)
        summary = summarize_diff(orig_text, proposal.new_code)
        confidence = 0.9 if summary.strip() else 0.0
        codex_summary = self._codex_summarize(
            orig_text,
            proposal.new_code,
            hint=summary,
            confidence=confidence,
            context_builder=self.context_builder,
        )
        if codex_summary:
            summary = codex_summary

        if self.manager is not None:
            desc = f"apply enhancement from {proposal.author_bot}\n\n{proposal.new_code}"
            outcome = self.manager.auto_run_patch(file_path, desc)
            summary = outcome.get("summary") if outcome else None
            failed_tests = int(summary.get("self_tests", {}).get("failed", 0)) if summary else 0
            patch_id = outcome.get("patch_id") if outcome else None
            if summary is None or failed_tests:
                logger.warning("enhancement validation failed: %s tests", failed_tests)
                engine = getattr(self.manager, "engine", None)
                if patch_id is not None and hasattr(engine, "rollback_patch"):
                    try:
                        engine.rollback_patch(str(patch_id))
                    except Exception:
                        logger.exception("enhancement rollback failed")
                file_path.write_text(orig_text)
                if summary is None:
                    return False
                return False
            registry = getattr(self.manager, "bot_registry", None)
            if registry is not None:
                try:
                    name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
                    registry.update_bot(name, str(file_path))
                except Exception:  # pragma: no cover - best effort
                    logger.exception("bot registry update failed")
        else:
            file_path.write_text(proposal.new_code)

        self.enh_db.record_history(
            EnhancementHistory(
                file_path=str(file_path),
                original_hash=orig_hash,
                enhanced_hash=new_hash,
                metric_delta=delta,
                author_bot=proposal.author_bot,
            )
        )
        # store summary and code for future training
        self.enh_db.add(
            Enhancement(
                idea="",
                rationale="",
                summary=summary,
                score=delta,
                timestamp=datetime.utcnow().isoformat(),
                before_code=orig_text,
                after_code=proposal.new_code,
                triggered_by=proposal.author_bot,
            )
        )
        return True

    # ------------------------------------------------------------------
    def run_cycle(self, proposals: Iterable[RefactorProposal]) -> None:
        for prop in proposals:
            try:
                if self.evaluate(prop):
                    resolved = resolve_path(prop.file_path)
                    subprocess.run(["git", "add", str(resolved)], check=False)
                    subprocess.run(
                        ["git", "commit", "-m", f"auto enhance {resolved.name}"],
                        check=False,
                    )
            except Exception:
                continue


__all__ = ["RefactorProposal", "EnhancementBot"]
