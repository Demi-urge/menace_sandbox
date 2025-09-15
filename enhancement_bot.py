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
from .coding_bot_interface import self_coding_managed
from typing import TYPE_CHECKING, Dict, Any
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)
pipeline = ModelAutomationPipeline(context_builder=_context_builder)
evolution_orchestrator = get_orchestrator("EnhancementBot", data_bot, engine)
_th = get_thresholds("EnhancementBot")
persist_sc_thresholds(
    "EnhancementBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)
manager = internalize_coding_bot(
    "EnhancementBot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=evolution_orchestrator,
    threshold_service=ThresholdService(),
    roi_threshold=_th.roi_drop,
    error_threshold=_th.error_increase,
    test_failure_threshold=_th.test_failure_increase,
)

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
        :meth:`ContextBuilder.build` to gather additional historical or code
        context which is appended to the prompt before invoking the LLM client.
        ``confidence`` weights the retrieval scope so more confident summaries
        fetch a broader context window.
        """

        context = ""
        if confidence > 0.0:
            try:  # pragma: no cover - builder failures are non fatal
                top_k = max(1, int(5 * confidence))
                retrieved = context_builder.build(hint, top_k=top_k)
                context = compress_snippets({"snippet": retrieved}).get("snippet", "")
            except Exception:
                context = ""

        intent_meta: Dict[str, Any] = {
            "before_hash": self._hash(before),
            "after_hash": self._hash(after),
        }
        if hint:
            intent_meta["refactor_summary"] = hint
        if context:
            intent_meta["retrieved_context"] = context
        prompt = context_builder.build_prompt(
            "Summarize the code change.",
            intent=intent_meta,
            top_k=0,
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
            self.manager.auto_run_patch(file_path, desc)
            registry = getattr(self.manager, "bot_registry", None)
            if registry is not None:
                try:
                    name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
                    registry.update_bot(name, str(file_path))
                except Exception:  # pragma: no cover - best effort
                    logger.exception("bot registry update failed")
        else:
            file_path.write_text(proposal.new_code)

        subprocess.run(["pytest", "-q"], check=False)

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
