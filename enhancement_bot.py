from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import ast
from difflib import SequenceMatcher

from .code_database import CodeDB
from .chatgpt_enhancement_bot import (
    EnhancementDB,
    EnhancementHistory,
    Enhancement,
)
from .micro_models.diff_summarizer import summarize_diff
from .micro_models.prefix_injector import inject_prefix


@dataclass
class RefactorProposal:
    """Proposed code improvement from Codex."""

    file_path: Path
    new_code: str
    author_bot: str = "codex"


class EnhancementBot:
    """Automatically validate and merge Codex refactors."""

    def __init__(
        self,
        code_db: CodeDB | None = None,
        enhancement_db: EnhancementDB | None = None,
        cadence: int = 86_400,
        confidence_override: float = 1.0,
    ) -> None:
        self.code_db = code_db or CodeDB()
        self.enh_db = enhancement_db or EnhancementDB()
        self.cadence = cadence
        self.confidence_override = confidence_override

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
        self, before: str, after: str, *, hint: str = "", confidence: float = 0.0
    ) -> str:
        """Summarise the code change via Codex/OpenAI.

        ``hint`` is optional prior knowledge from a micro-model that will be
        injected into the system prefix when ``confidence`` exceeds the
        configured threshold.
        """
        try:  # pragma: no cover - optional dependency
            import os
            import openai  # type: ignore
        except Exception:
            return ""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        openai.api_key = api_key

        try:
            from . import codex_db_helpers as cdh
            examples = cdh.aggregate_examples(
                order_by="confidence",
                limit=3,
            )
        except Exception:  # pragma: no cover - helper failures
            examples = []
        example_context = "\n".join(
            e.text for e in examples if getattr(e, "text", "")
        )

        prompt = (
            "Summarize the code change.\nBefore:\n" + before + "\nAfter:\n" + after
        )
        if example_context:
            prompt += "\n\n### Training Examples\n" + example_context
        messages = [{"role": "user", "content": prompt}]
        if hint:
            messages = inject_prefix(
                messages,
                f"Diff summary hint: {hint}",
                confidence,
                role="system",
            )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def evaluate(self, proposal: RefactorProposal) -> bool:
        orig_text = proposal.file_path.read_text()
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
        )
        if codex_summary:
            summary = codex_summary

        proposal.file_path.write_text(proposal.new_code)

        subprocess.run(["pytest", "-q"], check=False)

        self.enh_db.record_history(
            EnhancementHistory(
                file_path=str(proposal.file_path),
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
                    subprocess.run(["git", "add", str(prop.file_path)], check=False)
                    subprocess.run(
                        ["git", "commit", "-m", f"auto enhance {prop.file_path.name}"],
                        check=False,
                    )
            except Exception:
                continue


__all__ = ["RefactorProposal", "EnhancementBot"]
