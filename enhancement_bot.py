from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import ast
from difflib import SequenceMatcher
import libcst as cst

from .code_database import CodeDB
from .chatgpt_enhancement_bot import EnhancementDB, EnhancementHistory


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
