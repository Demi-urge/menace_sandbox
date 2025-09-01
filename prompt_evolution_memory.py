from __future__ import annotations

"""Persistent prompt logging for evolution experiments.

This module provides :class:`PromptEvolutionMemory` which appends structured
records about prompt executions to JSON lines files.  Two files are maintained –
one for successful executions and another for failures.  Each record captures
the prompt text, formatting metadata, execution result, optional ROI data and a
timestamp.  File writes are protected with a file lock to allow concurrent
writers across processes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, Dict
import json
import time

try:
    # Prefer the sandbox aware lock if available
    from lock_utils import SandboxLock as FileLock  # type: ignore
except Exception:  # pragma: no cover - fallback when lock_utils missing
    from filelock import FileLock  # type: ignore

try:  # pragma: no cover - avoid heavy import when Prompt is absent
    from llm_interface import Prompt  # type: ignore
except Exception:  # pragma: no cover - lightweight placeholder
    class Prompt:  # type: ignore
        system: str = ""
        user: str = ""
        examples: Iterable[str] = ()


@dataclass
class PromptEvolutionMemory:
    """Append prompt execution data to JSON lines files."""

    success_path: Path = Path("prompt_memory_success.jsonl")
    failure_path: Path = Path("prompt_memory_failure.jsonl")

    def __post_init__(self) -> None:
        for p in (self.success_path, self.failure_path):
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        """Append ``record`` as a JSON line to ``path`` with locking."""

        lock_path = Path(path).with_suffix(Path(path).suffix + ".lock")
        line = json.dumps(record)
        try:
            with FileLock(str(lock_path)):
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            # Swallow any error to avoid interrupting caller
            pass

    # ------------------------------------------------------------------
    def log_prompt(
        self,
        prompt: Prompt,
        success: bool,
        exec_result: Dict[str, Any],
        roi: Dict[str, Any] | None,
        format_meta: Dict[str, Any],
    ) -> None:
        """Persist details about a prompt execution.

        Parameters
        ----------
        prompt:
            Prompt object containing system/user text and examples.
        success:
            Whether the execution was successful.  Determines which file the
            record is appended to.
        exec_result:
            Structured outcome from executing the prompt.
        roi:
            Optional ROI data associated with this execution.
        format_meta:
            Metadata emitted during prompt formatting.
        """

        record: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "prompt": {
                "system": getattr(prompt, "system", ""),
                "user": getattr(prompt, "user", ""),
                "examples": list(getattr(prompt, "examples", [])),
            },
            "format": format_meta,
            "exec_result": exec_result,
        }
        if roi is not None:
            record["roi"] = roi

        path = self.success_path if success else self.failure_path
        self._append(path, record)


__all__ = ["PromptEvolutionMemory"]
