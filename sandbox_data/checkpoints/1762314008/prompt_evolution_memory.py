from __future__ import annotations

"""Lightweight persistence layer for prompt evolution experiments."""

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Dict

from filelock import FileLock
from dynamic_path_router import get_project_root, resolve_path


def _resolve_or_prepare(relative_path: str) -> Path:
    """Return an absolute path for ``relative_path`` creating parents if needed.

    The production environment ships ``sandbox_data`` files alongside the
    repository, however our test harnesses (and fresh checkouts on Windows)
    often lack these artefacts.  Import time should therefore not fail simply
    because the log files have not been created yet.  We attempt to resolve the
    path via :func:`dynamic_path_router.resolve_path` and, when that fails,
    fall back to constructing the path relative to the repository root.
    """

    try:
        return resolve_path(relative_path)
    except FileNotFoundError:
        base = get_project_root()
        path = base / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

try:  # pragma: no cover - optional dependency
    from llm_interface import Prompt  # type: ignore
except Exception:  # pragma: no cover - llm_interface unavailable
    try:  # pragma: no cover - optional dependency
        from prompt_types import Prompt  # type: ignore
    except Exception as exc:  # pragma: no cover - explicit failure
        raise ImportError(
            "Prompt dataclass is required for PromptEvolutionMemory. "
            "Install 'prompt_types' or ensure 'llm_interface' is available."
        ) from exc


@dataclass
class PromptEvolutionMemory:
    """Append prompt execution records to JSONL logs.

    The memory stores each prompt attempt in line-delimited JSON files.
    Successful executions are written to ``sandbox_data/prompt_success_log.jsonl``
    while failures are appended to ``sandbox_data/prompt_failure_log.jsonl``.
    Each record captures the prompt contents, optional formatting metadata,
    execution results and ROI metrics.
    """

    success_path: Path = field(
        default_factory=lambda: _resolve_or_prepare(
            "sandbox_data/prompt_success_log.jsonl"
        )
    )
    failure_path: Path = field(
        default_factory=lambda: _resolve_or_prepare(
            "sandbox_data/prompt_failure_log.jsonl"
        )
    )

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        for path in (self.success_path, self.failure_path):
            path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        lock = FileLock(str(path) + ".lock")
        line = json.dumps(record, ensure_ascii=False)
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ------------------------------------------------------------------
    def log(
        self,
        prompt: Prompt,
        success: bool,
        result: Dict[str, Any],
        roi: Dict[str, Any] | None = None,
        format_meta: Dict[str, Any] | None = None,
        *,
        module: str = "unknown",
        action: str = "unknown",
        prompt_text: str | None = None,
    ) -> None:
        """Record details of a prompt execution.

        Parameters
        ----------
        prompt:
            Prompt object containing ``system``/``user`` text and examples.
        success:
            Whether the execution succeeded.
        result:
            Structured information about the execution outcome.
        roi:
            Optional ROI metrics describing impact of the execution.
        format_meta:
            Optional formatting metadata produced while building the prompt.
        module, action:
            Identifier describing where the prompt originated.
        prompt_text:
            Flattened full prompt text. If omitted it will be derived from
            ``prompt``.
        """

        if prompt_text is None:
            parts = [prompt.system, *prompt.examples, prompt.user]
            prompt_text = "\n".join([p for p in parts if p])

        record: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "module": module,
            "action": action,
            "prompt": {
                "system": prompt.system,
                "user": prompt.user,
                "examples": list(prompt.examples),
                "metadata": dict(prompt.metadata),
            },
            "prompt_text": prompt_text,
            "result": result,
            "roi": roi or {},
            "success": success,
        }
        if format_meta:
            record["format"] = format_meta
        path = self.success_path if success else self.failure_path
        try:  # pragma: no cover - defensive
            self._append(path, record)
        except Exception:
            pass


PromptEvolutionLogger = PromptEvolutionMemory

__all__ = ["PromptEvolutionMemory", "PromptEvolutionLogger"]
