from __future__ import annotations

"""Append-only logger for prompt evolution experiments.

The :class:`PromptEvolutionLogger` records structured metadata about each
prompt execution for later analysis.  Successful executions are stored in
``prompt_success_log.json`` while failures are appended to
``prompt_failure_log.json``.  Each log entry captures the full prompt text,
metadata emitted by :class:`prompt_engine.PromptEngine`, outcome summaries and
ROI related metrics.  File operations are guarded with
:class:`filelock.FileLock` to avoid race conditions across processes.
"""

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Iterable, Any, Dict

from filelock import FileLock

try:  # pragma: no cover - avoid heavy imports in environments without Prompt
    from llm_interface import Prompt  # type: ignore
except Exception:  # pragma: no cover - minimal duck typing fallback
    class Prompt:  # type: ignore
        system: str = ""
        user: str = ""
        examples: Iterable[str] = ()
        tags: Iterable[str] = ()


@dataclass
class PromptEvolutionLogger:
    """Persist prompt execution records in JSON lines files."""

    success_path: Path = Path("prompt_success_log.json")
    failure_path: Path = Path("prompt_failure_log.json")

    def __post_init__(self) -> None:
        for p in (self.success_path, self.failure_path):
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock = FileLock(str(lock_path))
        line = json.dumps(record)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ------------------------------------------------------------------
    def _build_record(
        self,
        *,
        patch: str,
        prompt: Prompt,
        result_summary: str,
        roi_delta: float,
        coverage: float,
        runtime_delta: float | None = None,
        tags: Iterable[str] | None = None,
        prompt_engine: Any | None = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "patch": patch,
            "prompt": {
                "system": getattr(prompt, "system", ""),
                "user": getattr(prompt, "user", ""),
                "examples": list(getattr(prompt, "examples", [])),
            },
            "metadata": getattr(prompt_engine, "last_metadata", {}),
            "result": result_summary,
            "roi": {
                "roi_delta": roi_delta,
                "coverage": coverage,
            },
        }
        if runtime_delta is not None:
            record["roi"]["runtime_delta"] = runtime_delta
        if tags is None:
            tags = getattr(prompt, "tags", [])
        record["tags"] = list(tags)
        return record

    # ------------------------------------------------------------------
    def log_success(
        self,
        patch: str,
        prompt: Prompt,
        result_summary: str,
        *,
        roi_delta: float,
        coverage: float,
        runtime_delta: float | None = None,
        tags: Iterable[str] | None = None,
        prompt_engine: Any | None = None,
    ) -> None:
        """Append a successful execution record."""

        record = self._build_record(
            patch=patch,
            prompt=prompt,
            result_summary=result_summary,
            roi_delta=roi_delta,
            coverage=coverage,
            runtime_delta=runtime_delta,
            tags=tags,
            prompt_engine=prompt_engine,
        )
        try:
            self._append(self.success_path, record)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def log_failure(
        self,
        patch: str,
        prompt: Prompt,
        result_summary: str,
        *,
        roi_delta: float,
        coverage: float,
        runtime_delta: float | None = None,
        tags: Iterable[str] | None = None,
        prompt_engine: Any | None = None,
    ) -> None:
        """Append a failed execution record."""

        record = self._build_record(
            patch=patch,
            prompt=prompt,
            result_summary=result_summary,
            roi_delta=roi_delta,
            coverage=coverage,
            runtime_delta=runtime_delta,
            tags=tags,
            prompt_engine=prompt_engine,
        )
        try:
            self._append(self.failure_path, record)
        except Exception:
            pass


__all__ = ["PromptEvolutionLogger"]
