from __future__ import annotations

"""Lightweight logger for prompt evolution experiments."""

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Dict

from filelock import FileLock

try:  # pragma: no cover - optional dependency
    from prompt_types import Prompt
except Exception as exc:  # pragma: no cover - explicit failure
    raise ImportError(
        "prompt_types.Prompt is required for PromptEvolutionLogger"
    ) from exc


_ROOT = Path(__file__).resolve().parent


@dataclass
class PromptEvolutionLogger:
    """Append prompt execution records to JSONL logs.

    Successful executions are written to ``prompt_success_log.json`` while
    failures are appended to ``prompt_failure_log.json``.  Each record captures
    the prompt contents, optional formatting metadata, execution results and
    ROI metrics.  The files are JSONL formatted regardless of the ``.json``
    extension.
    """

    success_path: Path = _ROOT / "prompt_success_log.json"
    failure_path: Path = _ROOT / "prompt_failure_log.json"

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


__all__ = ["PromptEvolutionLogger"]
