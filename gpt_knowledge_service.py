"""High level summarisation service built on :mod:`gpt_memory`.

The :class:`GPTKnowledgeService` periodically scans recent interactions stored by
:class:`gpt_memory.GPTMemoryManager`, groups them by tag and writes short
summaries back into the memory using the ``INSIGHT`` tag.  Other bots can query
these summaries through :meth:`get_recent_insights`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import importlib.util
import sys
from pathlib import Path

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

_gpt_memory = load_internal("gpt_memory")
GPTMemoryManager = _gpt_memory.GPTMemoryManager
INSIGHT = _gpt_memory.INSIGHT
_summarise_text = _gpt_memory._summarise_text

from governed_retrieval import govern_retrieval, redact


class GPTKnowledgeService:
    """Summarise recent GPT interactions by tag.

    Parameters
    ----------
    manager:
        Instance of :class:`GPTMemoryManager` to read/write memory entries.
    max_per_tag:
        Maximum number of recent raw interactions per tag to include when
        generating a summary.
    """

    def __init__(self, manager: GPTMemoryManager, *, max_per_tag: int = 20) -> None:
        self.manager = manager
        self.max_per_tag = max_per_tag
        # Generate initial summaries on startup
        self.update_insights()

    # ------------------------------------------------------------------ helpers
    def _load_recent_by_tag(self) -> Dict[str, List[str]]:
        """Return recent interaction texts grouped by tag."""

        cur = self.manager.conn.execute(
            "SELECT prompt, response, tags FROM interactions ORDER BY ts DESC"
        )
        grouped: Dict[str, List[str]] = defaultdict(list)
        for prompt, response, tag_str in cur.fetchall():
            if not tag_str:
                continue
            tags = [t for t in tag_str.split(",") if t]
            if INSIGHT in tags:
                # Don't recursively summarise existing insights
                continue
            for tag in tags:
                if len(grouped[tag]) >= self.max_per_tag:
                    continue
                grouped[tag].append(f"{prompt} {response}")
        return grouped

    # ------------------------------------------------------------------ external
    def update_insights(self) -> None:
        """Generate and store summaries for each tag.

        Each summary is written back to memory with both the original tag and the
        ``INSIGHT`` tag so that it can be queried later.
        """

        grouped = self._load_recent_by_tag()
        for tag, texts in grouped.items():
            try:
                summary = _summarise_text("\n".join(texts))
            except Exception:  # pragma: no cover - defensive
                continue
            self.manager.log_interaction(
                f"insight:{tag}", summary, tags=[tag, INSIGHT]
            )

    def get_recent_insights(self, tag: str) -> str:
        """Return the latest stored insight for ``tag``."""

        entries = self.manager.retrieve("", tags=[INSIGHT, tag], limit=1)
        if not entries:
            return ""
        text = entries[0].response
        governed = govern_retrieval(text)
        if governed is None:
            return ""
        return redact(text)
