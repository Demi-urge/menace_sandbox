"""Simple GPT memory interface backed by :class:`MenaceMemoryManager`.

This module provides a thin wrapper that exposes two convenience APIs for
logging model interactions and retrieving context for follow up prompts.

Each logged entry stores the original ``prompt`` and ``response`` along with
placeholder metadata fields that can later be populated with human or
automatic feedback, error fixes, and suggested improvement paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List
import sqlite3

try:  # pragma: no cover - allow use outside package context
    from .menace_memory_manager import (
        MenaceMemoryManager,
        MemoryEntry,
        _summarise_text,
    )
except Exception:  # pragma: no cover - fallback when run as script
    from menace_memory_manager import (  # type: ignore
        MenaceMemoryManager,
        MemoryEntry,
        _summarise_text,
    )


class GPTMemory:
    """Persist and query conversation snippets for GPT style models."""

    def __init__(
        self,
        manager: MenaceMemoryManager | None = None,
        *,
        max_entries: int | None = 1000,
        max_age_days: int | None = None,
        summary_threshold: int = 2000,
    ) -> None:
        """Create a GPTMemory instance.

        Parameters
        ----------
        manager:
            Optional custom :class:`MenaceMemoryManager` instance.
        max_entries:
            Cap the total number of stored interactions. Older entries are
            merged and removed once the cap is exceeded.
        max_age_days:
            Remove entries older than this many days. ``None`` disables the
            age based retention policy.
        summary_threshold:
            Character length beyond which individual interactions are
            summarised using :func:`_summarise_text`.
        """

        self.manager = manager or MenaceMemoryManager()
        self.max_entries = max_entries
        self.max_age_days = max_age_days
        self.summary_threshold = summary_threshold

    # ------------------------------------------------------------------
    def log_interaction(self, prompt: str, response: str, tags: Iterable[str]) -> None:
        """Store a prompt/response pair with default metadata placeholders.

        Parameters
        ----------
        prompt:
            The user prompt supplied to the model.
        response:
            The model's textual response.
        tags:
            An iterable of tag strings associated with this interaction.
        """

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            # Metadata fields kept empty initially; they can be updated later
            "metadata": {
                "feedback": [],
                "error_fixes": [],
                "improvement_paths": [],
            },
        }

        # Summarise overly long interactions to keep context size manageable.
        if len(prompt) + len(response) > self.summary_threshold:
            payload["summary"] = _summarise_text(
                f"Prompt: {prompt}\nResponse: {response}"
            )

        entry = MemoryEntry(
            key=prompt[:100],  # use a truncated prompt as the key
            data=json.dumps(payload),
            version=1,
            tags=",".join(tags),
        )
        self.manager.log(entry)
        self._apply_retention()

    # ------------------------------------------------------------------
    def _apply_retention(self) -> None:
        """Merge and prune old entries according to retention policy."""

        cur = self.manager.conn.execute(
            "SELECT rowid, data, ts FROM memory ORDER BY ts DESC"
        )
        rows = cur.fetchall()
        if not rows:
            return

        remove_ids: List[int] = []

        if self.max_entries is not None and len(rows) > self.max_entries:
            remove_ids.extend(r[0] for r in rows[self.max_entries :])

        if self.max_age_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=self.max_age_days)
            for rowid, _data, ts in rows:
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        remove_ids.append(rowid)
                except Exception:
                    continue

        remove_ids = sorted(set(remove_ids))
        if not remove_ids:
            return

        # Merge removed entries into a single summary to preserve context.
        texts = [
            data for rowid, data, _ in rows if rowid in remove_ids
        ]
        summary = _summarise_text("\n".join(texts))
        if summary:
            try:
                self.manager.store("memory:summary", summary, tags="summary")
            except Exception:
                pass

        placeholders = ",".join("?" for _ in remove_ids)
        self.manager.conn.execute(
            f"DELETE FROM memory WHERE rowid IN ({placeholders})",
            tuple(remove_ids),
        )
        if getattr(self.manager, "has_fts", False):
            self.manager.conn.execute(
                f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})",
                tuple(remove_ids),
            )
        try:
            self.manager.conn.execute(
                f"DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})",
                tuple(remove_ids),
            )
        except sqlite3.Error:
            pass
        try:
            self.manager.conn.execute(
                f"DELETE FROM memory_clusters WHERE rowid IN ({placeholders})",
                tuple(remove_ids),
            )
        except sqlite3.Error:
            pass
        self.manager.conn.commit()

    # ------------------------------------------------------------------
    def search_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return stored interactions matching ``query``.

        The search leverages :class:`MenaceMemoryManager`'s text search which
        falls back to a simple ``LIKE`` lookup when FTS is unavailable.
        Results are returned as deserialised dictionaries containing the
        original prompt, response and metadata.
        """

        results = self.manager.search(query, limit)
        contexts: List[Dict[str, Any]] = []
        for entry in results:
            try:
                contexts.append(json.loads(entry.data))
            except json.JSONDecodeError:
                contexts.append({"prompt": entry.key, "response": entry.data, "metadata": {}})
        return contexts


__all__ = ["GPTMemory"]
