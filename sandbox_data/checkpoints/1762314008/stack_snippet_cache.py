from __future__ import annotations

"""Utilities for persisting small Stack dataset snippets on disk."""

from pathlib import Path
from typing import Optional

import os

try:  # pragma: no cover - prefer package-local import when available
    from .redaction_utils import redact_text as _redact_text  # type: ignore
except Exception:  # pragma: no cover - fallback for monolithic layout
    from redaction_utils import redact_text as _redact_text  # type: ignore


def _normalise_key(key: str) -> str:
    """Return a filesystem-safe representation of ``key``."""

    normalised = key.strip().lower()
    if not normalised:
        return ""
    return "".join(ch for ch in normalised if ch.isalnum())


class StackSnippetCache:
    """Persist redacted Stack snippets keyed by their summary hash."""

    def __init__(self, root: Path | str, *, max_chars: int = 1200) -> None:
        self.root = Path(root)
        self.max_chars = max(int(max_chars), 1)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _path_for(self, key: str) -> Path:
        safe = _normalise_key(key)
        if not safe:
            raise ValueError("snippet key must not be empty")
        prefix = safe[:2] or "00"
        subdir = safe[2:4] or prefix
        return self.root / prefix / subdir / f"{safe}.txt"

    # ------------------------------------------------------------------
    def _prepare(self, text: str) -> str:
        redacted = _redact_text(str(text or ""))
        redacted = redacted.replace("\x00", "").strip()
        if len(redacted) <= self.max_chars:
            return redacted
        # Avoid ellipsis when truncation would consume the whole snippet
        if self.max_chars <= 3:
            return redacted[: self.max_chars]
        return redacted[: self.max_chars - 3].rstrip() + "..."

    # ------------------------------------------------------------------
    def store(self, key: str, text: str) -> tuple[str, str]:
        """Persist ``text`` under ``key`` and return the snippet + pointer."""

        snippet = self._prepare(text)
        if not snippet:
            return "", ""

        path = self._path_for(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(snippet)
            os.replace(tmp_path, path)
        return snippet, str(path.relative_to(self.root))

    # ------------------------------------------------------------------
    def load(self, key: str) -> Optional[str]:
        """Return the cached snippet for ``key`` when available."""

        try:
            path = self._path_for(key)
        except ValueError:
            return None
        return self.load_by_pointer(str(path.relative_to(self.root)))

    # ------------------------------------------------------------------
    def load_by_pointer(self, pointer: str) -> Optional[str]:
        """Return snippet stored at ``pointer`` relative to the cache root."""

        if not pointer:
            return None
        path = (self.root / pointer).resolve()
        try:
            path.relative_to(self.root.resolve())
        except Exception:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None


__all__ = ["StackSnippetCache"]

