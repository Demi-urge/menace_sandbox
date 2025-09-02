from __future__ import annotations

"""Disk-backed cache for code chunk summaries.

This module stores pre-computed summaries for code chunks in individual JSON
files.  Each cache file is named after a hash of the file path so filenames stay
short and portable.  The JSON payload contains the original file path, a hash of
its current contents and the list of chunk summaries.  When the file contents
change the cache entry is ignored automatically.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import List, Dict

__all__ = ["ChunkSummaryCache"]


@dataclass
class ChunkSummaryCache:
    """Persist summaries for tokenised file chunks on disk.

    Parameters
    ----------
    cache_dir:
        Directory used to store JSON cache files.  Defaults to
        ``chunk_summary_cache/`` in the current working directory.
    """

    cache_dir: str | Path = "chunk_summary_cache"
    _lock: threading.Lock = field(init=False, repr=False)
    _paths: dict[str, Path] = field(init=False, default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def hash_path(self, path: str | Path) -> str:
        """Return a stable hash for ``path`` and remember the mapping."""

        p = Path(path)
        digest = hashlib.sha256(str(p).encode("utf-8")).hexdigest()
        self._paths[digest] = p
        return digest

    # ------------------------------------------------------------------
    def _cache_file(self, path_hash: str) -> Path:
        return self.cache_dir / f"{path_hash}.json"

    # ------------------------------------------------------------------
    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # ------------------------------------------------------------------
    def get(self, path_hash: str) -> Dict[str, List[Dict[str, object]]] | None:
        """Return cached summaries for ``path_hash`` if present and current."""

        cache_file = self._cache_file(path_hash)
        with self._lock:
            if not cache_file.exists():
                return None
            try:
                data = json.loads(cache_file.read_text())
            except Exception:
                return None
        path_str = data.get("path")
        file_hash = data.get("file_hash")
        if path_str and file_hash:
            p = Path(path_str)
            try:
                current_hash = self._file_hash(p)
            except OSError:
                return None
            if current_hash != file_hash:
                # file changed -> invalidate cache entry
                with self._lock:
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
                return None
        return data

    # ------------------------------------------------------------------
    def set(self, path_hash: str, summaries: List[Dict[str, object]]) -> None:
        """Store ``summaries`` for ``path_hash``.

        ``path_hash`` must have been produced by :meth:`hash_path` so that the
        original path can be looked up and the current file hash recorded.
        """

        path = self._paths.get(path_hash)
        if path is None:
            raise KeyError(f"unknown path hash: {path_hash}")
        data = {
            "path": str(path),
            "file_hash": self._file_hash(path),
            "summaries": summaries,
        }
        cache_file = self._cache_file(path_hash)
        tmp_file = cache_file.with_suffix(".tmp")
        with self._lock:
            with tmp_file.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp_file, cache_file)
