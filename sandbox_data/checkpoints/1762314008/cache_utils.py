from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Sequence, List, Dict

from retrieval_cache import RetrievalCache

# Shared cache configuration
_CACHE_TTL = 3600  # seconds
_CACHE_MAX_ENTRIES = 128

_cache: RetrievalCache | None = None


def _get_cache() -> RetrievalCache:
    """Return a singleton :class:`RetrievalCache` instance."""
    global _cache
    cache_path = Path.home() / ".cache" / "menace" / "retrieval_cache.db"
    if _cache is None or str(getattr(_cache, "path", "")) != str(cache_path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _cache = RetrievalCache(path=cache_path, ttl=_CACHE_TTL)
    return _cache


def get_cached_chain(query: str, dbs: Sequence[str] | None = None) -> List[dict[str, Any]] | None:
    """Fetch cached retrieval results for ``query`` and ``dbs``."""
    return _get_cache().get(query, dbs)


def set_cached_chain(
    query: str,
    dbs: Sequence[str] | None,
    results: List[dict[str, Any]],
) -> None:
    """Store retrieval results and enforce size limit."""
    cache = _get_cache()
    cache.set(query, dbs, results)
    if _CACHE_MAX_ENTRIES:
        conn = cache._conn
        count = conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0]
        if count > _CACHE_MAX_ENTRIES:
            to_delete = count - _CACHE_MAX_ENTRIES
            conn.execute(
                "DELETE FROM retrieval_cache WHERE rowid IN (SELECT rowid FROM retrieval_cache ORDER BY ts ASC LIMIT ?)",
                (to_delete,),
            )
            conn.commit()


def clear_cache() -> None:
    """Remove all cached retrieval results."""
    _get_cache().clear()


def show_cache() -> List[Dict[str, Any]]:
    """Return all cache entries as a list of dictionaries."""
    cache = _get_cache()
    rows = cache._conn.execute(
        "SELECT query, db_chain, ts, payload FROM retrieval_cache ORDER BY ts DESC"
    ).fetchall()
    entries: List[Dict[str, Any]] = []
    for query, db_chain, ts, payload in rows:
        try:
            results = json.loads(payload)
        except Exception:
            results = []
        entries.append(
            {
                "query": query,
                "dbs": [d for d in db_chain.split("|") if d],
                "ts": ts,
                "results": results,
            }
        )
    return entries


def cache_stats() -> Dict[str, Any]:
    """Return basic statistics about the cache."""
    cache = _get_cache()
    conn = cache._conn
    count = conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0]
    size = 0
    try:
        size = Path(cache.path).stat().st_size
    except OSError:
        pass
    return {"entries": count, "path": str(cache.path), "size": size}


__all__ = [
    "get_cached_chain",
    "set_cached_chain",
    "clear_cache",
    "show_cache",
    "cache_stats",
    "_get_cache",
]
