from __future__ import annotations

from pathlib import Path

from chunk_summary_cache import ChunkSummaryCache


def test_roundtrip_and_invalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ChunkSummaryCache(cache_dir)

    # Prepare sample file and compute path hash
    file = tmp_path / "sample.py"
    file.write_text("print('hello')\n")
    path_hash = cache.hash_path(file)

    summaries = [
        {"start_line": 1, "end_line": 1, "summary": "print"},
    ]

    cache.set(path_hash, summaries)

    # Cache hit
    data = cache.get(path_hash)
    assert data is not None
    assert data["summaries"] == summaries

    # Change file contents -> cache invalidated
    file.write_text("print('changed')\n")
    assert cache.get(path_hash) is None
    assert not (cache_dir / f"{path_hash}.json").exists()
