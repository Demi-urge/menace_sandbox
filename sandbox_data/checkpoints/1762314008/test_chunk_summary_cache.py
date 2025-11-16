from __future__ import annotations

from pathlib import Path
import threading

from chunk_summary_cache import ChunkSummaryCache
import chunking as pc


class DummyBuilder:
    def build(self, text: str) -> str:  # pragma: no cover - simple stub
        return ""


def test_roundtrip_and_invalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ChunkSummaryCache(cache_dir)

    # Prepare sample file and compute path hash
    file = tmp_path / "sample.py"  # path-ignore
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


def test_file_change_invalidates_cache_via_chunking(tmp_path: Path, monkeypatch) -> None:
    """Changing a source file triggers cache invalidation on access."""

    file = tmp_path / "sample.py"  # path-ignore
    file.write_text("def a():\n    return 1\n")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    calls: list[str] = []

    def fake_sum(code: str, llm=None, context_builder=None) -> str:
        calls.append(code)
        return "sum"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    first = pc.get_chunk_summaries(file, 20, context_builder=DummyBuilder())
    assert len(calls) == len(first)

    # Modify file so cached entry becomes stale
    file.write_text("def a():\n    return 2\n")
    second = pc.get_chunk_summaries(file, 20, context_builder=DummyBuilder())

    # Cache was invalidated -> summaries recomputed
    assert len(second) == len(first)
    assert len(calls) == len(first) * 2


def test_concurrent_requests_use_per_path_lock(tmp_path: Path, monkeypatch) -> None:
    """Parallel summary requests for the same file only compute once."""

    file = tmp_path / "sample.py"  # path-ignore
    file.write_text("def a():\n    return 1\n")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    calls: list[str] = []

    def fake_sum(code: str, llm=None, context_builder=None) -> str:
        calls.append(code)
        return "sum"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    def worker() -> None:
        pc.get_chunk_summaries(file, 20, context_builder=DummyBuilder())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # summarizer invoked only once per chunk despite concurrent callers
    assert len(calls) == len(pc.chunk_file(file, 20))
