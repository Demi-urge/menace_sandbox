from pathlib import Path
from typing import List

from chunk_summary_cache import ChunkSummaryCache
import chunking as pc
import threading


def _write_sample(path: Path) -> Path:
    code = """
def a():
    return 1


def b():
    return 2
"""
    file = path / "sample.py"  # path-ignore
    file.write_text(code)
    return file


def test_chunk_file_splits_top_level_defs(tmp_path: Path) -> None:
    file = _write_sample(tmp_path)
    chunks = pc.chunk_file(file, 5)
    assert any(ch.text.lstrip().startswith("def a") for ch in chunks)
    assert any(ch.text.lstrip().startswith("def b") for ch in chunks)
    for chunk in chunks:
        assert pc._count_tokens(chunk.text) <= 5


def test_get_chunk_summaries_cache(tmp_path: Path, monkeypatch) -> None:
    file = _write_sample(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    calls: List[str] = []

    def fake_sum(code: str, llm=None, context_builder=None) -> str:
        calls.append(code)
        return f"summary-{len(calls)}"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    chunks = pc.chunk_file(file, 5)
    first = pc.get_chunk_summaries(file, 5)
    assert len(first) == len(chunks)
    assert len(calls) == len(chunks)
    assert len(list(cache_dir.glob("*.json"))) == 1

    second = pc.get_chunk_summaries(file, 5)
    assert len(second) == len(chunks)
    assert len(calls) == len(chunks)  # cache hit -> no new summaries

    # modify file -> one chunk changes
    file.write_text(file.read_text().replace("return 1", "return 42"))
    third = pc.get_chunk_summaries(file, 5)
    assert len(third) == len(chunks)
    assert len(calls) == len(chunks) * 2  # summaries recomputed after change
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_get_chunk_summaries_concurrent(tmp_path: Path, monkeypatch) -> None:
    file = _write_sample(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    calls: List[str] = []

    def fake_sum(code: str, llm=None, context_builder=None) -> str:
        calls.append(code)
        return "sum"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    def worker():
        pc.get_chunk_summaries(file, 5)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == len(pc.chunk_file(file, 5))


def _write_large(path: Path, funcs: int = 4, lines: int = 900) -> Path:
    chunks = []
    for i in range(funcs):
        body = "\n".join(f"    v_{i}_{j} = {j}" for j in range(lines))
        chunks.append(f"def f{i}():\n{body}\n")
    file = path / "big.py"  # path-ignore
    file.write_text("\n\n".join(chunks))
    return file


def test_large_file_chunking_and_cache(tmp_path: Path, monkeypatch) -> None:
    file = _write_large(tmp_path)
    token_limit = 1000
    chunks = pc.chunk_file(file, token_limit)
    assert all(pc._count_tokens(c.text) <= token_limit for c in chunks)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    calls: List[str] = []

    def fake_sum(code: str, llm=None, context_builder=None) -> str:
        calls.append(code)
        return f"summary-{len(calls)}"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    first = pc.get_chunk_summaries(file, token_limit)
    assert len(first) == len(chunks)
    assert len(calls) == len(chunks)
    assert len(list(cache_dir.glob("*.json"))) == 1

    second = pc.get_chunk_summaries(file, token_limit)
    assert len(second) == len(chunks)
    assert len(calls) == len(chunks)

    file.write_text(file.read_text().replace("v_0_0 = 0", "v_0_0 = -1"))
    third = pc.get_chunk_summaries(file, token_limit)
    assert len(third) == len(chunks)
    assert len(calls) == len(chunks) * 2
    assert len(list(cache_dir.glob("*.json"))) == 1
