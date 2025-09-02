from pathlib import Path
from typing import List

import chunking as pc


def _write_sample(path: Path) -> Path:
    code = """
def a():
    return 1


def b():
    return 2
"""
    file = path / "sample.py"
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
    cache = tmp_path / "cache"
    monkeypatch.setattr(pc, "CACHE_DIR", cache)
    cache.mkdir()

    calls: List[str] = []

    def fake_sum(code: str, llm=None) -> str:
        calls.append(code)
        return f"summary-{len(calls)}"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    chunks = pc.chunk_file(file, 5)
    first = pc.get_chunk_summaries(file, 5)
    assert len(first) == len(chunks)
    assert len(calls) == len(chunks)
    assert len(list(cache.glob("*.json"))) == 1

    second = pc.get_chunk_summaries(file, 5)
    assert len(second) == len(chunks)
    assert len(calls) == len(chunks) * 2  # summaries recomputed

    # modify file -> one chunk changes
    file.write_text(file.read_text().replace("return 1", "return 42"))
    third = pc.get_chunk_summaries(file, 5)
    assert len(third) == len(chunks)
    assert len(calls) == len(chunks) * 3  # all summaries recomputed again
    assert len(list(cache.glob("*.json"))) == 2


def _write_large(path: Path, funcs: int = 4, lines: int = 900) -> Path:
    chunks = []
    for i in range(funcs):
        body = "\n".join(f"    v_{i}_{j} = {j}" for j in range(lines))
        chunks.append(f"def f{i}():\n{body}\n")
    file = path / "big.py"
    file.write_text("\n\n".join(chunks))
    return file


def test_large_file_chunking_and_cache(tmp_path: Path, monkeypatch) -> None:
    file = _write_large(tmp_path)
    token_limit = 1000
    chunks = pc.chunk_file(file, token_limit)
    assert all(pc._count_tokens(c.text) <= token_limit for c in chunks)

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(pc, "CACHE_DIR", cache)

    calls: List[str] = []

    def fake_sum(code: str, llm=None) -> str:
        calls.append(code)
        return f"summary-{len(calls)}"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    first = pc.get_chunk_summaries(file, token_limit)
    assert len(first) == len(chunks)
    assert len(calls) == len(chunks)
    assert len(list(cache.glob("*.json"))) == 1

    second = pc.get_chunk_summaries(file, token_limit)
    assert len(second) == len(chunks)
    assert len(calls) == len(chunks) * 2

    file.write_text(file.read_text().replace("v_0_0 = 0", "v_0_0 = -1"))
    third = pc.get_chunk_summaries(file, token_limit)
    assert len(third) == len(chunks)
    assert len(calls) == len(chunks) * 3
    assert len(list(cache.glob("*.json"))) == 2
