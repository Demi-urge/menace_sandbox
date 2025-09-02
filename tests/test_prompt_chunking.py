from __future__ import annotations

from pathlib import Path
from typing import List

import json

import prompt_chunking as pc


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


def test_chunk_code_splits_top_level_defs(tmp_path: Path) -> None:
    file = _write_sample(tmp_path)
    chunks = pc.chunk_code(file, 100)
    assert len(chunks) == 2
    assert chunks[0].lstrip().startswith("def a")
    assert chunks[1].lstrip().startswith("def b")
    for chunk in chunks:
        assert pc._count_tokens(chunk) <= 100


def test_get_chunk_summaries_cache(tmp_path: Path, monkeypatch) -> None:
    file = _write_sample(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(pc, "CACHE_DIR", cache)
    cache.mkdir()

    calls: List[str] = []

    def fake_sum(code: str) -> str:
        calls.append(code)
        return f"summary-{len(calls)}"

    monkeypatch.setattr(pc, "summarize_code", fake_sum)

    first = pc.get_chunk_summaries(file, 100)
    assert len(first) == 2
    assert len(calls) == 2
    assert len(list(cache.glob("*.json"))) == 2

    second = pc.get_chunk_summaries(file, 100)
    assert len(second) == 2
    assert len(calls) == 2  # cache hit

    # modify file -> one chunk changes
    file.write_text(file.read_text().replace("return 1", "return 42"))
    third = pc.get_chunk_summaries(file, 100)
    assert len(third) == 2
    assert len(calls) == 3  # one new summary

