from chunk_summary_cache import ChunkSummaryCache
import chunking as pc
import textwrap


class DummyBuilder:
    def build(self, text: str) -> str:  # pragma: no cover - simple stub
        return ""


def test_token_counting_respects_limit() -> None:
    code = "\n".join(f"print({i})" for i in range(50))
    chunks = pc.split_into_chunks(code, 30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 30


def test_ast_boundaries_preserved() -> None:
    code = textwrap.dedent(
        """\
        def a():
            return 1

        def b():
            return 2
        """
    )
    chunks = pc.split_into_chunks(code, 12)
    assert len(chunks) == 2
    assert chunks[0].text.lstrip().startswith("def a")
    assert chunks[0].start_line == 1
    assert chunks[1].text.lstrip().startswith("def b")
    assert chunks[1].start_line == 4
    assert chunks[0].end_line < chunks[1].start_line


def test_syntax_error_fallback() -> None:
    code = "def bad(:\n    pass"  # invalid syntax
    chunks = pc.split_into_chunks(code, 5)
    assert chunks  # returns something
    for chunk in chunks:
        assert chunk.token_count <= 5


def test_get_chunk_summaries_cache_hit(tmp_path, monkeypatch):
    file = tmp_path / "sample.py"  # path-ignore
    file.write_text("def a():\n    return 1\n")

    calls = {"n": 0}

    def fake_summary(code: str, llm=None, context_builder=None) -> str:
        calls["n"] += 1
        return "sum"

    monkeypatch.setattr(pc, "summarize_code", fake_summary)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    first = pc.get_chunk_summaries(file, 50, context_builder=DummyBuilder())
    assert calls["n"] == len(first)

    second = pc.get_chunk_summaries(file, 50, context_builder=DummyBuilder())
    assert second == first
    assert calls["n"] == len(first)  # cache hit


def test_get_chunk_summaries_cache_invalidation(tmp_path, monkeypatch):
    file = tmp_path / "sample.py"  # path-ignore
    file.write_text("def a():\n    return 1\n")

    calls = {"n": 0}

    def fake_summary(code: str, llm=None, context_builder=None) -> str:
        calls["n"] += 1
        return "sum"

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "summarize_code", fake_summary)
    monkeypatch.setattr(pc, "CHUNK_CACHE", ChunkSummaryCache(cache_dir))

    first = pc.get_chunk_summaries(file, 50, context_builder=DummyBuilder())
    assert calls["n"] == len(first)
    first_files = list(cache_dir.iterdir())
    assert len(first_files) == 1

    file.write_text("def a():\n    return 2\n")  # change content -> cache invalidated
    second = pc.get_chunk_summaries(file, 50, context_builder=DummyBuilder())
    assert calls["n"] == len(first) + len(second)
    cache_files = list(cache_dir.iterdir())
    assert len(cache_files) == 1  # file replaced
    assert first != second


def test_split_respects_line_ranges() -> None:
    code = "\n".join(f"print({i})" for i in range(1, 6))
    chunks = pc.split_into_chunks(code, 100, line_ranges=[(2, 3)])
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 1), (2, 3), (4, 5)]
