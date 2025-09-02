import prompt_chunker as pc
import textwrap

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - tiktoken may be missing
    tiktoken = None  # type: ignore

if tiktoken is not None:  # pragma: no branch - simple import logic
    _ENC = tiktoken.get_encoding("cl100k_base")
else:
    _ENC = None


def _count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text.split())


def test_token_counting_respects_limit() -> None:
    code = "\n".join(f"print({i})" for i in range(50))
    chunks = pc.split_into_chunks(code, 30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _count_tokens(chunk) <= 30


def test_ast_boundaries_preserved() -> None:
    code = textwrap.dedent(
        """
        def a():
            return 1

        def b():
            return 2
        """
    )
    chunks = pc.split_into_chunks(code, 20)
    assert len(chunks) == 2
    assert chunks[0].lstrip().startswith("def a")
    assert chunks[1].lstrip().startswith("def b")


def test_syntax_error_fallback() -> None:
    code = "def bad(:\n    pass"  # invalid syntax
    chunks = pc.split_into_chunks(code, 5)
    assert chunks  # returns something
    for chunk in chunks:
        assert _count_tokens(chunk) <= 5


def test_get_chunk_summaries_cache_hit(tmp_path, monkeypatch):
    file = tmp_path / "sample.py"
    file.write_text("def a():\n    return 1\n")

    calls = {"n": 0}

    def fake_summary(code: str) -> str:
        calls["n"] += 1
        return f"sum{calls['n']}"

    monkeypatch.setattr(pc, "summarize_chunk", fake_summary)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "CACHE_DIR", cache_dir)

    first = pc.get_chunk_summaries(file, 50)
    assert calls["n"] == len(first)

    second = pc.get_chunk_summaries(file, 50)
    assert second == first
    assert calls["n"] == len(first)  # no new summarisation


def test_get_chunk_summaries_cache_invalidation(tmp_path, monkeypatch):
    file = tmp_path / "sample.py"
    file.write_text("def a():\n    return 1\n")

    calls = {"n": 0}

    def fake_summary(code: str) -> str:
        calls["n"] += 1
        return f"sum{calls['n']}"

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pc, "summarize_chunk", fake_summary)
    monkeypatch.setattr(pc, "CACHE_DIR", cache_dir)

    first = pc.get_chunk_summaries(file, 50)
    assert calls["n"] == len(first)
    first_files = list(cache_dir.iterdir())
    assert len(first_files) == 1

    file.write_text("def a():\n    return 2\n")  # change content -> new hash
    second = pc.get_chunk_summaries(file, 50)
    assert calls["n"] == len(first) + len(second)
    cache_files = list(cache_dir.iterdir())
    assert len(cache_files) == 2  # old + new cache files
    assert first != second
