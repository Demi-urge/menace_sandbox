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
