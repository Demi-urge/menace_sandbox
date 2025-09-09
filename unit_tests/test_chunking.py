import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402
from chunking import chunk_file, summarize_code  # noqa: E402


def test_token_counting_respects_limit(tmp_path):
    code = (
        'def foo():\n'
        '    x = 1\n'
        '    return x\n\n'
        'def bar():\n'
        '    y = 2\n'
        '    return y\n'
    )
    path = tmp_path / 'sample.py'  # path-ignore
    path.write_text(code)
    enc = tiktoken.get_encoding('cl100k_base')
    limit = 20
    chunks = chunk_file(path, limit)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(enc.encode(chunk.text)) <= limit


def test_ast_boundary_accuracy(tmp_path):
    code = (
        'def foo():\n'
        '    x = 1\n'
        '    return x\n\n'
        'def bar():\n'
        '    y = 2\n'
        '    return y\n'
    )
    path = tmp_path / 'code.py'  # path-ignore
    path.write_text(code)
    chunks = chunk_file(path, 20)
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 3)
    assert (chunks[1].start_line, chunks[1].end_line) == (5, 7)


def test_summarize_code_fallback():
    code = '# comment\n\nclass Foo:\n    pass\n'
    summary = summarize_code(code, None, context_builder=None)
    assert summary.startswith('class Foo')
