import importlib.util
import math
import sys
import types

from pathlib import Path


def _load_metrics():
    repo = Path(__file__).resolve().parent.parent
    metrics_path = repo / "self_improvement" / "metrics.py"  # path-ignore
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement.metrics", metrics_path
    )
    metrics = importlib.util.module_from_spec(spec)
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(repo)]  # type: ignore[attr-defined]
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)  # type: ignore[union-attr]
    return metrics


def test_ast_fallback_maintainability(monkeypatch, tmp_path):
    metrics = _load_metrics()
    code = """def f(a, b):\n    if a > b:\n        return a - b\n    return a + b\n"""
    src = tmp_path / "foo.py"  # path-ignore
    src.write_text(code)
    monkeypatch.setattr(metrics, "cc_visit", None)
    monkeypatch.setattr(metrics, "mi_visit", None)
    per_file, total, avg, tests, ent, div = metrics._collect_metrics([src], tmp_path)
    assert total == per_file["foo.py"]["complexity"]  # path-ignore
    assert avg == per_file["foo.py"]["maintainability"]  # path-ignore
    # Expected maintainability calculated using the MI formula
    # from the AST fallback implementation.
    file_complexity = per_file["foo.py"]["complexity"]  # path-ignore
    import ast, keyword, tokenize, io

    ops, operands = set(), set()
    N1 = N2 = 0
    sloc = set()
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENCODING,
        ):
            continue
        sloc.add(tok.start[0])
        if tok.type == tokenize.OP or (
            tok.type == tokenize.NAME and tok.string in keyword.kwlist
        ):
            ops.add(tok.string)
            N1 += 1
        elif tok.type in (
            tokenize.NAME,
            tokenize.NUMBER,
            tokenize.STRING,
        ):
            operands.add(tok.string)
            N2 += 1
    n = len(ops) + len(operands)
    N = N1 + N2
    volume = N * math.log2(n)
    expected = max(
        0.0,
        (171 - 5.2 * math.log(volume) - 0.23 * file_complexity - 16.2 * math.log(len(sloc)))
        * 100
        / 171,
    )
    assert math.isclose(per_file["foo.py"]["maintainability"], expected, rel_tol=1e-6)  # path-ignore
    assert tests == 0
    assert ent >= 0.0 and div >= 0.0


def test_radon_metrics_used(monkeypatch, tmp_path):
    metrics = _load_metrics()
    code = "def f():\n    return 1\n"
    src = tmp_path / "foo.py"  # path-ignore
    src.write_text(code)
    calls = {"cc": 0, "mi": 0}

    class Block:
        def __init__(self, complexity):
            self.complexity = complexity

    def fake_cc(text):
        calls["cc"] += 1
        return [Block(7)]

    def fake_mi(text, *_args):
        calls["mi"] += 1
        return 88.0

    monkeypatch.setattr(metrics, "cc_visit", fake_cc)
    monkeypatch.setattr(metrics, "mi_visit", fake_mi)

    per_file, total, avg, tests, ent, div = metrics._collect_metrics([src], tmp_path)
    assert per_file["foo.py"] == {  # path-ignore
        "complexity": 7,
        "maintainability": 88.0,
        "token_entropy": ent,
        "token_diversity": div,
    }
    assert total == 7
    assert avg == 88.0
    assert calls == {"cc": 1, "mi": 1}
    assert tests == 0


def test_skip_dirs_setting(tmp_path):
    metrics = _load_metrics()
    build = tmp_path / "build"
    build.mkdir()
    src = build / "foo.py"  # path-ignore
    src.write_text("def f():\n    return 1\n")

    per_file, total, avg, tests, ent, div = metrics._collect_metrics([src], tmp_path)
    assert per_file == {}
    assert total == 0
    assert avg == 0.0
    assert tests == 0

    settings = metrics.SandboxSettings(metrics_skip_dirs=[])
    per_file, total, avg, tests, ent, div = metrics._collect_metrics(
        [src], tmp_path, settings=settings
    )
    assert "build/foo.py" in per_file  # path-ignore
