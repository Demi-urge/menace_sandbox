import ast

from sandbox_runner.orphan_discovery import _eval_simple


def _parse(expr: str) -> ast.AST:
    return ast.parse(expr, mode="eval").body


def test_list_comprehension():
    node = _parse("[x * 2 for x in [1, 2, 3]]")
    assert _eval_simple(node, {}, 1) == [2, 4, 6]


def test_dict_comprehension():
    node = _parse("{k: v for k, v in [('a', 1), ('b', 2)]}")
    assert _eval_simple(node, {}, 1) == {"a": 1, "b": 2}


def test_fstring():
    node = _parse("f'foo {1 + 1}'")
    assert _eval_simple(node, {}, 1) == "foo 2"


def test_arithmetic_expression():
    node = _parse("1 + 2 * 3 - (-4)")
    assert _eval_simple(node, {}, 1) == 1 + 2 * 3 - (-4)

