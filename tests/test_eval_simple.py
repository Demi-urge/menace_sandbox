import ast
import logging
import os

import pytest

from sandbox_runner.orphan_discovery import EvaluationError, _eval_simple


def _parse(expr: str) -> ast.AST:
    return ast.parse(expr, mode="eval").body


def test_env_var_resolution(monkeypatch):
    monkeypatch.setenv("MY_VAR", "value")
    node = _parse('os.getenv("MY_VAR", "fallback")')
    assert _eval_simple(node, {}, 1) == "value"
    monkeypatch.delenv("MY_VAR")
    assert _eval_simple(node, {}, 1) == "fallback"


def test_concat_with_assignment():
    assignments = {
        "a": [(0, _parse('"foo"'))],
    }
    node = _parse('a + "bar"')
    assert _eval_simple(node, assignments, 10) == "foobar"


def test_unresolved_logs(caplog):
    node = _parse("unknown")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EvaluationError):
            _eval_simple(node, {}, 1)
    assert "Unresolved expression" in caplog.text

