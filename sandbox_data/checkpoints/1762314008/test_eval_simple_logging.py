import ast
import logging

import pytest

from sandbox_runner.orphan_discovery import EvaluationError, _eval_simple


def test_eval_simple_logs_binop_exception(caplog):
    node = ast.parse("'%s' % ()", mode="eval").body
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EvaluationError):
            _eval_simple(node, {}, 1)
    assert "Unresolved expression" in caplog.text
    assert "not enough arguments" in caplog.text


def test_eval_simple_logs_call_exception(caplog):
    node = ast.parse("'hello'.index('x')", mode="eval").body
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EvaluationError):
            _eval_simple(node, {}, 1)
    assert "Unresolved expression" in caplog.text
    assert "substring not found" in caplog.text

