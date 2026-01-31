import inspect
import json
import logging

import pytest

from logging_wrappers import wrap_with_logging


def test_wrap_with_logging_preserves_behavior_and_signature(caplog):
    def add(a, b=1):
        return a + b

    wrapped = wrap_with_logging(add)
    assert inspect.signature(wrapped) == inspect.signature(add)

    caplog.set_level(logging.INFO)
    result = wrapped(2, b=3)

    assert result == 5
    assert len(caplog.records) == 1
    payload = caplog.records[0].payload
    assert payload["event"] == "function_call"
    assert payload["function"] == "add"
    assert payload["args"] == [2]
    assert payload["kwargs"] == {"b": 3}
    assert payload["return_value"]["value"] == 5
    assert payload["return_value"]["is_none"] is False


def test_wrap_with_logging_is_json_safe_and_deterministic(caplog):
    def echo(value, **kwargs):
        return value

    wrapped = wrap_with_logging(echo)
    caplog.set_level(logging.INFO)

    result = wrapped({"b": 2, "a": 1}, z=0, a=1, data={3, 2, 1})
    assert result == {"b": 2, "a": 1}

    payload = caplog.records[0].payload
    assert list(payload["args"][0].keys()) == ["a", "b"]
    assert list(payload["kwargs"].keys()) == ["a", "data", "z"]
    assert payload["kwargs"]["data"] == [1, 2, 3]
    json.dumps(payload)


def test_wrap_with_logging_logs_exceptions_and_reraises(caplog):
    def boom(value):
        raise ValueError(f"bad {value}")

    wrapped = wrap_with_logging(boom)
    caplog.set_level(logging.INFO)

    with pytest.raises(ValueError, match="bad 4"):
        wrapped(4)

    payload = caplog.records[0].payload
    assert payload["exception"]["type"] == "ValueError"
    assert "bad 4" in payload["exception"]["message"]
    assert payload["args"] == [4]


def test_wrap_with_logging_prevents_double_wrapping():
    def identity(value):
        return value

    wrapped = wrap_with_logging(identity)
    wrapped_again = wrap_with_logging(wrapped)

    assert wrapped is wrapped_again
