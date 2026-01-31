import inspect
import json
import logging
from contextlib import contextmanager

import pytest

from logging_utils import get_logger
from logging_wrapper import wrap_with_logging


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def capture_logger(name: str):
    logger = get_logger(name)
    handler = ListHandler()
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_wrap_with_logging_behavior_and_signature_preserved():
    logger_name = "tests.logging_wrapper.behavior"

    def sample(a, b=2, *, c="hi"):
        """Sample docstring."""
        return f"{a}-{b}-{c}"

    def returns_none():
        return None

    wrapped = wrap_with_logging(sample, config={"logger_name": logger_name})
    assert wrapped(1, b=3, c="x") == sample(1, b=3, c="x")
    assert inspect.signature(wrapped) == inspect.signature(sample)
    assert wrapped.__name__ == sample.__name__
    assert wrapped.__doc__ == sample.__doc__

    wrapped_none = wrap_with_logging(returns_none, config={"logger_name": logger_name})
    assert wrapped_none() is returns_none()


def test_wrap_with_logging_structured_logging_and_json_safe():
    logger_name = "tests.logging_wrapper.structured"

    def echo(value, *, text="ok"):
        return {"value": value, "text": text}

    wrapped = wrap_with_logging(echo, config={"logger_name": logger_name})
    with capture_logger(logger_name) as handler:
        result = wrapped(3, text="hello")

    assert result == {"value": 3, "text": "hello"}
    assert len(handler.records) == 1
    record = handler.records[0]
    payload = {
        "event": record.event,
        "function": record.function,
        "args": record.extra_args,
        "kwargs": record.kwargs,
        "duration_s": record.duration_s,
        "return_value": record.return_value,
    }
    json.dumps(payload)
    assert payload["event"] == "function_call"
    assert payload["function"] == "echo"
    assert payload["args"] == [3]
    assert payload["kwargs"] == {"text": "hello"}


def test_wrap_with_logging_exception_logging_and_reraise():
    logger_name = "tests.logging_wrapper.exception"

    def boom(code):
        raise ValueError(f"boom-{code}")

    wrapped = wrap_with_logging(boom, config={"logger_name": logger_name})
    with capture_logger(logger_name) as handler:
        with pytest.raises(ValueError, match="boom-5"):
            wrapped(5)

    assert len(handler.records) == 1
    record = handler.records[0]
    exc = record.exception
    assert exc["type"] == "ValueError"
    assert exc["message"] == "boom-5"
    assert isinstance(exc["traceback"], list)
    assert any("ValueError" in line for line in exc["traceback"])
    json.dumps(exc)


def test_wrap_with_logging_large_argument_truncation_and_order():
    logger_name = "tests.logging_wrapper.truncation"
    config = {
        "logger_name": logger_name,
        "max_items": 2,
        "max_string": 15,
        "truncate_marker": "<truncated>",
    }

    def echo(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    wrapped = wrap_with_logging(echo, config=config)
    with capture_logger(logger_name) as handler:
        wrapped([0, 1, 2, 3], {"b": 2, "a": 1, "c": 3}, "x" * 30)

    record = handler.records[0]
    args = record.extra_args
    assert args[0] == [0, 1, "<truncated>"]
    assert list(args[1].keys()) == ["a", "b", "<truncated>"]
    assert args[2] == "xxxx<truncated>"


def test_wrap_with_logging_double_wrapping_prevented():
    logger_name = "tests.logging_wrapper.double"

    def add(a, b):
        return a + b

    wrapped = wrap_with_logging(add, config={"logger_name": logger_name})
    wrapped_again = wrap_with_logging(wrapped, config={"logger_name": logger_name})
    assert wrapped_again is wrapped
    with capture_logger(logger_name) as handler:
        assert wrapped_again(1, 2) == 3
    assert len(handler.records) == 1
