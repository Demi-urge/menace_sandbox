import logging
from contextlib import contextmanager

import pytest

from logging_utils import get_logger
from logging_wrappers import wrap_with_logging


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


def test_wrap_with_logging_returns_values_and_logs_once():
    logger_name = "tests.logging_wrappers.return"

    def add(a, b=2):
        return a + b

    wrapped = wrap_with_logging(add, config={"logger_name": logger_name})
    with capture_logger(logger_name) as handler:
        assert wrapped(1, b=3) == 4

    assert len(handler.records) == 1


def test_wrap_with_logging_propagates_exceptions():
    logger_name = "tests.logging_wrappers.exception"

    def boom(code):
        raise RuntimeError(f"boom-{code}")

    wrapped = wrap_with_logging(boom, config={"logger_name": logger_name})
    with capture_logger(logger_name) as handler:
        with pytest.raises(RuntimeError, match="boom-7"):
            wrapped(7)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.exception["type"] == "RuntimeError"


def test_wrap_with_logging_prevents_double_wrapping():
    logger_name = "tests.logging_wrappers.double"

    def add(a, b):
        return a + b

    wrapped = wrap_with_logging(add, config={"logger_name": logger_name})
    wrapped_again = wrap_with_logging(wrapped, config={"logger_name": logger_name})

    assert wrapped_again is wrapped
    with capture_logger(logger_name) as handler:
        assert wrapped_again(2, 3) == 5

    assert len(handler.records) == 1


def test_wrap_with_logging_large_args_sanitized_deterministically():
    logger_name = "tests.logging_wrappers.truncation"
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
        wrapped([0, 1, 2, 3], {"b": 2, "a": 1, "c": 3}, "y" * 30)

    record = handler.records[0]
    args = record.extra_args
    assert args[0] == [0, 1, "<truncated>"]
    assert list(args[1].keys()) == ["a", "b", "<truncated>"]
    assert args[2] == "yyyy<truncated>"
