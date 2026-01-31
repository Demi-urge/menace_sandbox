import copy
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


def test_behavior_parity_and_input_integrity():
    logger_name = "tests.logging_wrapper.parity"
    side_effects: list[str] = []

    def mutate_external(values: list[int], meta: dict[str, int]):
        side_effects.append("called")
        return sum(values) + meta["offset"]

    def returns_none(payload: dict[str, int]):
        side_effects.append("none")
        _ = payload["offset"]
        return None

    args_list = [1, 2, 3]
    args_meta = {"offset": 4}
    list_snapshot = copy.deepcopy(args_list)
    meta_snapshot = copy.deepcopy(args_meta)

    wrapped = wrap_with_logging(mutate_external, config={"logger_name": logger_name})
    assert wrapped(args_list, args_meta) == mutate_external(args_list, args_meta)
    assert side_effects == ["called", "called"]
    assert args_list == list_snapshot
    assert args_meta == meta_snapshot

    wrapped_none = wrap_with_logging(returns_none, config={"logger_name": logger_name})
    assert wrapped_none(args_meta) is None
    assert side_effects[-1] == "none"


def test_exception_propagation():
    logger_name = "tests.logging_wrapper.exception"

    class CustomError(RuntimeError):
        pass

    def boom():
        raise CustomError("kaboom")

    wrapped = wrap_with_logging(boom, config={"logger_name": logger_name})
    with capture_logger(logger_name) as handler:
        with pytest.raises(CustomError, match="kaboom"):
            wrapped()

    assert len(handler.records) == 1
    assert handler.records[0].exception["type"] == "CustomError"


def test_signature_and_metadata_preserved():
    logger_name = "tests.logging_wrapper.signature"

    def sample(a, b=3, *, c="hi"):
        """Docstring."""
        return (a, b, c)

    wrapped = wrap_with_logging(sample, config={"logger_name": logger_name})
    assert inspect.signature(wrapped) == inspect.signature(sample)
    assert wrapped.__name__ == sample.__name__
    assert wrapped.__doc__ == sample.__doc__


def test_deterministic_logging_payloads(caplog):
    logger_name = "tests.logging_wrapper.deterministic"

    def echo(items, meta):
        return {"items": items, "meta": meta}

    wrapped = wrap_with_logging(echo, config={"logger_name": logger_name})
    logger = get_logger(logger_name)
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    caplog.set_level(logging.INFO, logger=logger_name)
    logger.handlers = [caplog.handler]
    logger.propagate = False
    try:
        wrapped({3, 1, 2}, {"b": 2, "a": 1})
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate

    assert len(caplog.records) == 1
    record = caplog.records[0]
    payload = {
        "args": record.extra_args,
        "kwargs": record.kwargs,
        "return_value": record.return_value,
    }
    json.dumps(payload)
    assert payload["args"][0] == [1, 2, 3]
    assert list(payload["kwargs"].keys()) == ["a", "b"]
    assert list(payload["return_value"]["meta"].keys()) == ["a", "b"]


def test_large_args_handling_is_deterministic():
    logger_name = "tests.logging_wrapper.large_args"
    config = {
        "logger_name": logger_name,
        "max_items": 3,
        "max_string": 12,
        "truncate_marker": "<truncated>",
    }

    def echo(values, text):
        return {"values": values, "text": text}

    wrapped = wrap_with_logging(echo, config=config)
    with capture_logger(logger_name) as handler:
        wrapped([1, 2, 3, 4, 5], "x" * 40)
        wrapped([1, 2, 3, 4, 5], "x" * 40)

    assert len(handler.records) == 2
    first_args = handler.records[0].extra_args
    second_args = handler.records[1].extra_args
    assert first_args == second_args
    assert first_args[0] == [1, 2, 3, "<truncated>"]
    assert first_args[1].endswith("<truncated>")
    assert len(first_args[1]) <= config["max_string"]
    assert handler.records[0].return_value["values"] == [1, 2, 3, "<truncated>"]


def test_double_wrap_prevention_contract():
    logger_name = "tests.logging_wrapper.double_wrap"

    def add(a, b):
        return a + b

    wrapped = wrap_with_logging(add, config={"logger_name": logger_name})
    wrapped_again = wrap_with_logging(wrapped, config={"logger_name": logger_name})
    assert wrapped_again is wrapped


def test_nested_wrap_safety():
    logger_name = "tests.logging_wrapper.nested"

    def inner(value):
        return value * 2

    def outer(value):
        return wrapped_inner(value) + 1

    wrapped_inner = wrap_with_logging(inner, config={"logger_name": logger_name})
    wrapped_outer = wrap_with_logging(outer, config={"logger_name": logger_name})

    with capture_logger(logger_name) as handler:
        assert wrapped_outer(3) == 7

    assert len(handler.records) == 2
    assert {record.function for record in handler.records} == {"inner", "outer"}
