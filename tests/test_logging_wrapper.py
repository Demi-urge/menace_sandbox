import logging
import uuid
from datetime import datetime

import pytest

from menace.infra.logging import get_logger
from menace.infra.logging_wrapper import wrap_with_logging


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def logger_handler():
    name = f"test.logging_wrapper.{uuid.uuid4().hex}"
    logger = get_logger(name)["data"]["logger"]
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield logger, handler
    logger.removeHandler(handler)


def _iso_with_timezone(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None


def test_wrap_preserves_return_value(logger_handler) -> None:
    logger, _handler = logger_handler

    def add(a: int, b: int) -> int:
        return a + b

    wrapped = wrap_with_logging(add, {"logger_name": logger.name})

    assert wrapped(2, 3) == 5


def test_wrap_preserves_none(logger_handler) -> None:
    logger, _handler = logger_handler

    def return_none() -> None:
        return None

    wrapped = wrap_with_logging(return_none, {"logger_name": logger.name})

    assert wrapped() is None


def test_wrap_propagates_exception(logger_handler) -> None:
    logger, _handler = logger_handler

    class CustomError(Exception):
        pass

    def fail() -> None:
        raise CustomError("boom")

    wrapped = wrap_with_logging(fail, {"logger_name": logger.name})

    with pytest.raises(CustomError, match="boom"):
        wrapped()


def test_double_wrapping_is_noop(logger_handler) -> None:
    logger, _handler = logger_handler

    def echo(value: str) -> str:
        return value

    wrapped = wrap_with_logging(echo, {"logger_name": logger.name})
    wrapped_again = wrap_with_logging(wrapped, {"logger_name": logger.name})

    assert wrapped_again is wrapped


def test_argument_serialization_is_deterministic_and_truncated(logger_handler) -> None:
    logger, handler = logger_handler

    class Unserializable:
        def __init__(self, value: str) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f"Unserializable({self.value})"

    long_string = "x" * 20
    complex_arg = {"b": [1, 2, 3, 4], "a": {"nested": long_string}}
    list_arg = [1, 2, 3, 4, 5]
    tuple_arg = ("tuple", {"z": long_string}, [9, 8, 7, 6])

    def func(*_args, **_kwargs):
        return "ok"

    wrapped = wrap_with_logging(
        func,
        {
            "logger_name": logger.name,
            "max_collection_items": 3,
            "max_string_length": 10,
        },
    )

    wrapped(
        complex_arg,
        list_arg,
        tuple_arg,
        alpha={"d": 4, "b": 2, "a": 1, "c": 3},
        beta=[long_string, "short", "more", "extra"],
        gamma=Unserializable("payload"),
    )

    call_record = next(record for record in handler.records if record.event.endswith(".call"))
    context = call_record.context

    assert context["args"] == [
        {"a": {"nested": "xxxxxxx..."}, "b": [1, 2, 3]},
        [1, 2, 3],
        ["tuple", {"z": "xxxxxxx..."}, [9, 8, 7]],
    ]
    assert context["kwargs"] == {
        "alpha": {"a": 1, "b": 2, "c": 3},
        "beta": ["xxxxxxx...", "short", "more"],
        "gamma": "Unseria...",
    }


def test_logging_schema_and_timestamps(logger_handler) -> None:
    logger, handler = logger_handler

    class CustomError(Exception):
        pass

    def ok(value: str, *, flag: bool = True) -> str:
        return f"ok-{value}-{flag}"

    def fail() -> None:
        raise CustomError("boom")

    wrapped_ok = wrap_with_logging(
        ok,
        {"logger_name": logger.name, "include_timestamp": True},
    )
    wrapped_fail = wrap_with_logging(
        fail,
        {"logger_name": logger.name, "include_timestamp": True},
    )

    wrapped_ok("value", flag=False)

    with pytest.raises(CustomError):
        wrapped_fail()

    call_records = [r for r in handler.records if r.event.endswith(".call")]
    return_records = [r for r in handler.records if r.event.endswith(".return")]
    exception_records = [r for r in handler.records if r.event.endswith(".exception")]

    assert call_records
    assert return_records
    assert exception_records

    for record in call_records:
        context = record.context
        assert {"function", "module", "args", "kwargs", "timestamp"} <= set(context)
        _iso_with_timezone(context["timestamp"])

    for record in return_records:
        context = record.context
        assert {"function", "module", "args", "kwargs", "duration_ms", "result", "timestamp"} <= set(
            context
        )
        assert isinstance(context["duration_ms"], (int, float))
        assert context["duration_ms"] >= 0
        _iso_with_timezone(context["timestamp"])

    for record in exception_records:
        context = record.context
        assert {"function", "module", "args", "kwargs", "duration_ms", "timestamp"} <= set(
            context
        )
        assert context["exception_type"] == "CustomError"
        assert "boom" in context["exception_message"]
        assert context["duration_ms"] >= 0
        _iso_with_timezone(context["timestamp"])
        assert "traceback" in context


def test_wrap_does_not_mutate_arguments(logger_handler) -> None:
    logger, _handler = logger_handler

    data = {"items": [1, 2, 3]}

    def consume(payload):
        assert payload is data
        assert payload["items"] is data["items"]
        return len(payload["items"])

    wrapped = wrap_with_logging(consume, {"logger_name": logger.name})

    before_items_id = id(data["items"])
    before_items_copy = list(data["items"])

    assert wrapped(data) == 3

    assert data["items"] == before_items_copy
    assert id(data["items"]) == before_items_id
