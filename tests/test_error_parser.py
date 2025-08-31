import traceback

from error_parser import ErrorParser


def test_parse_pytest_assertion():
    trace = (
        "tests/test_sample.py:3: in test_example\n"
        "    assert 1 == 2\n"
        "E   AssertionError: assert 1 == 2\n"
    )
    result = ErrorParser.parse(trace)
    assert result["error_type"] == "assertion_error"
    assert result["files"] == ["tests/test_sample.py"]
    assert result["tags"] == ["assertion_error"]
    assert result["signature"]


def test_parse_runtime_error():
    try:
        1 / 0
    except ZeroDivisionError:
        trace = traceback.format_exc()
    result = ErrorParser.parse(trace)
    assert "zero_division_error" in result["tags"]
    assert result["error_type"] == "zero_division_error"


def test_parse_duplicate_skipped():
    trace = "Traceback\nValueError: boom"
    first = ErrorParser.parse(trace)
    second = ErrorParser.parse(trace)
    assert first
    assert second == {}
