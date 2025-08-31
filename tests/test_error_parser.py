import traceback

from error_parser import ErrorParser, parse_failure


def test_parse_failure_pytest_assertion():
    trace = (
        "tests/test_sample.py:3: in test_example\n"
        "    assert 1 == 2\n"
        "E   AssertionError: assert 1 == 2\n"
    )
    result = ErrorParser.parse_failure(trace)
    assert result["strategy_tag"] == "assertion_error"
    assert "tests/test_sample.py" in result["stack"]
    assert result["signature"]


def test_parse_failure_runtime_error():
    try:
        1 / 0
    except ZeroDivisionError:
        trace = traceback.format_exc()
    result = ErrorParser.parse_failure(trace)
    assert result["strategy_tag"] == "zero_division_error"


def test_parse_failure_extracts_tags():
    trace = "Traceback\nValueError: boom"
    report = parse_failure(trace)
    assert report.tags == ["value_error"]
