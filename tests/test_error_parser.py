import traceback

from error_parser import parse_failure


def test_parse_pytest_assertion():
    trace = (
        "tests/test_sample.py:3: in test_example\n"
        "    assert 1 == 2\n"
        "E   AssertionError: assert 1 == 2\n"
    )
    report = parse_failure(trace)
    assert report.tags == ["assertion_error"]


def test_parse_runtime_error():
    try:
        1 / 0
    except ZeroDivisionError:
        trace = traceback.format_exc()
    report = parse_failure(trace)
    assert "zero_division_error" in report.tags
