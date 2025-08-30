import traceback

from error_parser import ErrorParser


def test_parse_pytest_assertion():
    trace = (
        "tests/test_sample.py:3: in test_example\n"
        "    assert 1 == 2\n"
        "E   AssertionError: assert 1 == 2\n"
    )
    result = ErrorParser.parse(trace)
    assert "tests/test_sample.py" in result["files"]
    assert result["error_type"] == "AssertionError"
    assert "test" in result["tags"]


def test_parse_runtime_error():
    try:
        1 / 0
    except ZeroDivisionError:
        trace = traceback.format_exc()
    result = ErrorParser.parse(trace)
    assert any(path.endswith("test_error_parser.py") for path in result["files"])
    assert result["error_type"] == "ZeroDivisionError"
    assert "runtime" in result["tags"]
