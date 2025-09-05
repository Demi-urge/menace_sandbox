import sys
import traceback
from pathlib import Path

from error_parser import ErrorParser, parse_failure


def test_parse_failure_pytest_assertion():
    trace = (
        "tests/test_sample.py:3: in test_example\n"  # path-ignore
        "    assert 1 == 2\n"
        "E   AssertionError: assert 1 == 2\n"
    )
    result = ErrorParser.parse_failure(trace)
    assert result["strategy_tag"] == "assertion_error"
    assert "tests/test_sample.py" in result["stack"]  # path-ignore
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


def test_parse_returns_target_region(tmp_path):
    mod = tmp_path / "sample.py"  # path-ignore
    mod.write_text("def fail():\n    raise RuntimeError('x')\nfail()\n")
    try:
        code = compile(mod.read_text(), str(mod), "exec")
        exec(code, {})
    except Exception:
        trace = traceback.format_exc()
    result = ErrorParser.parse(trace)
    region = result.get("target_region")
    assert region is not None
    assert Path(region.filename) == mod
    assert region.function == "fail"


def test_parse_failure_nested_exception():
    def inner():
        raise ValueError("inner")

    def outer():
        try:
            inner()
        except ValueError as exc:
            raise RuntimeError("outer") from exc

    try:
        outer()
    except RuntimeError:
        trace = traceback.format_exc()

    result = ErrorParser.parse_failure(trace)
    assert Path(result["file"]).resolve() == Path(__file__).resolve()
    assert result["function"] == "outer"
    expected_line = outer.__code__.co_firstlineno + 4  # line with raise RuntimeError
    assert result["line"] == str(expected_line)


def test_parse_failure_multifile_trace(tmp_path):
    mod_inner = tmp_path / "inner.py"  # path-ignore
    mod_inner.write_text("def boom():\n    raise ValueError('x')\n")

    mod_outer = tmp_path / "outer.py"  # path-ignore
    mod_outer.write_text(
        "import inner\n\n"
        "def run():\n"
        "    inner.boom()\n\n"
        "run()\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        try:
            __import__("outer")
        except Exception:
            trace = traceback.format_exc()
    finally:
        sys.path.pop(0)
        sys.modules.pop("inner", None)
        sys.modules.pop("outer", None)

    result = ErrorParser.parse_failure(trace)
    assert Path(result["file"]) == mod_inner
    assert result["line"] == "2"
    assert result["function"] == "boom"
