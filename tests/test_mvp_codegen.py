import pytest

from mvp_codegen import run_generation


FALLBACK_MESSAGE = "internal error: code generation failed"
FALLBACK_SCRIPT = f'print("{FALLBACK_MESSAGE}")'


@pytest.mark.parametrize(
    "task",
    [
        None,
        "not-a-dict",
        {},
        {"objective": ""},
        {"objective": "   "},
        {"objective": 123},
        {"objective": "do the thing"},
    ],
)
def test_run_generation_always_returns_string(task):
    result = run_generation(task)  # type: ignore[arg-type]

    assert isinstance(result, str)
    assert "print(" in result


def test_run_generation_fallback_on_wrapper_error():
    def wrapper(_prompt):
        raise RuntimeError("boom")

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = run_generation(task)

    assert FALLBACK_MESSAGE in result


@pytest.mark.parametrize("unsafe_code", ["import os\nprint('hi')", "import subprocess\nprint('hi')"])
def test_run_generation_rejects_unsafe_imports(unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = run_generation(task)

    assert result == FALLBACK_SCRIPT


def test_run_generation_sanitizes_code_fences():
    def wrapper(_prompt):
        return "```python\nprint('hi')\n```"

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = run_generation(task)

    assert "```" not in result
    assert result.strip() == "print('hi')"


def test_run_generation_truncates_or_falls_back_on_oversized_output():
    def wrapper(_prompt):
        return ("print('a')\n" * 500).rstrip()

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = run_generation(task)

    assert isinstance(result, str)
    assert result == FALLBACK_SCRIPT or len(result) <= 4000
