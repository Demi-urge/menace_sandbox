import pytest

from mvp_codegen import run_generation


FALLBACK_SCRIPT = 'print("internal error: code generation failed")'


def test_run_generation_returns_fallback_for_invalid_task():
    assert run_generation("not-a-dict") == FALLBACK_SCRIPT
    assert run_generation({}) == FALLBACK_SCRIPT


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timeout")),
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_run_generation_returns_fallback_on_wrapper_failure(wrapper):
    task = {
        "objective": "say hi",
        "model_wrapper": wrapper,
        "timeout_s": 0.01,
    }

    assert run_generation(task) == FALLBACK_SCRIPT


@pytest.mark.parametrize(
    "generated",
    [
        "import os\nprint('nope')",
        "from builtins import __import__\nprint('nope')",
    ],
)
def test_run_generation_rejects_banned_imports(generated):
    task = {
        "objective": "write safe code",
        "model_wrapper": lambda *_args, **_kwargs: generated,
    }

    assert run_generation(task) == FALLBACK_SCRIPT


def test_run_generation_truncates_oversized_output():
    line = "print('ok')\n"
    oversized_output = line * 600
    task = {
        "objective": "emit many prints",
        "model_wrapper": lambda *_args, **_kwargs: oversized_output,
    }

    result = run_generation(task)

    assert result != FALLBACK_SCRIPT
    assert len(result) <= 4000
    assert result.strip().endswith("print('ok')")


def test_run_generation_returns_valid_output_string():
    task = {
        "objective": "say hello",
        "model_wrapper": lambda *_args, **_kwargs: "print('hello')",
    }

    assert run_generation(task) == "print('hello')"
