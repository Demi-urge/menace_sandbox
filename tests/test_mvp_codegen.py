import ast

import pytest

import local_model_wrapper
import mvp_codegen


FALLBACK_SCRIPT = 'print("internal error")'


def _make_task(objective="Say hi"):
    return {"objective": objective, "model": object(), "tokenizer": object()}


@pytest.mark.parametrize(
    "task",
    [
        None,
        123,
        "not a dict",
        {},
        {"objective": None, "model": object(), "tokenizer": object()},
        {"objective": " ", "model": object(), "tokenizer": object()},
    ],
)
def test_run_generation_returns_str_for_invalid_inputs(task):
    result = mvp_codegen.run_generation(task)  # type: ignore[arg-type]
    assert isinstance(result, str)
    assert result == FALLBACK_SCRIPT


@pytest.mark.parametrize(
    "unsafe_output",
    [
        "import os\nprint('hi')",
        "from subprocess import Popen\nprint('hi')",
        "print(open('secrets.txt').read())",
    ],
)
def test_run_generation_filters_unsafe_output(monkeypatch, unsafe_output):
    def fake_generate(self, *args, **kwargs):
        return unsafe_output

    monkeypatch.setattr(local_model_wrapper.LocalModelWrapper, "generate", fake_generate)

    result = mvp_codegen.run_generation(_make_task())
    assert result == FALLBACK_SCRIPT


def test_run_generation_truncates_oversized_output(monkeypatch):
    line = "print('1234567890')\n"
    oversized_output = line * 201
    assert len(line) == 20
    assert len(oversized_output.encode("utf-8")) > 4000

    def fake_generate(self, *args, **kwargs):
        return oversized_output

    monkeypatch.setattr(local_model_wrapper.LocalModelWrapper, "generate", fake_generate)

    result = mvp_codegen.run_generation(_make_task())
    assert result != FALLBACK_SCRIPT
    assert len(result.encode("utf-8")) <= 4000
    assert result == oversized_output.encode("utf-8")[:4000].decode("utf-8").strip()
    ast.parse(result)


@pytest.mark.parametrize(
    "bad_output",
    [
        "",
        "   \n\n",
        "print('oops'",
    ],
)
def test_run_generation_fallback_on_empty_or_garbled_output(monkeypatch, bad_output):
    def fake_generate(self, *args, **kwargs):
        return bad_output

    monkeypatch.setattr(local_model_wrapper.LocalModelWrapper, "generate", fake_generate)

    result = mvp_codegen.run_generation(_make_task())
    assert result == FALLBACK_SCRIPT


def test_run_generation_fallback_on_wrapper_error(monkeypatch):
    def fake_generate(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(local_model_wrapper.LocalModelWrapper, "generate", fake_generate)

    result = mvp_codegen.run_generation(_make_task())
    assert result == FALLBACK_SCRIPT
