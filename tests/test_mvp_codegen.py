import pytest

import mvp_codegen


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
    result = mvp_codegen.run_generation(task)  # type: ignore[arg-type]

    assert isinstance(result, str)
    assert "print(" in result


def test_run_generation_fallback_on_wrapper_error(monkeypatch):
    def wrapper(_prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert FALLBACK_MESSAGE in result


@pytest.mark.parametrize(
    "task",
    [
        {"objective": "do the thing"},
        {"objective": "do the thing", "constraints": ["fast", "safe"]},
        {"objective": "do the thing", "constraints": "just a string"},
    ],
)
def test_run_generation_uses_internal_wrapper_resolver(monkeypatch, task):
    calls = []

    def wrapper(prompt):
        calls.append(prompt)
        return "print('hi')"

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)

    result = mvp_codegen.run_generation(task)

    assert isinstance(result, str)
    assert result.strip() == "print('hi')"
    assert calls


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import os\nprint('hi')",
        "import subprocess\nprint('hi')",
        "import importlib\nprint('hi')",
        "import ssl\nprint('hi')",
        "import numpy\nprint('hi')",
        "import tempfile\nprint('hi')",
        "import ftplib\nprint('hi')",
    ],
)
def test_run_generation_rejects_unsafe_imports(monkeypatch, unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert result == FALLBACK_SCRIPT


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import builtins as b\nprint(b.open('x'))",
        "import builtins as b\nprint(b)",
        "from builtins import open as o\nprint(o('x'))",
        "__builtins__['open']('x')",
        "import_module('os')",
    ],
)
def test_run_generation_rejects_builtins_alias_bypasses(monkeypatch, unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert result == FALLBACK_SCRIPT


def test_run_generation_rejects_importlib_dynamic_import(monkeypatch):
    def wrapper(_prompt):
        return "import importlib\nprint(importlib.import_module('os'))"

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert result == FALLBACK_SCRIPT


def test_run_generation_sanitizes_code_fences(monkeypatch):
    def wrapper(_prompt):
        return "```python\nprint('hi')\n```"

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert "```" not in result
    assert result.strip() == "print('hi')"


def test_run_generation_truncates_or_falls_back_on_oversized_output(monkeypatch):
    def wrapper(_prompt):
        return ("print('a')\n" * 500).rstrip()

    monkeypatch.setattr(mvp_codegen, "_get_model_wrapper", lambda: wrapper)
    task = {"objective": "do the thing"}

    result = mvp_codegen.run_generation(task)

    assert isinstance(result, str)
    assert result == FALLBACK_SCRIPT or len(result) <= 4000
