import pytest

import mvp_codegen


def _build_expected_fallback(objective: str, constraints: list[str]) -> str:
    objective_text = objective.strip() if isinstance(objective, str) else ""
    constraint_texts = [
        item.strip() for item in constraints if isinstance(item, str) and item.strip()
    ]
    constraints_line = ", ".join(constraint_texts) if constraint_texts else "none"
    lines = [
        '"""Deterministic placeholder script generated without a model wrapper."""',
        "",
        "def main() -> None:",
        f"    objective = {objective_text!r}",
        f"    constraints = {constraints_line!r}",
        "    summary_lines = [",
        '        "Placeholder execution: no model wrapper provided.",',
        '        f"Objective: {objective}",',
        '        f"Constraints: {constraints}",',
        '        "Status: completed safe placeholder run.",',
        "    ]",
        "    print(\"\\n\".join(summary_lines))",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
    ]
    return "\n".join(lines)


def _normalize_constraints(constraints=None) -> list[str]:
    normalized: list[str] = []
    if isinstance(constraints, list):
        for item in constraints:
            if isinstance(item, str):
                text = item.strip()
            elif item is None:
                text = ""
            else:
                try:
                    text = str(item).strip()
                except Exception:
                    text = ""
            if text:
                normalized.append(text)
    elif constraints is not None:
        if isinstance(constraints, str):
            text = constraints.strip()
        else:
            try:
                text = str(constraints).strip()
            except Exception:
                text = ""
        if text:
            normalized.append(text)
    return normalized


FALLBACK_SCRIPT = _build_expected_fallback("unspecified objective", [])


def _expected_fallback(objective: str, constraints=None) -> str:
    normalized = _normalize_constraints(constraints)
    return _build_expected_fallback(objective, normalized)


@pytest.mark.parametrize(
    "task",
    [
        None,
        "not a dict",
        123,
        [],
    ],
)
def test_run_generation_fallback_on_non_dict_task(task):
    result = mvp_codegen.run_generation(task)

    assert result == FALLBACK_SCRIPT


@pytest.mark.parametrize(
    "task",
    [
        {"model_wrapper": lambda _prompt: "print('hi')"},
        {"objective": "", "model_wrapper": lambda _prompt: "print('hi')"},
        {"objective": "   ", "model_wrapper": lambda _prompt: "print('hi')"},
        {"objective": 123, "model_wrapper": lambda _prompt: "print('hi')"},
    ],
)
def test_run_generation_fallback_on_empty_or_invalid_objective(task):
    result = mvp_codegen.run_generation(task)

    assert result == FALLBACK_SCRIPT


def test_run_generation_fallback_on_wrapper_error():
    def wrapper(_prompt):
        raise RuntimeError("boom")

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_fallback_on_str_failure():
    class BadString:
        def __str__(self):
            raise ValueError("nope")

    def wrapper(_prompt):
        return BadString()

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_passes_timeout_to_wrapper():
    received = {}

    def wrapper(_prompt, timeout_s=None):
        received["timeout"] = timeout_s
        return "print('hi')"

    task = {"objective": "do the thing", "timeout_s": 12, "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result.strip() == "print('hi')"
    assert received["timeout"] == 12


def test_run_generation_fallback_on_timeout_error():
    def wrapper(_prompt, timeout_s=None):
        raise TimeoutError("too slow")

    task = {"objective": "do the thing", "timeout_s": 3, "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


@pytest.mark.parametrize(
    "task",
    [
        {"objective": "do the thing"},
        {"objective": "do the thing", "constraints": ["fast", "safe"]},
        {"objective": "do the thing", "constraints": "just a string"},
    ],
)
def test_run_generation_uses_model_wrapper(task):
    calls = []

    def wrapper(prompt):
        calls.append(prompt)
        return "print('hi')"

    task = {**task, "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert isinstance(result, str)
    assert result.strip() == "print('hi')"
    assert calls


@pytest.mark.parametrize(
    "task",
    [
        {"objective": "do the thing"},
        {"objective": "do the thing", "constraints": ["fast", "safe"]},
        {"objective": "do the thing", "constraints": "just a string"},
    ],
)
def test_run_generation_falls_back_without_wrapper(task):
    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback(task["objective"], task.get("constraints"))


@pytest.mark.parametrize(
    "task",
    [
        {"objective": "do the thing", "model_wrapper": None},
        {"objective": "do the thing", "constraints": ["fast", "safe"], "model_wrapper": None},
        {"objective": "do the thing", "constraints": "just a string", "model_wrapper": None},
        {"objective": "do the thing", "model_wrapper": "not callable"},
    ],
)
def test_run_generation_falls_back_with_non_callable_wrapper(task):
    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback(task["objective"], task.get("constraints"))


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
def test_run_generation_rejects_unsafe_imports(unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_rejects_io_open_usage():
    def wrapper(_prompt):
        return "import io\nio.open('data.txt')"

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_rejects_io_open_alias_usage():
    def wrapper(_prompt):
        return "import io as i\ni.open('data.txt')"

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import builtins as b\nprint(b.open('x'))",
        "import builtins as b\nprint(b)",
        "from builtins import open as o\nprint(o('x'))",
        "__builtins__['open']('x')",
        "__builtins__['__import__']('os')",
        "getattr(__builtins__, 'open')('x', 'w')",
        "setattr(__builtins__, 'open', None)",
        "globals()['open']('x')",
        "import_module('os')",
    ],
)
def test_run_generation_rejects_builtins_alias_bypasses(unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


@pytest.mark.parametrize(
    "unsafe_code",
    [
        "import builtins\nbuiltins.__dict__['open']('x')",
        "globals()['__builtins__']['open']('x')",
        "import builtins\nvars(builtins)['open']('x')",
    ],
)
def test_run_generation_rejects_builtins_dict_access(unsafe_code):
    def wrapper(_prompt):
        return unsafe_code

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_rejects_importlib_dynamic_import():
    def wrapper(_prompt):
        return "import importlib\nprint(importlib.import_module('os'))"

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert result == _expected_fallback("do the thing")


def test_run_generation_sanitizes_code_fences():
    def wrapper(_prompt):
        return "```python\nprint('hi')\n```"

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert "```" not in result
    assert result.strip() == "print('hi')"


def test_run_generation_truncates_or_falls_back_on_oversized_output():
    def wrapper(_prompt):
        return ("print('a')\n" * 500).rstrip()

    task = {"objective": "do the thing", "model_wrapper": wrapper}

    result = mvp_codegen.run_generation(task)

    assert isinstance(result, str)
    assert result == _expected_fallback("do the thing") or len(result) <= 4000
