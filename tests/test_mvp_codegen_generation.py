import pytest

import mvp_codegen
from mvp_codegen import run_generation


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

    assert run_generation(task) == _expected_fallback("say hi")


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

    assert run_generation(task) == _expected_fallback("write safe code")


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
