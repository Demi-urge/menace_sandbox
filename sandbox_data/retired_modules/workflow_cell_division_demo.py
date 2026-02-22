"""Minimal workflow module to exercise self-debug patch promotion."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the sum after self-debug promotion."""
    return a + b


def run() -> bool:
    """Validate the self-debugged addition workflow."""
    expected = 3
    actual = add(1, 2)
    assert actual == expected, "contract violation: addition mismatch"
    return True


def main() -> bool:
    """Alias for workflow runners expecting a main callable."""
    return run()
