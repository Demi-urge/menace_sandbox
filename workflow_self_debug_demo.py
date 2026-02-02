"""Workflow used to trigger and validate self-debug patching."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the correct sum after self-debug patching."""
    return a + b


def run() -> bool:
    """Run the workflow and raise an assertion when the bug is present."""
    expected = 3
    actual = add(1, 2)
    assert actual == expected, "contract violation: addition mismatch"
    return True


def main() -> bool:
    """Alias for workflow runners expecting a main callable."""
    return run()
