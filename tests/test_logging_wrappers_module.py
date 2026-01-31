import inspect

from logging_wrappers import wrap_with_logging


def add(a: int, b: int = 2) -> int:
    """Add numbers for testing."""
    return a + b


def test_wrap_with_logging_preserves_behavior_and_signature():
    wrapped = wrap_with_logging(add)

    assert wrapped(3) == 5
    assert wrapped(3, b=4) == 7
    assert inspect.signature(wrapped) == inspect.signature(add)


def test_wrap_with_logging_prevents_double_wrapping():
    wrapped = wrap_with_logging(add)
    wrapped_again = wrap_with_logging(wrapped)

    assert wrapped_again is wrapped
