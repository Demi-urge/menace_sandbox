"""Utility functions demonstrating basic numerical algorithms."""

from __future__ import annotations

from typing import Iterable, List


def join_range(start: int, end: int, *, sep: str = " ") -> str:
    """Return a string containing the numbers ``start`` through ``end``."""

    if end < start:
        rng = range(start, end - 1, -1)
    else:
        rng = range(start, end + 1)
    return sep.join(str(i) for i in rng)


def print_ten() -> None:
    """Print the numbers from 1 through 10 separated by spaces."""

    print(join_range(1, 10))


def fibonacci(n: int) -> int:
    """Return the ``n``-th Fibonacci number using an iterative algorithm."""

    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def print_eleven() -> None:
    """Print the 11th Fibonacci number."""

    print(str(fibonacci(11)))


def factorial(n: int) -> int:
    """Compute ``n`` factorial without using the ``math`` helpers."""

    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def print_twelve() -> None:
    """Print the factorial of twelve."""

    print(str(factorial(12)))


__all__ = [
    "print_ten",
    "print_eleven",
    "print_twelve",
    "join_range",
    "fibonacci",
    "factorial",
]
