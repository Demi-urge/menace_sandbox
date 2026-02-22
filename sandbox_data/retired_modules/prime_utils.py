"""Utility functions for number theory."""

from __future__ import annotations


def is_prime(n: int) -> bool:
    """Return True if *n* is a prime number."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(n**0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True
