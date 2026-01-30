"""Tiny demo for deterministic TypeError."""

def add_values(left, right):
    return left + right

result = add_values("1", 2)
print(result)
