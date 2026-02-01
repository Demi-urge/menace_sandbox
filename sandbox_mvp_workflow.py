def add(a: int, b: int) -> int:
    return a - b


def run() -> None:
    expected = 5
    actual = add(2, 3)
    if actual != expected:
        raise AssertionError(f"add(2, 3) should be {expected}, got {actual}")
    print("workflow ok")
