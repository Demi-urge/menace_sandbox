def add(a, b):
    return a - b


def _self_test() -> None:
    expected = 5
    actual = add(2, 3)
    if actual != expected:
        raise AssertionError(f"add(2, 3) should be {expected}, got {actual}")


if __name__ == "__main__":
    _self_test()
    print(add(2, 3))
