import menace.prime_utils as pu


def test_is_prime():
    assert pu.is_prime(2)
    assert pu.is_prime(13)
    assert not pu.is_prime(1)
    assert not pu.is_prime(4)
