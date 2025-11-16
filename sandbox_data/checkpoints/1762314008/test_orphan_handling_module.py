import pytest

orphan_handling = pytest.importorskip("menace.self_improvement.orphan_handling")


def test_integrate_orphans_requires_runner():
    with pytest.raises(RuntimeError):
        orphan_handling.integrate_orphans()


def test_post_round_scan_requires_runner():
    with pytest.raises(RuntimeError):
        orphan_handling.post_round_orphan_scan()
