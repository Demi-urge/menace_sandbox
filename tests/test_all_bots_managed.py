"""Ensure all bot classes are decorated with ``@self_coding_managed``.

This test invokes :mod:`tools.check_self_coding_registration` which scans the
repository for classes inheriting from known bot base classes (currently just
``AdminBotBase``) or named with the ``Bot`` suffix.  Any class missing the
``@self_coding_managed`` decorator causes the check to fail, preventing new
coding bots from bypassing registration.
"""

from tools import check_self_coding_registration


def test_all_bots_are_managed() -> None:
    assert check_self_coding_registration.main() == 0
