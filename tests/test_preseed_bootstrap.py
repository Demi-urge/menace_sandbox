import time

import pytest

from sandbox.preseed_bootstrap import _run_with_timeout


def test_run_with_timeout_respects_requested_timeout():
    """Ensure bootstrap helpers don't stretch timeouts to the overall deadline."""

    requested_timeout = 0.1
    deadline = time.monotonic() + 10.0

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        _run_with_timeout(
            lambda: time.sleep(0.5),
            timeout=requested_timeout,
            bootstrap_deadline=deadline,
            description="sleepy",
        )

    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
