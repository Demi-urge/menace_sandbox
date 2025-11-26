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


def test_run_with_timeout_emits_metadata(capsys):
    def sleepy_task(duration: float) -> None:
        time.sleep(duration)

    with pytest.raises(TimeoutError):
        _run_with_timeout(
            sleepy_task,
            timeout=0.05,
            bootstrap_deadline=time.monotonic() + 5.0,
            description="metadata-check",
            duration=0.2,
        )

    captured = capsys.readouterr().out
    assert "[bootstrap-timeout][metadata]" in captured
    assert "metadata-check" in captured
    assert "[bootstrap-timeout][thread=" in captured
