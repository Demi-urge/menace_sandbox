import time

import pytest

import sandbox.preseed_bootstrap as bootstrap
from sandbox.preseed_bootstrap import _run_with_timeout


@pytest.fixture(autouse=True)
def reset_bootstrap_timeline():
    bootstrap.BOOTSTRAP_STEP_TIMELINE.clear()
    bootstrap._BOOTSTRAP_TIMELINE_START = None
    bootstrap.BOOTSTRAP_PROGRESS["last_step"] = "not-started"
    yield
    bootstrap.BOOTSTRAP_STEP_TIMELINE.clear()
    bootstrap._BOOTSTRAP_TIMELINE_START = None
    bootstrap.BOOTSTRAP_PROGRESS["last_step"] = "not-started"


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


def test_run_with_timeout_reports_timeline(capsys):
    bootstrap._mark_bootstrap_step("phase-one")
    time.sleep(0.01)
    bootstrap._mark_bootstrap_step("phase-two")

    with pytest.raises(TimeoutError):
        _run_with_timeout(
            lambda: time.sleep(0.2),
            timeout=0.05,
            description="timeline-check",
        )

    captured = capsys.readouterr().out
    assert "active_step=phase-two" in captured
    assert "[bootstrap-timeout][timeline] phase-one" in captured
    assert "[bootstrap-timeout][timeline] phase-two" in captured
