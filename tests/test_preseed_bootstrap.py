"""Tests for sandbox.preseed_bootstrap helpers."""

import time

import pytest


def test_run_with_timeout_includes_last_step() -> None:
    """Timeout errors should surface the last recorded bootstrap step."""

    from sandbox import preseed_bootstrap as pb

    pb.BOOTSTRAP_PROGRESS["last_step"] = "prepare_pipeline"

    def _blocker() -> None:
        time.sleep(0.2)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            pb._run_with_timeout(  # type: ignore[attr-defined]
                _blocker,
                timeout=0.05,
                bootstrap_deadline=time.monotonic() + 5,
                description="prepare_pipeline_for_bootstrap",
                abort_on_timeout=True,
            )
        assert "last_step=prepare_pipeline" in str(excinfo.value)
    finally:
        pb.BOOTSTRAP_PROGRESS.pop("last_step", None)

