import logging
import time
from pathlib import Path

import pytest

from menace_sandbox.self_coding_thresholds import _read_text_with_timeout


def test_bootstrap_read_returns_cached_without_waiting(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)

    cfg_path = tmp_path / "thresholds.yaml"
    cfg_path.write_text("slow", encoding="utf-8")

    read_calls: list[float] = []

    class SlowPath(type(cfg_path)):
        def read_text(self, encoding: str = "utf-8") -> str:  # pragma: no cover - patched
            read_calls.append(time.perf_counter())
            time.sleep(0.5)
            return "slow"

    slow_path = SlowPath(cfg_path)

    start = time.perf_counter()
    result = _read_text_with_timeout(
        slow_path,
        timeout=1.0,
        bootstrap_mode=True,
        fallback_text="cached",
    )
    duration = time.perf_counter() - start

    assert result == "cached"
    assert not read_calls, "bootstrap fast path should avoid file IO"
    assert duration < 0.1, f"bootstrap read waited too long ({duration:.3f}s)"
    assert any(
        "bootstrap read" in record.getMessage() for record in caplog.records
    ), "expected bootstrap fast-path log entry"
