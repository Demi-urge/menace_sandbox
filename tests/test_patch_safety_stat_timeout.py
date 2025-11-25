import time

from patch_safety import PatchSafety, Path


def test_load_failures_stat_timeout(monkeypatch, tmp_path):
    """load_failures should not hang when stat blocks."""

    probe_file = tmp_path / "failures.jsonl"
    probe_file.write_text("", encoding="utf-8")

    original_stat = Path.stat

    def slow_stat(self, follow_symlinks=True):  # pragma: no cover - timing helper
        if self == probe_file:
            time.sleep(0.5)
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", slow_stat)

    ps = PatchSafety(storage_path=None, failure_db_path=None, refresh_interval=0, stat_timeout=0.05)

    start = time.monotonic()
    result = ps._mtime_with_timeout(probe_file)
    elapsed = time.monotonic() - start

    assert result == 0.0
    assert elapsed < 0.2
