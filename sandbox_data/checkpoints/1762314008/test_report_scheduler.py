import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.report_generation_bot as rgb
import menace.data_bot as db
from pathlib import Path


def test_schedule_triggers(monkeypatch, tmp_path: Path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    reporter = rgb.ReportGenerationBot(db=mdb, reports_dir=tmp_path)

    # patch Celery and APScheduler out
    monkeypatch.setattr(rgb, "Celery", None)
    monkeypatch.setattr(rgb, "BackgroundScheduler", None)

    called = {}

    def fake_compile(opts, limit=None, start=None, end=None):
        called["start"] = start
        called["end"] = end
        return tmp_path / "r.txt"

    monkeypatch.setattr(reporter, "compile_report", fake_compile)

    class FakeTimer:
        def __init__(self, interval, func):
            self.func = func
            self.daemon = True

        def start(self):
            pass

    monkeypatch.setattr(rgb.threading, "Timer", FakeTimer)

    # preset last report
    reporter.last_report_file.write_text("2000-01-01T00:00:00")
    reporter.schedule(rgb.ReportOptions(metrics=[]), interval=1)

    reporter.timer.func()
    assert called["start"] == "2000-01-01T00:00:00"
    assert called["end"]
