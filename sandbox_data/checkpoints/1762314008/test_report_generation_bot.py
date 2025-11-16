import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from unittest.mock import patch
import menace.report_generation_bot as rgb
import menace.data_bot as db


def _populate(tmp_path):
    metrics_db = db.MetricsDB(tmp_path / "m.db")
    metrics_db.add(db.MetricRecord(bot="a", cpu=1.0, memory=2.0, response_time=0.1, disk_io=1.0, net_io=1.0, errors=0))
    metrics_db.add(db.MetricRecord(bot="b", cpu=2.0, memory=3.0, response_time=0.2, disk_io=2.0, net_io=2.0, errors=1))
    return metrics_db


def test_compile_report(tmp_path):
    mdb = _populate(tmp_path)
    bot = rgb.ReportGenerationBot(db=mdb, reports_dir=tmp_path)
    opts = rgb.ReportOptions(metrics=["cpu", "memory"], title="Report")
    report = bot.compile_report(opts)
    assert report.exists()
    content = report.read_text()
    assert "Metrics summary" in content


@patch("smtplib.SMTP")
def test_send_report(mock_smtp, tmp_path):
    mdb = _populate(tmp_path)
    bot = rgb.ReportGenerationBot(db=mdb, reports_dir=tmp_path)
    opts = rgb.ReportOptions(metrics=["cpu"], title="Report", recipients=["a@example.com"])
    report = bot.compile_report(opts)
    bot.send_report(report, ["a@example.com"])
    mock_smtp.assert_called_once()
