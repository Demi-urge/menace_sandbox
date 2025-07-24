from datetime import datetime, timedelta

from menace.weekly_metrics_bot import WeeklyMetricsBot
from menace.data_bot import MetricsDB, MetricRecord


def test_compile_message(tmp_path, monkeypatch):
    db = MetricsDB(tmp_path / "m.db")
    now = datetime.utcnow()
    # this week metrics
    for i in range(3):
        ts = (now - timedelta(days=i)).isoformat()
        db.add(MetricRecord(bot="A", cpu=0.0, memory=0.0, response_time=0.0,
                            disk_io=0.0, net_io=0.0, errors=0,
                            revenue=10.0, expense=5.0, ts=ts))
    db.add(MetricRecord(bot="B", cpu=0.0, memory=0.0, response_time=0.0,
                        disk_io=0.0, net_io=0.0, errors=0,
                        revenue=4.0, expense=1.0, ts=now.isoformat()))
    # previous week metrics
    for i in range(2):
        ts = (now - timedelta(days=8 + i)).isoformat()
        db.add(MetricRecord(bot="A", cpu=0.0, memory=0.0, response_time=0.0,
                            disk_io=0.0, net_io=0.0, errors=0,
                            revenue=5.0, expense=2.5, ts=ts))

    captured = {}

    def fake_alert(msg: str, webhook: str) -> bool:
        captured['msg'] = msg
        captured['webhook'] = webhook
        return True

    monkeypatch.setattr("menace.weekly_metrics_bot.send_discord_alert", fake_alert)
    bot = WeeklyMetricsBot(db, webhook_url="http://example.com")
    bot.send_weekly_report()
    assert "Profit" in captured['msg']
    assert captured['webhook'] == "http://example.com"


def test_env_webhook_used(tmp_path, monkeypatch):
    db = MetricsDB(tmp_path / "m.db")
    monkeypatch.setenv("WEEKLY_METRICS_WEBHOOK", "http://env.example")

    import importlib
    import menace.weekly_metrics_bot as wmb
    importlib.reload(wmb)

    captured = {}

    def fake_alert(msg: str, webhook: str) -> bool:
        captured["webhook"] = webhook
        return True

    monkeypatch.setattr(wmb, "send_discord_alert", fake_alert)
    bot = wmb.WeeklyMetricsBot(db)
    bot.send_weekly_report()
    assert captured["webhook"] == "http://env.example"
