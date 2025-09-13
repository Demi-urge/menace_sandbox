import yaml


def test_reload_thresholds_persists_and_broadcasts(tmp_path, monkeypatch):
    cfg = tmp_path / "self_coding_thresholds.yaml"
    cfg.write_text(
        """default:\n  roi_drop: -0.1\n  error_increase: 1.0\n  test_failure_increase: 0.0\nbots: {}\n""",
        encoding="utf-8",
    )

    import menace.self_coding_thresholds as sct

    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg)

    class DummyBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def publish(self, topic: str, payload: object) -> None:
            self.events.append((topic, payload))

        def subscribe(self, _topic: str, _fn):
            pass

    from menace.data_bot import DataBot, MetricsDB
    from menace.threshold_service import ThresholdService

    bus = DummyBus()
    svc = ThresholdService(event_bus=bus)
    db = MetricsDB(path=tmp_path / "metrics.db")
    bot = DataBot(db=db, event_bus=bus, threshold_service=svc, start_server=False)
    bot.reload_thresholds("alpha")

    data = yaml.safe_load(cfg.read_text())
    assert "alpha" in data.get("bots", {})
    assert any(topic == "thresholds:updated" for topic, _ in bus.events)
