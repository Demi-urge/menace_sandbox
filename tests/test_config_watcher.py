import time
import pytest

from menace import config


# Skip the entire module if watchdog is unavailable
if config.Observer is None:  # pragma: no cover - optional dependency
    pytest.skip("watchdog not installed", allow_module_level=True)


class DummyBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, topic: str, payload: object) -> None:
        self.events.append((topic, payload))


def test_watcher_triggers_reload(tmp_path, monkeypatch):
    # Ensure no watcher is running from previous tests
    config.shutdown()

    settings = tmp_path / "settings.yaml"
    profile = tmp_path / "dev.yaml"
    settings.write_text(
        f"""
paths:
  data_dir: {tmp_path}
  log_dir: {tmp_path}
thresholds:
  error: 0.1
  alert: 0.2
api_keys:
  openai: key
  serp: key
logging:
  verbosity: DEBUG
vector:
  dimensions: 1
  distance_metric: cosine
bot:
  learning_rate: 0.1
  epsilon: 0.1
""",
        encoding="utf-8",
    )
    profile.write_text("", encoding="utf-8")

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_SETTINGS_FILE", settings)
    config._MODE = "dev"
    config._CONFIG_PATH = None
    config._OVERRIDES = {}
    config.CONFIG = None
    bus = DummyBus()
    config.set_event_bus(bus)
    config.get_config(watch=True)

    profile.write_text("logging:\n  verbosity: INFO\n", encoding="utf-8")

    # wait until reload event is observed
    for _ in range(50):
        if any(topic == "config.reload" for topic, _ in bus.events):
            break
        time.sleep(0.1)

    assert any(topic == "config.reload" for topic, _ in bus.events)

    # cleanup watcher
    config.shutdown()
    config.set_event_bus(None)
