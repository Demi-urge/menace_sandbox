import os
import logging
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.safety_monitor as sm

class FailingBus:
    def __init__(self):
        self.published = []
    def publish(self, topic, event):
        self.published.append((topic, event))
        raise RuntimeError("boom")

def test_flag_logs_publish_error(caplog):
    bus = FailingBus()
    monitor = sm.SafetyMonitor(tester=types.SimpleNamespace(), event_bus=bus)
    caplog.set_level(logging.ERROR)
    for _ in range(monitor.config.fail_threshold):
        monitor._flag("bot")
    assert bus.published
    assert "failed publishing safety flag" in caplog.text

