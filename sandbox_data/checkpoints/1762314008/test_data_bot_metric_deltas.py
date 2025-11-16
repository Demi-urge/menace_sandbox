import types
import sys


# provide minimal stubs for optional dependencies
class DummyGauge:
    def __init__(self, *a, **k):
        pass

    def labels(self, **k):
        return self

    def set(self, value):
        pass


prom = types.SimpleNamespace(Gauge=DummyGauge, CollectorRegistry=lambda: object())
psutil = types.SimpleNamespace(
    disk_io_counters=lambda: types.SimpleNamespace(read_bytes=0, write_bytes=0),
    net_io_counters=lambda: types.SimpleNamespace(bytes_sent=0, bytes_recv=0),
    cpu_percent=lambda: 0.0,
    virtual_memory=lambda: types.SimpleNamespace(percent=0.0),
)
sys.modules.setdefault("prometheus_client", prom)
sys.modules.setdefault("psutil", psutil)

from menace import data_bot as db  # noqa: E402
from menace.self_coding_thresholds import SelfCodingThresholds  # noqa: E402
db.load_sc_thresholds = lambda bot=None, settings=None: SelfCodingThresholds(
    roi_drop=-0.1, error_increase=1.0, test_failure_increase=0.0
)
DataBot = db.DataBot


def test_metric_delta_events():
    events = []

    class Bus:
        def publish(self, topic, event):
            events.append((topic, event))

    db = types.SimpleNamespace(add=lambda rec: None)
    bot = DataBot(db, event_bus=Bus())
    bot.collect("agent", revenue=10.0, expense=0.0, errors=0, tests_failed=0, tests_run=10)
    bot.collect(
        "agent", revenue=5.0, expense=0.0, errors=3, tests_failed=2, tests_run=10
    )

    deltas = [e for e in events if e[0] == "metrics:delta"]
    assert len(deltas) == 2
    second = deltas[-1][1]
    assert second["roi_breach"] and second["error_breach"] and second["test_failure_breach"]
    assert second["delta_roi"] == -5.0
    assert second["delta_errors"] == 3.0
    assert second["delta_tests_failed"] == 2.0
    assert second["roi_baseline"] == 10.0
    assert second["errors_baseline"] == 0.0
    assert second["tests_failed_baseline"] == 0.0
