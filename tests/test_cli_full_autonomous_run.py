import os
import sys
import types

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

# Minimal Flask stub for MetricsDashboard import
flask_stub = types.ModuleType("flask")


class _DummyFlask:
    def add_url_rule(self, *a, **k):
        pass

    def run(self, host="0.0.0.0", port=0):
        pass


flask_stub.Flask = lambda *a, **k: _DummyFlask()
flask_stub.jsonify = lambda *a, **k: {}
sys.modules.setdefault("flask", flask_stub)

import sandbox_runner.cli as cli

class DummyTracker:
    def __init__(self):
        self.module_deltas = {"m": [0.1]}
        self.metrics_history = {
            "security_score": [0.5],
            "synergy_roi": [0.2],
        }
        self.predicted_roi = [0.3]
        self.actual_roi = [0.25]
        self.predicted_metrics = {"security_score": [0.4]}
        self.actual_metrics = {"security_score": [0.5]}

    def diminishing(self):
        return 0.01

    def rankings(self):
        return [("m", 0.1)]


def test_full_autonomous_run_cli(monkeypatch):
    dummy_preset = {"env": "dev"}
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [dummy_preset])
    calls: list[dict] = []
    trackers: list[DummyTracker] = []

    def fake_capture(preset, args):
        calls.append(preset)
        t = DummyTracker()
        trackers.append(t)
        return t

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    cli.main([
        "--preset-count",
        "1",
        "full-autonomous-run",
        "--max-iterations",
        "1",
        "--roi-cycles",
        "2",
        "--synergy-cycles",
        "2",
    ])

    assert calls == [dummy_preset]
    assert trackers
    tracker = trackers[0]
    assert "synergy_roi" in tracker.metrics_history
    assert tracker.predicted_roi and tracker.actual_roi


def test_full_autonomous_run_dashboard(monkeypatch):
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [{"env": "dev"}])
    monkeypatch.setattr(cli, "_capture_run", lambda p, a: DummyTracker())

    started: dict[str, int] = {}

    class DummyDash:
        def __init__(self, file):
            started["file"] = file

        def run(self, host="0.0.0.0", port=0):
            started["port"] = port

    class DummyThread:
        def __init__(self, target=None, kwargs=None, daemon=None):
            self.target = target
            self.kwargs = kwargs or {}

        def start(self):
            if self.target:
                self.target(**self.kwargs)

    monkeypatch.setattr(cli, "MetricsDashboard", DummyDash)
    monkeypatch.setattr(cli, "Thread", DummyThread)

    cli.main([
        "full-autonomous-run",
        "--max-iterations",
        "1",
        "--dashboard-port",
        "1234",
    ])

    assert started.get("port") == 1234


def test_run_complete_cli(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_capture_run", lambda p, a: calls.append(p) or DummyTracker())

    cli.main([
        "run-complete",
        "{\"env\": \"dev\"}",
        "--max-iterations",
        "1",
    ])

    assert calls == [{"env": "dev"}]


def test_run_complete_dashboard(monkeypatch):
    monkeypatch.setattr(cli, "_capture_run", lambda p, a: DummyTracker())

    started: dict[str, int] = {}

    class DummyDash:
        def __init__(self, file):
            started["file"] = file

        def run(self, host="0.0.0.0", port=0):
            started["port"] = port

    class DummyThread:
        def __init__(self, target=None, kwargs=None, daemon=None):
            self.target = target
            self.kwargs = kwargs or {}

        def start(self):
            if self.target:
                self.target(**self.kwargs)

    monkeypatch.setattr(cli, "MetricsDashboard", DummyDash)
    monkeypatch.setattr(cli, "Thread", DummyThread)

    cli.main([
        "run-complete",
        "{\"env\": \"dev\"}",
        "--max-iterations",
        "1",
        "--dashboard-port",
        "4321",
    ])

    assert started.get("port") == 4321

