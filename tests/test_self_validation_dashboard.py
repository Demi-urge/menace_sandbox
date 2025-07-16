from pathlib import Path
import sys
import types

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
jinja = types.ModuleType("jinja2")
jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja)

from menace.self_validation_dashboard import SelfValidationDashboard
import json
import threading


class DummyDataBot:
    def long_term_roi_trend(self, limit: int = 200) -> float:
        return 0.1


def test_generate_report(tmp_path):
    dash = SelfValidationDashboard(DummyDataBot())
    path = tmp_path / "rep.json"
    dest = dash.generate_report(path)
    assert dest.exists()
    data = dest.read_text()
    assert "roi_trend" in data


def test_scheduler_aggregates(monkeypatch, tmp_path):
    dash = SelfValidationDashboard(DummyDataBot(), history_file=tmp_path / "hist.json")

    class FakeTimer:
        def __init__(self, interval, func):
            self.func = func
            self.daemon = True

        def start(self):
            pass

    monkeypatch.setattr(threading, "Timer", FakeTimer)

    dash.schedule(tmp_path / "dash.json", interval=1)
    dash.timer.func()
    dash.timer.func()

    hist = json.loads((tmp_path / "hist.json").read_text())
    assert len(hist) == 2
    data = json.loads((tmp_path / "dash.json").read_text())
    assert data.get("aggregates", {}).get("roi_trend_avg") == 0.1
