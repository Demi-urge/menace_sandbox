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

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

spec = importlib.util.spec_from_file_location(
    "menace.self_validation_dashboard",
    ROOT / "self_validation_dashboard.py",  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
svd_mod = importlib.util.module_from_spec(spec)
sys.modules["menace.self_validation_dashboard"] = svd_mod
spec.loader.exec_module(svd_mod)
SelfValidationDashboard = svd_mod.SelfValidationDashboard
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


def test_cli_generates_file(monkeypatch, tmp_path):
    import menace.self_validation_dashboard as svd

    class DummyBot:
        def long_term_roi_trend(self, limit: int = 200) -> float:
            return 0.2

    class DummyForecaster:
        def __init__(self, db):
            pass

        def forecast(self) -> float:
            return 0.1

    class DummyUpdater:
        def _outdated(self):
            return []

    monkeypatch.setattr(svd, "DataBot", lambda: DummyBot())
    monkeypatch.setattr(svd, "ErrorForecaster", lambda db: DummyForecaster(db))
    monkeypatch.setattr(svd, "DependencyUpdater", lambda: DummyUpdater())

    out = tmp_path / "dash.json"
    svd.cli(["--interval", "0", "--output", str(out)])
    assert out.exists()
