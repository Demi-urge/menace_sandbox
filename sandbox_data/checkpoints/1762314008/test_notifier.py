import os
import sys
import logging
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
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
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
nx_mod = types.ModuleType("networkx")
nx_mod.DiGraph = object
sys.modules.setdefault("networkx", nx_mod)
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass
engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

import menace.watchdog as wd


def _stub_requests(monkeypatch):
    monkeypatch.setattr(
        wd,
        "requests",
        types.SimpleNamespace(post=lambda *a, **k: types.SimpleNamespace(status_code=200)),
        raising=False,
    )


def test_notify_logs_attachment_error_slack(tmp_path, monkeypatch, caplog):
    n = wd.Notifier(slack_webhook="http://example.com")
    _stub_requests(monkeypatch)
    missing = tmp_path / "missing.txt"
    caplog.set_level(logging.ERROR)
    n.notify("msg", attachments=[str(missing)])
    assert "Failed to read attachment" in caplog.text


def test_notify_logs_attachment_error_telegram(tmp_path, monkeypatch, caplog):
    n = wd.Notifier(telegram_token="t", telegram_chat_id="c")
    _stub_requests(monkeypatch)
    missing = tmp_path / "missing.txt"
    caplog.set_level(logging.ERROR)
    n.notify("msg", attachments=[str(missing)])
    assert "Failed to read attachment" in caplog.text
