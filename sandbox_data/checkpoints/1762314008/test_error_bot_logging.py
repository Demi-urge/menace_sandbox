import logging
import os
import sys
import types

os.environ["MENACE_LIGHT_IMPORTS"] = "1"
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_mod)
for name in ["env_config", "httpx", "requests", "numpy"]:
    mod = types.ModuleType(name)
    if name == "env_config":
        mod.DATABASE_URL = "sqlite:///:memory:"
    sys.modules.setdefault(name, mod)
crypto = types.ModuleType("cryptography")
haz = types.ModuleType("cryptography.hazmat.primitives.asymmetric.ed25519")
haz.Ed25519PrivateKey = object
haz.Ed25519PublicKey = object
sys.modules.setdefault("cryptography", crypto)
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("cryptography.hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("cryptography.hazmat.primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("cryptography.hazmat.primitives.asymmetric"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", haz)
serialization_mod = types.ModuleType("cryptography.hazmat.primitives.serialization")
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization_mod)

from menace.error_bot import ErrorBot, ErrorDB

class BadBus:
    def subscribe(self, *a, **k):
        raise RuntimeError("fail")
    def publish(self, *a, **k):
        raise RuntimeError("fail")

class BadMem:
    def subscribe(self, *a, **k):
        raise RuntimeError("fail")


class DummyBuilder:
    def refresh_db_weights(self):
        pass


def test_error_bot_logs_failures(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    bus = BadBus()
    bot = ErrorBot(
        ErrorDB(tmp_path / "e.db", event_bus=bus),
        event_bus=bus,
        memory_mgr=BadMem(),
        context_builder=DummyBuilder(),
    )
    bot.db._publish("t", {})
    text = caplog.text
    assert "event bus subscription failed" in text
    assert "memory subscription failed" in text
    assert "publish failed" in text
