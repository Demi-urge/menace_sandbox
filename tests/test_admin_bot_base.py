import pytest
from types import SimpleNamespace

import os
import sys
import types
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Provide lightweight stubs for cryptography
# Minimal cryptography stubs
crypto = types.ModuleType("cryptography")
hazmat = types.SimpleNamespace()
primitives = types.SimpleNamespace()
asymmetric = types.SimpleNamespace(
    ed25519=types.SimpleNamespace(
        Ed25519PrivateKey=object, Ed25519PublicKey=object
    )
)
serialization = types.SimpleNamespace(
    Encoding=object, PublicFormat=object, PrivateFormat=object, NoEncryption=object
)
primitives.asymmetric = asymmetric
primitives.serialization = serialization
hazmat.primitives = primitives
crypto.hazmat = hazmat
sys.modules.setdefault("cryptography", crypto)
sys.modules.setdefault("cryptography.hazmat", hazmat)
sys.modules.setdefault("cryptography.hazmat.primitives", primitives)
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", asymmetric)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    asymmetric.ed25519,
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization",
    serialization,
)

from menace.admin_bot_base import AdminBotBase


class DummyRouter:
    def __init__(self):
        self.calls = []

    def get_connection(self, table_name: str):
        self.calls.append(table_name)
        class DummyConn:
            def execute(self, _sql):
                return None

        return DummyConn()


def test_health_check_uses_router_connection():
    router = DummyRouter()

    AdminBotBase(db_router=router)

    assert router.calls == ["bots"]


def test_query_is_not_implemented():
    bot = AdminBotBase(db_router=DummyRouter(), perform_health_check=False)
    with pytest.raises(NotImplementedError):
        bot.query("anything")


def test_skip_health_check():
    router = DummyRouter()
    AdminBotBase(db_router=router, perform_health_check=False)
    assert router.calls == []
