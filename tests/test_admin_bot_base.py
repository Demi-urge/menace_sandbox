import pytest
from types import SimpleNamespace

import os
import sys
import types

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
from menace.database_router import DBResult


class DummyRouter:
    def __init__(self):
        self.calls = []

    def query_all(self, term: str, **options):
        self.calls.append((term, options))
        return DBResult(code=[], bots=[], info=[], memory=[], menace=[])


def test_health_check_term_override():
    class CustomBot(AdminBotBase):
        def health_check_term(self) -> str:
            return "custom:check"

    router = DummyRouter()
    CustomBot(db_router=router)
    assert router.calls == [("custom:check", {})]


def test_query_passes_options_and_returns_result():
    router = DummyRouter()
    bot = AdminBotBase(db_router=router, perform_health_check=False)
    res = bot.query("hello", requesting_bot="tester")
    assert router.calls == [("hello", {"requesting_bot": "tester"})]
    assert isinstance(res, DBResult)


def test_skip_health_check():
    router = DummyRouter()
    AdminBotBase(db_router=router, perform_health_check=False)
    assert router.calls == []
