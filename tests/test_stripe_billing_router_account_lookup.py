import os
import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = PACKAGE_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

# ``stripe_billing_router`` pulls in a wide range of optional infrastructure
# modules.  For the focused unit tests below we stub the handful that are
# accessed during import so we avoid dragging in the entire runtime stack.
for name in (
    "billing",
    "billing.billing_ledger",
    "billing.billing_log_db",
    "billing.stripe_ledger",
    "alert_dispatcher",
    "rollback_manager",
    "sandbox_review",
    "menace_sanity_layer",
    "vault_secret_provider",
):
    if name not in sys.modules:
        module = types.ModuleType(name)
        sys.modules[name] = module

os.environ.setdefault("STRIPE_SECRET_KEY", "sk_live_stub_123")
os.environ.setdefault("STRIPE_PUBLIC_KEY", "pk_live_stub_123")

sys.modules["billing"].__path__ = []  # type: ignore[attr-defined]
sys.modules["billing"].billing_ledger = sys.modules["billing.billing_ledger"]
sys.modules["billing"].billing_log_db = sys.modules["billing.billing_log_db"]
sys.modules["billing"].stripe_ledger = sys.modules["billing.stripe_ledger"]

if not hasattr(sys.modules["billing"], "billing_logger"):
    sys.modules["billing"].billing_logger = types.SimpleNamespace(log_event=lambda **_: None)

if not hasattr(sys.modules["billing.billing_ledger"], "record_payment"):
    sys.modules["billing.billing_ledger"].record_payment = lambda *_, **__: None

if not hasattr(sys.modules["billing.billing_log_db"], "log_billing_event"):
    sys.modules["billing.billing_log_db"].log_billing_event = lambda *_, **__: None

if not hasattr(sys.modules["billing.stripe_ledger"], "StripeLedger"):
    class _StubLedger:
        def log_event(self, *_, **__):
            return None

    sys.modules["billing.stripe_ledger"].StripeLedger = _StubLedger

sys.modules["alert_dispatcher"].dispatch_alert = lambda *_, **__: None


class _StubRollbackManager:
    def rollback(self, *_args, **_kwargs):
        return None


sys.modules["rollback_manager"].RollbackManager = _StubRollbackManager

sys.modules["sandbox_review"].pause_bot = lambda *_args, **_kwargs: None


class _StubMenaceSanityLayer(types.SimpleNamespace):
    def refresh_billing_instructions(self, *_args, **_kwargs):
        return None

    def record_event(self, *_args, **_kwargs):
        return None


sys.modules["menace_sanity_layer"].refresh_billing_instructions = (
    _StubMenaceSanityLayer().refresh_billing_instructions
)
sys.modules["menace_sanity_layer"].record_payment_anomaly = lambda *_, **__: None
sys.modules["menace_sanity_layer"].record_billing_event = lambda *_, **__: None
sys.modules["menace_sanity_layer"].record_event = _StubMenaceSanityLayer().record_event

if not hasattr(sys.modules["vault_secret_provider"], "VaultSecretProvider"):
    class _StubVaultSecretProvider:
        def get(self, *_args, **_kwargs):
            return ""

    sys.modules["vault_secret_provider"].VaultSecretProvider = _StubVaultSecretProvider

import menace_sandbox.stripe_billing_router as sbr


class _StubStripeAccount:
    def __init__(self):
        self.calls: list[str] = []

    def retrieve(self, *, api_key: str):
        self.calls.append(api_key)
        return {"id": "acct_module"}


def test_get_account_id_uses_accounts_resource(monkeypatch):
    class Accounts:
        def retrieve(self):
            return {"id": "acct_client"}

    class Client:
        accounts = Accounts()

    monkeypatch.setattr(sbr, "_client", lambda key: Client())
    monkeypatch.setattr(
        sbr,
        "stripe",
        types.SimpleNamespace(Account=_StubStripeAccount()),
    )

    account_id = sbr._get_account_id("sk_live_dummy")

    assert account_id == "acct_client"


def test_get_account_id_falls_back_to_module_level(monkeypatch):
    class Accounts:
        def retrieve(self):
            raise TypeError("account identifier required")

    class Client:
        accounts = Accounts()

    stub_account = _StubStripeAccount()

    monkeypatch.setattr(sbr, "_client", lambda key: Client())
    monkeypatch.setattr(sbr, "stripe", types.SimpleNamespace(Account=stub_account))

    account_id = sbr._get_account_id("sk_live_dummy")

    assert account_id == "acct_module"
    assert stub_account.calls == ["sk_live_dummy"]
