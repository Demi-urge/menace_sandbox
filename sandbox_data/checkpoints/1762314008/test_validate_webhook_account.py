import sys
import types
from pathlib import Path

# Stub modules to avoid heavy dependencies during import
billing = types.ModuleType("billing")
billing.billing_logger = types.SimpleNamespace(log_event=lambda **kwargs: None)
sys.modules["billing"] = billing
sys.modules["billing.billing_ledger"] = types.SimpleNamespace(
    record_payment=lambda *a, **k: None
)
sys.modules["billing.billing_log_db"] = types.SimpleNamespace(
    log_billing_event=lambda *a, **k: None
)
sys.modules["billing.stripe_ledger"] = types.SimpleNamespace(
    StripeLedger=lambda: types.SimpleNamespace(log_event=lambda *a, **k: None)
)

stub_disc = types.ModuleType("discrepancy_db")


class DummyDiscrepancyDB:
    def add(self, rec):
        pass


class DummyDiscrepancyRecord:
    def __init__(self, message: str, metadata=None, ts: str = "", id: int = 0):
        self.message = message
        self.metadata = metadata or {}


stub_disc.DiscrepancyDB = DummyDiscrepancyDB
stub_disc.DiscrepancyRecord = DummyDiscrepancyRecord
sys.modules["discrepancy_db"] = stub_disc

vault = types.ModuleType("vault_secret_provider")


class DummyVault:
    def get(self, name):
        if name == "stripe_secret_key":
            return "sk_dummy"
        if name == "stripe_public_key":
            return "pk_dummy"
        return None


vault.VaultSecretProvider = DummyVault
sys.modules["vault_secret_provider"] = vault

sys.modules["alert_dispatcher"] = types.SimpleNamespace(
    dispatch_alert=lambda *a, **k: None
)


class DummyRM:
    def log_healing_action(self, *a, **k):
        pass

    def rollback(self, *a, **k):
        pass


dummy_rm_module = types.SimpleNamespace(RollbackManager=DummyRM)
sys.modules["rollback_manager"] = dummy_rm_module
sys.modules["sandbox_review"] = types.SimpleNamespace(pause_bot=lambda *a, **k: None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.pop("stripe_billing_router", None)
import stripe_billing_router as sbr  # noqa: E402


def test_validate_webhook_account_valid(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_alert(bot_id, account_id, message="", amount=None):
        calls.append((bot_id, account_id))

    monkeypatch.setattr(sbr, "_alert_mismatch", fake_alert, raising=False)
    event = {
        "id": "evt_1",
        "account": sbr.STRIPE_MASTER_ACCOUNT_ID,
        "data": {"object": {"metadata": {"bot_id": "finance:bot"}}},
    }
    assert sbr.validate_webhook_account(event) is True
    assert calls == []


def test_validate_webhook_account_mismatch(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_alert(bot_id, account_id, message="", amount=None):
        calls.append((bot_id, account_id))

    monkeypatch.setattr(sbr, "_alert_mismatch", fake_alert, raising=False)
    event = {
        "id": "evt_2",
        "data": {
            "object": {
                "on_behalf_of": "acct_BAD",
                "metadata": {"bot_id": "finance:bot"},
            }
        },
    }
    assert sbr.validate_webhook_account(event) is False
    assert calls == [("finance:bot", "acct_BAD")]
