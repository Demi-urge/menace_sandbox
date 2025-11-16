import pytest

from .test_stripe_billing_router import _import_module


def test_foreign_keys_excluded(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    monkeypatch.setenv("STRIPE_ALLOWED_SECRET_KEYS", "sk_live_dummy,sk_foreign")

    def fake_get_account_id(key: str) -> str:
        return sbr.STRIPE_MASTER_ACCOUNT_ID if key == "sk_live_dummy" else "acct_foreign"

    monkeypatch.setattr(sbr, "_get_account_id", fake_get_account_id)
    assert sbr._load_allowed_keys() == {"sk_live_dummy"}


def test_only_foreign_keys(monkeypatch, tmp_path):
    sbr = _import_module(monkeypatch, tmp_path)
    monkeypatch.setenv("STRIPE_ALLOWED_SECRET_KEYS", "sk_foreign")
    monkeypatch.setattr(sbr, "_get_account_id", lambda k: "acct_foreign")
    assert sbr._load_allowed_keys() == set()
