import json
from types import SimpleNamespace

import stripe_watchdog as sw


def _fake_list(items):
    class FakeList(list):
        def auto_paging_iter(self):
            return iter(self)
    return FakeList(items)


def test_check_events_detects_missing(monkeypatch, tmp_path):
    ledger = tmp_path / sw.LEDGER_FILE.name
    ledger.write_text(json.dumps({"id": "ch_logged"}) + "\n")
    monkeypatch.setattr(sw, "LEDGER_FILE", ledger)

    charges = _fake_list([{"id": "ch_logged"}, {"id": "ch_new"}])
    refunds = _fake_list([{"id": "re_new"}])
    fake_stripe = SimpleNamespace(
        Charge=SimpleNamespace(list=lambda limit, api_key: charges),
        Refund=SimpleNamespace(list=lambda limit, api_key: refunds),
    )
    monkeypatch.setattr(sw, "stripe", fake_stripe)
    monkeypatch.setattr(sw, "load_api_key", lambda: "sk_test_dummy")

    missing = sw.check_events()
    assert set(missing) == {"ch_new", "re_new"}
