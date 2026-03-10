import types

import session_vault as sv


def test_session_vault_uses_memory_conn_when_router_stubbed(monkeypatch):
    monkeypatch.setattr(sv, "router", types.SimpleNamespace())
    vault = sv.SessionVault()
    row = vault.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert row[0] == 0
