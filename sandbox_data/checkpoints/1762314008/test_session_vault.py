import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.session_vault as sv


def test_add_and_get(tmp_path):
    vault = sv.SessionVault(path=tmp_path / "s.db")
    sid = vault.add(
        "example.com",
        sv.SessionData(cookies={"a": "1"}, user_agent="ua", fingerprint="fp", last_seen=0),
    )
    data = vault.get_least_recent("example.com")
    assert data and data.session_id == sid
    assert data.cookies == {"a": "1"}


def test_report(tmp_path):
    vault = sv.SessionVault(path=tmp_path / "s.db")
    sid = vault.add(
        "example.com",
        sv.SessionData(cookies={}, user_agent="ua", fingerprint="fp", last_seen=0),
    )
    vault.report(sid, success=True)
    data = vault.get_least_recent("example.com")
    assert data and data.success_count == 1


def test_count(tmp_path):
    vault = sv.SessionVault(path=tmp_path / "s.db")
    assert vault.count("example.com") == 0
    vault.add(
        "example.com",
        sv.SessionData(cookies={}, user_agent="ua", fingerprint="fp", last_seen=0),
    )
    assert vault.count("example.com") == 1
    assert vault.count() == 1
