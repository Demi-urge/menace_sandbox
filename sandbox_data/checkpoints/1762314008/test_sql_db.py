import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

os.environ.pop("NEURO_DB_POOL_SIZE", None)
os.environ.pop("NEURO_DB_MAX_OVERFLOW", None)

from neurosales.sql_db import (
    create_session,
    UserProfile,
    UserPreference,
    ConversationMessage,
    ArchetypeStats,
    MatchHistory,
)


def test_sql_models():
    Session = create_session("sqlite://")
    with Session() as s:
        user = UserProfile(id="u1", username="alice", elo=1200.0, archetype="a1")
        s.add(user)
        s.commit()

        pref = UserPreference(user_id="u1", key="color", value="blue", confidence=0.8)
        msg = ConversationMessage(user_id="u1", role="user", message="hello")
        arch = ArchetypeStats(name="a1", elo=1100.0, interactions=1)
        match = MatchHistory(user_id="u1", archetype="a1", outcome="win")
        s.add_all([pref, msg, arch, match])
        s.commit()

        assert s.get(UserProfile, "u1").username == "alice"
        assert s.query(UserPreference).filter_by(user_id="u1").count() == 1
        assert s.query(ConversationMessage).filter_by(user_id="u1").count() == 1
        assert s.get(ArchetypeStats, "a1").elo == 1100.0
        assert s.query(MatchHistory).filter_by(user_id="u1").count() == 1


def test_create_session_pool_args():
    """create_session should accept engine kwargs for pooling."""
    pytest.importorskip("psycopg2")
    Session = create_session("postgresql://user:pass@localhost/db", pool_size=2)
    assert callable(Session)


def test_create_session_env_pool(monkeypatch):
    """Pool args can come from environment variables."""
    pytest.importorskip("psycopg2")
    monkeypatch.setenv("NEURO_DB_POOL_SIZE", "2")
    monkeypatch.setenv("NEURO_DB_MAX_OVERFLOW", "4")
    Session = create_session("postgresql://user:pass@localhost/db")
    assert callable(Session)
