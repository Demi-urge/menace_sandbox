import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.user_preferences import DatabasePreferenceEngine
from neurosales.sql_db import create_session as create_sql_session


def test_persistent_preferences():
    Session = create_sql_session("sqlite://")
    eng1 = DatabasePreferenceEngine(session_factory=Session)
    eng1.add_message("u1", "I like apples")

    eng2 = DatabasePreferenceEngine(session_factory=Session)
    prof = eng2.get_profile("u1")
    assert prof.keyword_freq.get("apples", 0) > 0


def test_preference_db_url(tmp_path):
    db_file = tmp_path / "prefs.db"
    url = f"sqlite:///{db_file}"
    eng1 = DatabasePreferenceEngine(db_url=url)
    eng1.add_message("u1", "I love oranges")

    eng2 = DatabasePreferenceEngine(db_url=url)
    prof = eng2.get_profile("u1")
    assert prof.keyword_freq.get("oranges", 0) > 0
