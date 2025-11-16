import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.self_learning import SelfLearningBuffer
from neurosales.sql_db import create_session
from neurosales.self_learning import SelfLearningEngine


def test_self_learning_buffer_persistence():
    Session = create_session("sqlite://")
    buf = SelfLearningBuffer(ttl_seconds=None, session_factory=Session)
    buf.add_event("uh", "hesitation")
    buf2 = SelfLearningBuffer(ttl_seconds=None, session_factory=Session)
    events = buf2.get_events()
    assert len(events) == 1
    assert events[0].text == "uh"
    assert events[0].issue == "hesitation"


def test_self_learning_engine_state_persistence():
    Session = create_session("sqlite://")
    eng1 = SelfLearningEngine(session_factory=Session)
    eng1.log_interaction("fix", "correction", correction="fact")

    eng2 = SelfLearningEngine(session_factory=Session)
    assert ("fact", 1.0) in eng2.corrections_history
    assert eng2.fact_weights.get("fact") is not None


def test_self_learning_engine_db_url(tmp_path):
    db_file = tmp_path / "eng.db"
    url = f"sqlite:///{db_file}"
    eng1 = SelfLearningEngine(db_url=url)
    eng1.log_interaction("fix", "correction", correction="fact")
    eng2 = SelfLearningEngine(db_url=url)
    assert ("fact", 1.0) in eng2.corrections_history


def test_env_var_db_url(tmp_path, monkeypatch):
    db_file = tmp_path / "env.db"
    url = f"sqlite:///{db_file}"
    monkeypatch.setenv("NEURO_DB_URL", url)
    eng1 = SelfLearningEngine()
    eng1.log_interaction("fix", "correction", correction="fa")
    eng2 = SelfLearningEngine()
    assert ("fa", 1.0) in eng2.corrections_history
    monkeypatch.delenv("NEURO_DB_URL", raising=False)

