import os
import sys
import logging
import pytest
from unittest.mock import patch
import time

from dynamic_path_router import resolve_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import types

stub_embed = types.ModuleType("embedding")
stub_embed.embed_text = lambda x: [0.0, 0.0]
stub_sql = types.ModuleType("sql_db")
class DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def add(self, *a, **kw):
        pass

    def commit(self):
        pass

    def query(self, *a, **kw):
        return self

    def order_by(self, *a, **kw):
        return self

    def filter(self, *a, **kw):
        return self

    def all(self):
        return []

    def first(self):
        return None

def create_session(db_url="sqlite://"):
    def _session():
        return DummySession()

    return _session

stub_sql.create_session = create_session
stub_sql.SelfLearningEvent = type("SelfLearningEvent", (), {})
stub_sql.SelfLearningState = type("SelfLearningState", (), {"id": 0, "data": {}})
stub_sql.log_rl_feedback = lambda *a, **kw: None

spec = importlib.util.spec_from_file_location(
    "neurosales.self_learning", str(resolve_path("neurosales/self_learning.py"))
)
self_learning = importlib.util.module_from_spec(spec)
sys.modules.setdefault("neurosales", types.ModuleType("neurosales"))
sys.modules["neurosales.embedding"] = stub_embed
sys.modules["neurosales.sql_db"] = stub_sql
sys.modules["neurosales.self_learning"] = self_learning
spec.loader.exec_module(self_learning)
sys.modules["neurosales"].self_learning = self_learning
class DummyRepo:
    def __init__(self, *a, **kw):
        self.events = []
        self.state = {}

    def add_event(self, *a, **kw):
        text, issue = a[0], a[1]
        correction = kw.get("correction")
        engagement = kw.get("engagement", 1.0)
        self.events.append(self_learning.LearningEvent(time.time(), text, issue, correction, engagement))

    def fetch_events(self, since=None):
        return self.events

    def load_state(self):
        return self.state

    def save_state(self, data):
        self.state = data

    session_factory = lambda self: None

self_learning.SelfLearningRepo = DummyRepo
SelfLearningEngine = self_learning.SelfLearningEngine


def test_watchdog_schedules_jobs():
    engine = SelfLearningEngine()
    engine.log_interaction("uh", "hesitation")
    engine.log_interaction("um", "hesitation")
    engine.run_watchdog(threshold=2)
    assert "hesitation" in engine.micro_jobs


def test_confidence_decay_and_audit():
    engine = SelfLearningEngine(decay_rate=0.8)
    engine.fact_weights["fact"] = 1.0
    engine.log_interaction("fix", "correction", correction="fact")
    assert engine.fact_weights["fact"] < 1.0
    engine.weekly_audit()
    assert engine.main_weights["fact"] == engine.fact_weights["fact"]


def test_embedding_based_clustering():
    engine = SelfLearningEngine()
    with patch("neurosales.self_learning.embed_text") as et:

        def fake_embed(t):
            if "hello" in t or "hi" in t:
                return [1.0, 0.0]
            return [0.0, 1.0]

        et.side_effect = fake_embed

        engine.record_correction("hello")
        engine.record_correction("hi")
        engine.record_correction("bye")
        engine.record_correction("goodbye")

        engine.link_concepts()

        categories = {}
        for cat, names in engine.knowledge_graph.items():
            for name in names:
                categories[name] = cat

        assert categories.get("hello") == categories.get("hi")
        assert categories.get("bye") == categories.get("goodbye")


def test_link_concepts_logs_failure(caplog):
    engine = SelfLearningEngine()
    with patch(
        "neurosales.self_learning.embed_text",
        side_effect=RuntimeError("bad"),
    ), caplog.at_level(logging.ERROR):
        engine.record_correction("hello")
        with pytest.raises(RuntimeError):
            engine.link_concepts()
    assert any("Failed to embed" in r.getMessage() for r in caplog.records)
