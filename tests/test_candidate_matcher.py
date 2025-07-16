import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

stub = types.ModuleType("stub")
sys.modules.setdefault("menace.task_handoff_bot", stub)
sys.modules.setdefault("menace.unified_event_bus", stub)
sys.modules.setdefault("menace.chatgpt_enhancement_bot", stub)
sys.modules.setdefault("menace.normalize_scraped_data", stub)
sys.modules.setdefault("menace.database_manager", stub)

stub.WorkflowDB = lambda *a, **k: None
stub.WorkflowRecord = object
stub.UnifiedEventBus = object
stub.EnhancementDB = object
stub.Enhancement = object
stub.NicheCandidate = object
stub.DB_PATH = Path("db")
stub.get_connection = lambda db_path: None
stub.init_db = lambda conn: None

import menace.candidate_matcher as cm


def test_text_similarity_sklearn(monkeypatch):
    calls = {}

    class Vec:
        def fit(self, docs):
            calls["fit"] = True
            return self

        def transform(self, docs):
            calls["transform"] = True
            return [0, 0]

    monkeypatch.setattr(cm, "TfidfVectorizer", lambda *a, **k: Vec())
    monkeypatch.setattr(cm, "cosine_similarity", lambda a, b: [[0.6]])
    monkeypatch.setattr(
        cm, "SequenceMatcher", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )

    sim = cm._text_similarity("a", "a")
    assert sim == 0.6
    assert calls == {"fit": True, "transform": True}


def test_text_similarity_fallback(monkeypatch):
    monkeypatch.setattr(cm, "TfidfVectorizer", None)
    monkeypatch.setattr(cm, "cosine_similarity", None)

    called = {"ratio": 0}

    class DummySeq:
        def __init__(self, *a, **k):
            pass

        def ratio(self):
            called["ratio"] += 1
            return 0.0

    monkeypatch.setattr(cm, "SequenceMatcher", DummySeq)

    sim = cm._text_similarity("spam eggs", "spam")
    assert sim > 0.0
    assert called["ratio"] == 0


def test_tfidf_failure_logged(monkeypatch, caplog):
    monkeypatch.setattr(cm, "TfidfVectorizer", None)
    monkeypatch.setattr(cm, "cosine_similarity", None)

    def boom(a, b):
        raise RuntimeError("explode")

    monkeypatch.setattr(cm, "_simple_tfidf_similarity", boom)

    called = {"ratio": 0}

    class DummySeq:
        def __init__(self, *a, **k):
            pass

        def ratio(self):
            called["ratio"] += 1
            return 0.5

    monkeypatch.setattr(cm, "SequenceMatcher", DummySeq)

    import logging

    caplog.set_level(logging.WARNING)

    sim = cm._text_similarity("one", "two")
    assert sim == 0.5
    assert called["ratio"] == 1
    assert "explode" in caplog.text
    assert "one" in caplog.text and "two" in caplog.text


def test_simple_tfidf_stopwords_weighting():
    """Scores should improve compared to SequenceMatcher when stop words are removed."""
    text1 = "the quick brown fox"
    text2 = "quick brown fox"
    sim = cm._simple_tfidf_similarity(text1, text2)
    seq = cm.SequenceMatcher(None, text1, text2).ratio()
    assert sim > seq


def test_simple_tfidf_token_weighting():
    text1 = "a bigword bigword"
    text2 = "bigword"
    sim = cm._simple_tfidf_similarity(text1, text2)
    seq = cm.SequenceMatcher(None, text1, text2).ratio()
    assert sim > seq


def test_simple_tfidf_bigrams_and_normalization():
    cm._TFIDF_CORPUS.clear()
    sim = cm._simple_tfidf_similarity("Machine-learning!", "machine learning")
    assert sim > 0.9


def test_simple_tfidf_corpus_updates():
    cm._TFIDF_CORPUS.clear()
    cm._simple_tfidf_similarity("one two", "three four")
    assert len(cm._TFIDF_CORPUS) == 2
    cm._simple_tfidf_similarity("five six", "seven eight")
    assert len(cm._TFIDF_CORPUS) == 4
