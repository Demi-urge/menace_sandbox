from unittest import mock
from pathlib import Path

import menace.username_generator as ug


def test_generate_username_unique():
    words = ["alpha", "beta", "gamma", "delta"]
    with mock.patch("menace.username_generator._get_topic_words", return_value=words):
        first = ug.generate_username_for_topic("test")
        second = ug.generate_username_for_topic("test", {first})
    assert first != second
    assert any(w in first.lower() for w in words)
    assert any(w in second.lower() for w in words)


def test_load_adjectives_wordnet(monkeypatch, tmp_path):
    class FakeSynset:
        def __init__(self, names):
            self._names = names

        def lemma_names(self):
            return self._names

    class FakeWN:
        def synsets(self, word):
            mapping = {"fast": ["quick"], "smart": ["clever"]}
            return [FakeSynset(mapping.get(word, []))]

    monkeypatch.setattr(ug, "wn", FakeWN())
    monkeypatch.setattr(ug, "_ADJ_FILE", tmp_path / "missing.txt")
    monkeypatch.setattr(ug, "_DEFAULT_ADJECTIVES", ["fast", "smart"])
    adjectives = ug._load_adjectives()
    assert adjectives == ["clever", "quick"]


def test_load_adjectives_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(ug, "wn", None)
    file_path = tmp_path / "adjectives.txt"
    file_path.write_text("cool\nfun\n")
    monkeypatch.setattr(ug, "_ADJ_FILE", file_path)
    monkeypatch.setattr(ug, "_fetch_online_adjectives", lambda: set())
    monkeypatch.setattr(ug, "_DEFAULT_ADJECTIVES", ["base"])
    adjectives = ug._load_adjectives()
    assert adjectives == ["cool", "fun"]


def test_load_adjectives_default(monkeypatch, tmp_path):
    monkeypatch.setattr(ug, "wn", None)
    monkeypatch.setattr(ug, "_ADJ_FILE", tmp_path / "missing.txt")
    monkeypatch.setattr(ug, "_fetch_online_adjectives", lambda: set())
    monkeypatch.setattr(ug, "_DEFAULT_ADJECTIVES", ["solo"])
    adjectives = ug._load_adjectives()
    assert adjectives == ["solo"]


def test_load_adjectives_remote(monkeypatch):
    monkeypatch.setattr(ug, "wn", None)
    monkeypatch.setattr(ug, "_ADJ_FILE", Path("missing.txt"))
    monkeypatch.setattr(ug, "_fetch_online_adjectives", lambda: {"remote"})
    monkeypatch.setattr(ug, "_DEFAULT_ADJECTIVES", ["base"])
    adjectives = ug._load_adjectives()
    assert adjectives == ["remote"]
