import os
import sys
import types
import logging
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Provide minimal stubs for optional heavy dependencies
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault(
    "menace.chatgpt_idea_bot",
    types.SimpleNamespace(ChatGPTClient=object),
)

import menace.chatgpt_research_bot as crb


def test_summarise_text_exception_logged(monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(crb, "summarize", boom)
    monkeypatch.setattr(crb, "TfidfVectorizer", None)
    monkeypatch.setattr(crb, "TruncatedSVD", None)

    caplog.set_level(logging.ERROR)
    summary = crb.summarise_text("One. Two. Three.", ratio=0.5)

    assert "gensim summarization failed" in caplog.text
    assert summary
    assert summary.count(".") <= 2
