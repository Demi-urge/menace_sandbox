import types
import pytest

import code_vectorizer
import bot_vectorizer
import error_vectorizer
import workflow_vectorizer


def _setup(monkeypatch, module):
    captured = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return [[0.0] * module._EMBED_DIM for _ in texts]

    monkeypatch.setattr(module, "_embed_texts", fake_embed)
    monkeypatch.setattr(module, "compress_snippets", lambda d: {"snippet": d.get("snippet", "")})
    monkeypatch.setattr(module, "find_semantic_risks", lambda lines: [])
    return captured


def test_code_vectorizer_generalises(monkeypatch):
    captured = _setup(monkeypatch, code_vectorizer)
    monkeypatch.setattr(
        code_vectorizer,
        "split_into_chunks",
        lambda text, max_tokens: [types.SimpleNamespace(text=text)],
    )
    code_vectorizer.CodeVectorizer().transform({"content": "The cat and the dog"})
    assert captured["texts"] == ["cat dog"]


def test_bot_vectorizer_generalises(monkeypatch):
    captured = _setup(monkeypatch, bot_vectorizer)
    bot = {"name": "The Bot", "description": "Does the thing"}
    bot_vectorizer.BotVectorizer().transform(bot)
    assert "the" not in captured["texts"][0]


def test_error_vectorizer_generalises(monkeypatch):
    captured = _setup(monkeypatch, error_vectorizer)
    monkeypatch.setattr(
        error_vectorizer,
        "split_into_chunks",
        lambda text, max_tokens: [types.SimpleNamespace(text=text)],
    )
    err = {"message": "The bad error", "stack_trace": "The stack overflow"}
    error_vectorizer.ErrorVectorizer().transform(err)
    assert "the" not in captured["texts"][0]


def test_workflow_vectorizer_generalises(monkeypatch):
    captured = _setup(monkeypatch, workflow_vectorizer)
    wf = {"name": "The workflow", "workflow": [{"description": "The step"}]}
    workflow_vectorizer.WorkflowVectorizer().transform(wf)
    assert captured["texts"] == ["workflow step"]
