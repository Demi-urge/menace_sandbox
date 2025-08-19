from __future__ import annotations

import sys
import types

import pytest

from vector_service import Retriever, FallbackResult
from license_detector import fingerprint as fp
import asyncio


class _DummyUR:
    def retrieve_with_confidence(self, query: str, top_k: int = 5):
        return [], 0.0, []


def _stub_code_db(monkeypatch: pytest.MonkeyPatch, snippet: str) -> None:
    class _CodeDB:
        def search_fts(self, q, limit):
            return [{"id": 1, "code": snippet}]

    monkeypatch.setitem(sys.modules, "code_database", types.SimpleNamespace(CodeDB=_CodeDB))


def test_fallback_fts(monkeypatch):
    snippet = "def foo():\n    return 'bar'"
    _stub_code_db(monkeypatch, snippet)
    retriever = Retriever(retriever=_DummyUR(), use_fts_fallback=False)
    res = retriever.search("query")
    assert isinstance(res, FallbackResult)
    assert len(res) == 1
    item = res[0]
    expected_fp = fp(snippet)
    assert item["origin_db"] == "code"
    assert item["license_fingerprint"] == expected_fp
    assert item["metadata"]["license_fingerprint"] == expected_fp
    assert item["metadata"]["redacted"] is True


def test_fallback_fts_async(monkeypatch):
    snippet = "def async_foo():\n    return 1"
    _stub_code_db(monkeypatch, snippet)
    retriever = Retriever(retriever=_DummyUR(), use_fts_fallback=False)
    res = asyncio.run(retriever.search_async("query"))
    assert isinstance(res, FallbackResult)
    assert len(res) == 1
    item = res[0]
    expected_fp = fp(snippet)
    assert item["origin_db"] == "code"
    assert item["license_fingerprint"] == expected_fp
    assert item["metadata"]["license_fingerprint"] == expected_fp
    assert item["metadata"]["redacted"] is True

