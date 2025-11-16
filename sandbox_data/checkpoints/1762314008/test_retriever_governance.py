import pytest
from vector_service import Retriever

GPL_TEXT = """\
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

UNSAFE_TEXT = "eval('data') # possible danger"


class _GPLHit:
    origin_db = "code"
    score = 1.0
    reason = "match"
    text = GPL_TEXT
    metadata = {"redacted": True}

    def to_dict(self):
        return {
            "origin_db": self.origin_db,
            "score": self.score,
            "reason": self.reason,
            "text": self.text,
            "metadata": self.metadata,
        }


class _UnsafeHit:
    origin_db = "code"
    score = 1.0
    reason = "match"
    text = UNSAFE_TEXT
    metadata = {"redacted": True}

    def to_dict(self):
        return {
            "origin_db": self.origin_db,
            "score": self.score,
            "reason": self.reason,
            "text": self.text,
            "metadata": self.metadata,
        }


class _DummyUR:
    def retrieve_with_confidence(self, query: str, top_k: int = 1):
        return [_GPLHit(), _UnsafeHit()], 1.0, []


def test_retriever_filters_license_and_flags_semantic_risk():
    retriever = Retriever(retriever=_DummyUR())
    hits = retriever.search("query", top_k=2)
    assert len(hits) == 1
    assert UNSAFE_TEXT in hits[0].get("text", "")
    alerts = hits[0]["metadata"].get("semantic_alerts")
    assert alerts
    assert any(msg == "use of eval" for _, msg, _ in alerts)
