from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
from security.redaction import redact_secrets


def test_redacts_various_secrets():
    api = "api_key=ABCD1234EFGH5678IJKL"
    aws = "AKIA1234567890ABCD12"
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    priv = "-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----"
    combined = f"{api} {aws} {jwt} {priv}"
    redacted = redact_secrets(combined)
    for secret in ("ABCD1234EFGH5678IJKL", aws, jwt, "BEGIN PRIVATE KEY"):
        assert secret not in redacted
    assert redacted.count("[REDACTED]") >= 4


class _DummyTokenizer:
    def __init__(self):
        self.last_text = ""

    def encode(self, text: str):
        self.last_text = text
        return [1]


class _DummyModel:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.last_text = ""

    def encode(self, text: str):
        self.last_text = text
        class _Vec:
            def tolist(self):
                return [0.0]

        return _Vec()


def test_encode_text_uses_redacted_text():
    obj = EmbeddableDBMixin.__new__(EmbeddableDBMixin)
    obj._model = _DummyModel()
    obj._last_embedding_tokens = 0
    obj._last_embedding_time = 0.0

    secret = "api_key=ABCD1234EFGH5678IJKL"
    obj.encode_text(secret)
    assert secret not in obj._model.tokenizer.last_text
    assert secret not in obj._model.last_text
    assert "[REDACTED]" in obj._model.last_text


def test_govern_retrieval_filters_by_severity(monkeypatch):
    import governed_retrieval as gr

    def fake_find(lines):
        return [("line", "msg", 0.9)]

    monkeypatch.setattr(gr, "find_semantic_risks", fake_find)
    assert gr.govern_retrieval("text", max_alert_severity=0.5) is None


def test_retriever_skips_risky_hits(monkeypatch):
    from vector_service import retriever as rmod

    class Hit:
        def __init__(self, text):
            self.metadata = {"redacted": True}
            self.text = text
            self.score = 0.0

        def to_dict(self):  # pragma: no cover - simple container
            return {"metadata": self.metadata, "text": self.text, "score": self.score}

    def fake_govern(text, meta=None, reason=None, max_alert_severity=1.0):
        if "bad" in text:
            return None
        return ({**(meta or {}), "alignment_severity": 0.4}, reason)

    monkeypatch.setattr(rmod, "govern_retrieval", fake_govern)
    r = rmod.Retriever()
    hits = [Hit("ok"), Hit("bad")]
    results = r._parse_hits(hits, max_alert_severity=0.5)
    assert len(results) == 1
    assert results[0]["alignment_severity"] == 0.4
