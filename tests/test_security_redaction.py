from embeddable_db_mixin import EmbeddableDBMixin
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
