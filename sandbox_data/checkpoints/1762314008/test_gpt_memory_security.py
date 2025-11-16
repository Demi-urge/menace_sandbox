from menace_sandbox.gpt_memory import GPTMemoryManager
import logging
import json

try:
    from security.secret_redactor import redact as redact_secrets
except Exception:  # pragma: no cover - fallback
    from secret_redactor import redact_secrets  # type: ignore


class DummyEmbedder:
    def __init__(self):
        self.last = None

    def encode(self, text):
        self.last = text[0] if isinstance(text, list) else text
        return [0.0]

    @property
    def tokenizer(self):
        class _Tok:
            def encode(self, text):
                return text.split()
        return _Tok()


def test_prompt_redaction_and_search_returns_redacted(caplog):
    emb = DummyEmbedder()
    mem = GPTMemoryManager(db_path=":memory:", embedder=emb)
    prompt = "my password=abc12345"
    response = "Bearer ABCDEFGHIJKLMNOPQRSTUV"
    with caplog.at_level(logging.WARNING):
        mem.log_interaction(prompt, response)

    expected_prompt = redact_secrets(prompt)

    # Embedding should see redacted prompt
    assert emb.last == expected_prompt

    # Database stores redacted prompt but original response
    cur = mem.conn.execute("SELECT prompt, response FROM interactions")
    db_prompt, db_response = cur.fetchone()
    assert db_prompt == expected_prompt
    assert "ABCDEFGHIJKLMNOPQRSTUV" in db_response

    # Query with unredacted secret still retrieves entry and redacts fields
    with caplog.at_level(logging.WARNING):
        res = mem.search_context("password=abc12345")
    assert res and res[0].prompt == expected_prompt
    assert "[REDACTED]" in res[0].response
    assert any("redacted" in r.msg for r in caplog.records)
    mem.close()


def test_disallowed_license_skips_embedding(caplog):
    emb = DummyEmbedder()
    mem = GPTMemoryManager(db_path=":memory:", embedder=emb)
    prompt = "This uses code under the GNU General Public License"
    with caplog.at_level(logging.WARNING):
        mem.log_interaction(prompt, "resp")

    cur = mem.conn.execute("SELECT embedding FROM interactions")
    (embedding,) = cur.fetchone()
    assert embedding is None
    assert emb.last is None  # encode was not called
    assert any("license" in r.msg for r in caplog.records)
    mem.close()


def test_semantic_risk_skips_embedding(caplog):
    emb = DummyEmbedder()
    mem = GPTMemoryManager(db_path=":memory:", embedder=emb)
    prompt = "eval('data')"
    with caplog.at_level(logging.WARNING):
        mem.log_interaction(prompt, "resp")

    cur = mem.conn.execute("SELECT embedding, alerts FROM interactions")
    embedding, alerts_json = cur.fetchone()
    assert embedding is None
    alerts = json.loads(alerts_json)
    assert alerts and any("eval" in a[1] for a in alerts)
    assert emb.last is None
    assert any("semantic" in r.msg for r in caplog.records)
    mem.close()
