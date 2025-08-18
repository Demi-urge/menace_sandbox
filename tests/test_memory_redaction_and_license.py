import menace.memory_bot as mb


class DummyEmbedder:
    def __init__(self) -> None:
        self.called = False
        self.last = None

    def encode(self, texts):
        self.called = True
        self.last = texts[0]
        return [[0.0, 0.0, 0.0]]


def test_gpl_snippet_skipped(tmp_path):
    embedder = DummyEmbedder()
    store = mb.VectorMemoryStorage(tmp_path / "mem.json.gz", embedder=embedder)
    gpl = "This code is covered by the GNU GENERAL PUBLIC LICENSE."
    store.add(mb.MemoryRecord(user="u", text=gpl))
    assert not embedder.called
    stored = store.query("gnu")
    assert stored and (stored[0].meta is None or "embedding" not in stored[0].meta)


def test_secret_redacted(tmp_path):
    embedder = DummyEmbedder()
    store = mb.VectorMemoryStorage(tmp_path / "mem.json.gz", embedder=embedder)
    secret = "password=supersecret"
    store.add(mb.MemoryRecord(user="u", text=secret))
    stored = store.query("REDACTED")[0]
    assert "[REDACTED]" in stored.text and "supersecret" not in stored.text
    results = store.query_vector(secret)
    assert "[REDACTED]" in embedder.last and "supersecret" not in embedder.last
    assert "[REDACTED]" in results[0].text and "supersecret" not in results[0].text

