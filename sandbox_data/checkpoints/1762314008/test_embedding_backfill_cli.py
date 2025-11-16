import vector_service.embedding_backfill as eb


def test_cli_verify_invokes_verify_only(monkeypatch):
    calls = {}

    class Dummy(eb.EmbeddingBackfill):
        def _verify_registry(self, names=None):
            calls["verify"] = names

        def run(self, *a, **k):  # pragma: no cover - ensure not called
            calls["run"] = True

        def watch(self, *a, **k):  # pragma: no cover - ensure not called
            calls["watch"] = True

    monkeypatch.setattr(eb, "EmbeddingBackfill", lambda: Dummy())
    eb.main(["--verify", "--db", "code"])
    assert calls["verify"] == ["code"]
    assert "run" not in calls and "watch" not in calls
