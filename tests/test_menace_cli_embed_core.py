from tests.test_menace_cli_embed import _load_cli


def test_embed_core(monkeypatch):
    calls = {}

    class DummyBackfill:
        def run(self, session_id="cli", dbs=None, batch_size=None, backend=None):
            calls["dbs"] = dbs

    menace_cli = _load_cli(monkeypatch, DummyBackfill())
    rc = menace_cli.main(["embed", "core"])
    assert rc == 0
    assert calls["dbs"] == ["code", "bot", "error", "workflow"]
