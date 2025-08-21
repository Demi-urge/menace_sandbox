import pkgutil
import textwrap

from vector_service.embedding_backfill import EmbeddingBackfill
import vector_service.embedding_backfill as eb


def test_backfill_discovers_new_db(tmp_path, monkeypatch):
    module = tmp_path / "tempdb.py"
    module.write_text(textwrap.dedent(
        """
        from embeddable_db_mixin import EmbeddableDBMixin

        class TempDB(EmbeddableDBMixin):
            def backfill_embeddings(self, batch_size=0):
                self.batch = batch_size
        """
    ))
    monkeypatch.syspath_prepend(str(tmp_path))

    def fake_walk_packages(path=None, prefix="", onerror=None):
        yield pkgutil.ModuleInfo(None, "tempdb", False)

    monkeypatch.setattr(eb.pkgutil, "walk_packages", fake_walk_packages)

    processed = []

    def fake_process(self, db, *, batch_size, session_id=""):
        processed.append(type(db).__name__)
        return []

    monkeypatch.setattr(EmbeddingBackfill, "_process_db", fake_process)

    backfill = EmbeddingBackfill(batch_size=5)
    backfill.run()
    assert "TempDB" in processed
