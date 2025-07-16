from menace.menace_memory_manager import MenaceMemoryManager


def test_refresh_embeddings(tmp_path):
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm._embed = lambda text: [1.0]  # type: ignore
    mm.store("k1", "text")
    mm.conn.execute("DELETE FROM memory_embeddings")
    mm.conn.commit()
    count = mm.refresh_embeddings()
    assert count == 1
