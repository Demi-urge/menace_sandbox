from menace.menace_memory_manager import MenaceMemoryManager


def test_summarise_memory(tmp_path):
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    mm.store("k", "Sentence one. Sentence two. Sentence three.")
    mm.store("k", "Another sentence.")
    summary = mm.summarise_memory("k", limit=2)
    assert summary
    assert "Sentence" in summary
