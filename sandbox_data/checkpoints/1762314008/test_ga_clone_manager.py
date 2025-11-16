import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.ga_clone_manager as gcm


def test_evolution_logs(tmp_path, monkeypatch):
    mem = gcm.MenaceMemoryManager(tmp_path / "m.db")
    manager = gcm.GALearningManager(["bot"], memory=mem)

    class StubGA(gcm.GeneticAlgorithmBot):
        def evolve(self, generations: int = 1):
            rec = gcm.GARecord(params=[0.0, 0.0, 0.0], roi=1.0)
            self.store.add(rec)
            return rec

    class StubPred(gcm.GAPredictionBot):
        def evolve(self, generations: int = 1):
            return gcm.TemplateEntry(params=[0.1, 0.1, 0.1], score=0.9)

    manager.lineage["bot"].ga_bot = StubGA()
    manager.lineage["bot"].pred_bot = StubPred([[0]], [0])
    manager.run_cycle("bot")
    rows = mem.query("bot_ga")
    pred_rows = mem.query("bot_gapred")
    assert rows and rows[0].version == 1
    assert pred_rows and pred_rows[0].version >= 1
