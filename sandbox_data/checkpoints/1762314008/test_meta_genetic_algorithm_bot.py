import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.meta_genetic_algorithm_bot as mgb


def test_meta_ga_evolves(monkeypatch):
    class StubGA(mgb.GeneticAlgorithmBot):
        def evolve(self, generations: int = 1):
            return mgb.GARecord(params=[0.0, 0.0, 0.0], roi=1.0)
    monkeypatch.setattr(mgb, "GeneticAlgorithmBot", StubGA)
    bot = mgb.MetaGeneticAlgorithmBot(population=2)
    rec = bot.evolve()
    assert rec.roi >= 0
