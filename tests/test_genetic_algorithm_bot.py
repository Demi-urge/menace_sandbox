import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.genetic_algorithm_bot as gab


def test_evolve_improves(tmp_path):
    store = gab.GAStore(tmp_path / "ga.csv")
    bot = gab.GeneticAlgorithmBot(pop_size=6, store=store)
    rec = bot.evolve(generations=2)
    assert isinstance(rec.roi, float)
    assert bot.evaluation_count() > 0
    assert len(store.df) >= 2
