import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.genetic_hatchery import GeneticHatchery
from neurosales.sql_db import create_session, RLFeedback


def test_hatchery_generates_new_population():
    hatch = GeneticHatchery(["a", "b"], state_dim=2, pop_size=4)
    first = [row[:] for row in hatch.population]
    hatch.next_generation()
    assert len(hatch.population) == 4
    assert hatch.best_genome() is not None
    # ensure population changed
    changed = any(hatch.population[i] != first[i] for i in range(len(first)))
    assert changed


def test_evaluate_uses_feedback_scores():
    Session = create_session("sqlite://")
    with Session() as s:
        s.add(RLFeedback(text="x", feedback="a", score=0.1))
        s.add(RLFeedback(text="x", feedback="b", score=0.9))
        s.commit()

    hatch = GeneticHatchery(
        ["a", "b"],
        state_dim=1,
        pop_size=2,
        session_factory=Session,
    )
    # genome 0 favors action 'a', genome 1 favors 'b'
    hatch.population = [[[1.0], [0.0]], [[0.0], [1.0]]]
    hatch.evaluate()
    assert hatch.fitness[1] > hatch.fitness[0]
