import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.marl_grid import FactionScaleMarlGrid


def test_faction_ranking_and_epsilon():
    factions = {"red": ["r1"], "blue": ["b1"]}
    grid = FactionScaleMarlGrid(factions, actions=["x", "y"])
    a_act = grid.act("r1", (0,))
    b_act = grid.act("b1", (0,))
    grid.skirmish(
        "r1",
        a_act,
        (0,),
        (1,),
        1.0,
        "b1",
        b_act,
        (0,),
        (1,),
        0.0,
    )
    assert grid.ranks.get("red", 0.0) >= grid.ranks.get("blue", 0.0)
    eps_red = grid.agents["r1"].epsilon
    eps_blue = grid.agents["b1"].epsilon
    assert eps_red < eps_blue

