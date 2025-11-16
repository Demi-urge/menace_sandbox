from __future__ import annotations

from typing import Dict, List, Tuple

from .rl_integration import QLearningModule
from .engagement_graph import pagerank


class FactionScaleMarlGrid:
    """Coordinate factions of agents with decentralized Q-learning."""

    def __init__(self, factions: Dict[str, List[str]], actions: List[str]) -> None:
        self.factions = factions
        self.actions = actions
        self.agents: Dict[str, QLearningModule] = {}
        self.edges: Dict[str, Dict[str, float]] = {}
        self.ranks: Dict[str, float] = {}
        for faction, members in factions.items():
            self.edges.setdefault(faction, {})
            for agent in members:
                self.agents[agent] = QLearningModule()

    # ------------------------------------------------------------------
    def _faction(self, agent_id: str) -> str:
        for fac, members in self.factions.items():
            if agent_id in members:
                return fac
        raise ValueError(f"Unknown agent: {agent_id}")

    def act(self, agent_id: str, state: Tuple[int, ...]) -> str:
        module = self.agents[agent_id]
        return module.best_action(state, self.actions)

    def _recalculate_ranks(self) -> None:
        self.ranks = pagerank(self.edges)
        if not self.ranks:
            return
        max_rank = max(self.ranks.values()) or 1.0
        for fac, members in self.factions.items():
            rank = self.ranks.get(fac, 0.0)
            epsilon = 0.2 * (1.0 - rank / max_rank) + 0.05
            for agent in members:
                self.agents[agent].epsilon = epsilon

    # ------------------------------------------------------------------
    def skirmish(
        self,
        agent_a: str,
        action_a: str,
        state_a: Tuple[int, ...],
        next_state_a: Tuple[int, ...],
        reward_a: float,
        agent_b: str,
        action_b: str,
        state_b: Tuple[int, ...],
        next_state_b: Tuple[int, ...],
        reward_b: float,
    ) -> None:
        mod_a = self.agents[agent_a]
        mod_b = self.agents[agent_b]
        mod_a.update(state_a, action_a, reward_a, next_state_a, self.actions)
        mod_b.update(state_b, action_b, reward_b, next_state_b, self.actions)
        fac_a = self._faction(agent_a)
        fac_b = self._faction(agent_b)
        if reward_a > reward_b:
            winner, loser, diff = fac_a, fac_b, reward_a - reward_b
        else:
            winner, loser, diff = fac_b, fac_a, reward_b - reward_a
        edges = self.edges.setdefault(loser, {})
        edges[winner] = edges.get(winner, 0.0) + diff
        self._recalculate_ranks()

