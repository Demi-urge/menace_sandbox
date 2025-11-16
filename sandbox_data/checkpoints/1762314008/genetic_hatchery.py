from __future__ import annotations

import random
from typing import List, Optional
import os


class GeneticHatchery:
    """Simple genetic algorithm for policy search."""

    def __init__(
        self,
        actions: List[str],
        state_dim: int,
        pop_size: int = 10,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.actions = actions
        self.state_dim = state_dim
        self.pop_size = pop_size
        if session_factory is None:
            from .sql_db import create_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_session(db_url)
        self.session_factory = session_factory
        self.population: List[List[List[float]]] = [self._random_genome() for _ in range(pop_size)]
        self.fitness: List[float] = [0.0 for _ in range(pop_size)]

    # ------------------------------------------------------------------
    def _random_genome(self) -> List[List[float]]:
        return [[random.uniform(-1.0, 1.0) for _ in range(self.state_dim)] for _ in self.actions]

    def _dot(self, w: List[float], s: List[float]) -> float:
        return sum(wi * si for wi, si in zip(w, s))

    def _predict_action(self, g: List[List[float]], state: List[float]) -> str:
        scores = [self._dot(row, state) for row in g]
        idx = scores.index(max(scores))
        return self.actions[idx]

    def _crossover(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        child: List[List[float]] = []
        for row_a, row_b in zip(a, b):
            row = [random.choice([ra, rb]) for ra, rb in zip(row_a, row_b)]
            child.append(row)
        return child

    def _mutate(self, g: List[List[float]], rate: float = 0.1) -> None:
        for i in range(len(g)):
            for j in range(len(g[i])):
                if random.random() < rate:
                    g[i][j] += random.gauss(0, 0.5)

    # ------------------------------------------------------------------
    def evaluate(self) -> None:
        """Assign fitness based on recorded RL feedback and experiences."""
        from .sql_db import RLFeedback, ReplayExperience

        Session = self.session_factory
        with Session() as s:
            fb_rows = (
                s.query(RLFeedback)
                .order_by(RLFeedback.id.desc())
                .limit(100)
                .all()
            )
            xp_rows = (
                s.query(ReplayExperience)
                .order_by(ReplayExperience.id.desc())
                .limit(100)
                .all()
            )

        data = [
            ([float(x) for x in (r.state or [])], r.action, float(r.reward))
            for r in xp_rows
        ]
        data.extend(
            ([float(len(r.text))], r.feedback, float(r.score)) for r in fb_rows
        )

        if not data:
            self.fitness = [0.0 for _ in self.population]
            return

        self.fitness = []
        for g in self.population:
            total = 0.0
            count = 0
            for state, action, reward in data:
                pred = self._predict_action(g, state)
                if pred == action:
                    total += reward
                    count += 1
            self.fitness.append(total / count if count else 0.0)

    def next_generation(self) -> None:
        self.evaluate()
        ranked = sorted(range(len(self.population)), key=lambda i: self.fitness[i], reverse=True)
        elite = [self.population[ranked[0]], self.population[ranked[1]]]
        new_pop = [g[:] for g in elite]
        while len(new_pop) < self.pop_size:
            p1, p2 = random.sample(ranked[: max(2, self.pop_size // 2)], 2)
            child = self._crossover(self.population[p1], self.population[p2])
            self._mutate(child)
            new_pop.append(child)
        self.population = new_pop
        self.fitness = [0.0 for _ in range(self.pop_size)]

    def best_genome(self) -> List[List[float]]:
        if not any(self.fitness):
            self.evaluate()
        idx = max(range(len(self.population)), key=lambda i: self.fitness[i])
        return [row[:] for row in self.population[idx]]
