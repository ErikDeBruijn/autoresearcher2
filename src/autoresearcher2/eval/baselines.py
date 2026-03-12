from abc import ABC, abstractmethod
import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment


class BaselineAgent(ABC):
    def __init__(self, schema: InterventionSchema):
        self.schema = schema

    @abstractmethod
    def select_next(self) -> int:
        ...

    def observe(self, cell_index: int, outcome: float) -> None:
        pass


class RandomAgent(BaselineAgent):
    def __init__(self, schema: InterventionSchema, seed: int | None = None):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)

    def select_next(self) -> int:
        return int(self.rng.integers(0, self.schema.n_cells))


class TabularAgent(BaselineAgent):
    """Per-cell mean tracking. No generalization across cells."""

    def __init__(self, schema: InterventionSchema, seed: int | None = None):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)
        self.observations: dict[int, list[float]] = {}

    def select_next(self) -> int:
        untried = [
            c for c in range(self.schema.n_cells) if c not in self.observations
        ]
        if untried:
            return int(self.rng.choice(untried))

        # UCB1-style: mean + exploration bonus
        best_cell = -1
        best_score = -np.inf
        total_n = sum(len(v) for v in self.observations.values())
        for cell, outcomes in self.observations.items():
            mean = np.mean(outcomes)
            n = len(outcomes)
            bonus = np.sqrt(2 * np.log(total_n) / n)
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def observe(self, cell_index: int, outcome: float) -> None:
        self.observations.setdefault(cell_index, []).append(outcome)


class GreedyAgent(BaselineAgent):
    def __init__(self, schema: InterventionSchema, seed: int | None = None):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)
        self.observations: dict[int, list[float]] = {}

    def select_next(self) -> int:
        untried = [
            c for c in range(self.schema.n_cells) if c not in self.observations
        ]
        if untried:
            return int(self.rng.choice(untried))

        best_cell = max(
            self.observations,
            key=lambda c: np.mean(self.observations[c]),
        )
        return best_cell

    def observe(self, cell_index: int, outcome: float) -> None:
        self.observations.setdefault(cell_index, []).append(outcome)


def run_baseline(
    agent: BaselineAgent, env: Environment, n_experiments: int
) -> list[dict]:
    results = []
    for i in range(n_experiments):
        cell = agent.select_next()
        outcome = env.run(cell)
        agent.observe(cell, outcome)
        results.append({"experiment": i, "cell": cell, "outcome": outcome})
    return results
