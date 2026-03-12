from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class InterventionSchema:
    factors: dict[str, list[str]]

    @property
    def factor_names(self) -> list[str]:
        return list(self.factors.keys())

    @property
    def factor_sizes(self) -> list[int]:
        return [len(v) for v in self.factors.values()]

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def n_cells(self) -> int:
        result = 1
        for size in self.factor_sizes:
            result *= size
        return result

    @property
    def n_main_effects(self) -> int:
        return sum(self.factor_sizes)

    @property
    def n_interactions(self) -> int:
        total = 0
        sizes = self.factor_sizes
        for i in range(len(sizes)):
            for j in range(i + 1, len(sizes)):
                total += sizes[i] * sizes[j]
        return total

    def cell_to_config(self, cell_index: int) -> dict[str, str]:
        parts = {}
        remaining = cell_index
        for name in reversed(self.factor_names):
            levels = self.factors[name]
            parts[name] = levels[remaining % len(levels)]
            remaining //= len(levels)
        # Return in factor declaration order
        return {name: parts[name] for name in self.factor_names}

    def config_to_cell(self, config: dict[str, str]) -> int:
        index = 0
        for name in self.factor_names:
            levels = self.factors[name]
            index = index * len(levels) + levels.index(config[name])
        return index

    def one_hot(self, cell_index: int) -> NDArray[np.float64]:
        config = self.cell_to_config(cell_index)
        parts = []
        for name in self.factor_names:
            levels = self.factors[name]
            vec = np.zeros(len(levels))
            vec[levels.index(config[name])] = 1.0
            parts.append(vec)
        return np.concatenate(parts)

    def feature_vector(
        self, cell_index: int, include_interactions: bool = False
    ) -> NDArray[np.float64]:
        main = self.one_hot(cell_index)
        if not include_interactions:
            return main

        config = self.cell_to_config(cell_index)
        interaction_parts = []
        names = self.factor_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                vec_i = np.zeros(len(self.factors[names[i]]))
                vec_i[self.factors[names[i]].index(config[names[i]])] = 1.0
                vec_j = np.zeros(len(self.factors[names[j]]))
                vec_j[self.factors[names[j]].index(config[names[j]])] = 1.0
                interaction_parts.append(np.outer(vec_i, vec_j).ravel())

        return np.concatenate([main, *interaction_parts])
