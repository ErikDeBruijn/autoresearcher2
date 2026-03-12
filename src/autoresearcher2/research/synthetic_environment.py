import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.environment import Environment


class SyntheticEnvironment(Environment):
    """Synthetic environment with known factor effects for testing.

    outcome = baseline + sum(factor_effects) + noise
    """

    def __init__(
        self,
        schema: InterventionSchema,
        true_effects: dict[str, dict[str, float]],
        noise_std: float = 0.05,
        baseline: float = 0.5,
        interactions: dict[tuple[str, str, str, str], float] | None = None,
        seed: int | None = None,
    ):
        self.schema = schema
        self.true_effects = true_effects
        self.noise_std = noise_std
        self.baseline = baseline
        self.interactions = interactions or {}
        self.rng = np.random.default_rng(seed)

    def run(self, cell_index: int) -> float:
        config = self.schema.cell_to_config(cell_index)
        value = self.baseline
        for factor_name, factor_value in config.items():
            effects = self.true_effects.get(factor_name, {})
            value += effects.get(factor_value, 0.0)
        for (f1, v1, f2, v2), effect in self.interactions.items():
            if config.get(f1) == v1 and config.get(f2) == v2:
                value += effect
        value += self.rng.normal(0, self.noise_std)
        return float(np.clip(value, 0.0, 1.0))
