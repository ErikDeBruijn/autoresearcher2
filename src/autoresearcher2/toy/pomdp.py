import numpy as np


class ToyPOMDP:
    """27-cell POMDP for validating canonical active inference.

    States: one per cell (unknown true quality)
    Observations: 5 outcome bands (failed, poor, mediocre, good, excellent)
    Actions: run experiment in cell i
    Transition: identity (cells don't change)
    """

    def __init__(
        self,
        n_factors: int = 3,
        levels_per_factor: int = 3,
        n_obs: int = 5,
        seed: int | None = None,
    ):
        self.n_states = levels_per_factor**n_factors
        self.n_observations = n_obs
        self.rng = np.random.default_rng(seed)

        raw = self.rng.dirichlet(np.ones(n_obs), size=self.n_states)
        self.A = raw  # shape: (n_states, n_obs)

        self.C = np.array([-4.0, -2.0, 0.0, 1.0, 2.0])

    def observe(self, action: int) -> int:
        probs = self.A[action]
        return int(self.rng.choice(self.n_observations, p=probs))
