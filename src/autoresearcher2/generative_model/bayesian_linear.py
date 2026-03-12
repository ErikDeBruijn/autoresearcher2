import numpy as np
from numpy.typing import NDArray

from autoresearcher2.core.schema import InterventionSchema


class BayesianLinearModel:
    """Bayesian linear regression over intervention features.

    Maintains a Gaussian posterior N(mu_w, sigma_w) over weights.
    Predictions are linear: y = x^T w + noise.
    Updates are exact conjugate (known noise variance).
    """

    def __init__(
        self,
        schema: InterventionSchema,
        include_interactions: bool = True,
        prior_variance: float = 10.0,
        noise_variance: float = 0.1,
    ):
        self.schema = schema
        self.include_interactions = include_interactions
        self.noise_variance = noise_variance

        self.n_features = schema.n_main_effects + (
            schema.n_interactions if include_interactions else 0
        )
        self.mu_w = np.zeros(self.n_features)
        self.sigma_w = np.eye(self.n_features) * prior_variance

    def _features(self, cell_index: int) -> NDArray[np.float64]:
        return self.schema.feature_vector(
            cell_index, include_interactions=self.include_interactions
        )

    def predict(self, cell_index: int) -> tuple[float, float]:
        x = self._features(cell_index)
        mean = float(x @ self.mu_w)
        variance = float(x @ self.sigma_w @ x) + self.noise_variance
        return mean, variance

    def epistemic_variance(self, cell_index: int) -> float:
        x = self._features(cell_index)
        return float(x @ self.sigma_w @ x)

    def update(self, cell_index: int, outcome: float) -> None:
        x = self._features(cell_index)
        sigma_inv = np.linalg.inv(self.sigma_w)
        sigma_inv_new = sigma_inv + np.outer(x, x) / self.noise_variance
        self.sigma_w = np.linalg.inv(sigma_inv_new)
        self.mu_w = self.sigma_w @ (
            sigma_inv @ self.mu_w + x * outcome / self.noise_variance
        )

    def snapshot(self) -> dict:
        return {
            "mu_w": self.mu_w.copy(),
            "sigma_w": self.sigma_w.copy(),
            "noise_variance": self.noise_variance,
        }

    def factor_importances(self) -> dict[str, float]:
        """Heuristic effect magnitude: weight range per factor. Not a causal claim."""
        importances = {}
        offset = 0
        for name in self.schema.factor_names:
            size = len(self.schema.factors[name])
            weights = self.mu_w[offset : offset + size]
            importances[name] = float(np.max(weights) - np.min(weights))
            offset += size
        return importances
