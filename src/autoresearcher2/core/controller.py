import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel


class Controller:
    """One-step EFE-inspired action selection.

    Action selection uses Thompson sampling: sample weights from posterior,
    compute expected utility under the sample. The pragmatic + epistemic
    decomposition is computed as diagnostics for reporting only.
    """

    def __init__(
        self,
        schema: InterventionSchema,
        model: BayesianLinearModel,
        preferred_outcome: float = 1.0,
        excluded_cells: set[int] | None = None,
        seed: int | None = None,
    ):
        self.schema = schema
        self.model = model
        self.preferred_outcome = preferred_outcome
        self.excluded_cells = excluded_cells or set()
        self.rng = np.random.default_rng(seed)

    def select_next(self) -> int:
        w_sample = self.rng.multivariate_normal(
            self.model.mu_w, self.model.sigma_w
        )
        best_cell = -1
        best_score = -np.inf
        for cell in range(self.schema.n_cells):
            if cell in self.excluded_cells:
                continue
            x = self.model._features(cell)
            predicted = float(x @ w_sample)
            score = -(predicted - self.preferred_outcome) ** 2
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def score_cell(self, cell: int) -> dict[str, float]:
        """Compute pragmatic + epistemic decomposition for a single cell."""
        mean, _ = self.model.predict(cell)
        epistemic = self.model.epistemic_variance(cell)
        pragmatic = -(mean - self.preferred_outcome) ** 2
        return {
            "pragmatic": pragmatic,
            "epistemic": epistemic,
            "total": pragmatic + epistemic,
        }

    def score_all_cells(self) -> dict[int, dict[str, float]]:
        return {cell: self.score_cell(cell) for cell in range(self.schema.n_cells)}
