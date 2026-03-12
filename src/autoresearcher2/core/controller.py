import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel


class Controller:
    """EFE-inspired action selection with optional two-step lookahead.

    Action selection uses Thompson sampling: sample weights from posterior,
    compute expected utility under the sample. The pragmatic + epistemic
    decomposition is computed as diagnostics for reporting only.

    Two-step lookahead: evaluates (cell1, cell2) pairs by simulating
    the belief update from cell1's hypothetical outcome, then scoring
    cell2 under the updated belief. Returns the first cell of the best pair.
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

    def select_next_lookahead(self, n_candidates: int = 5) -> int:
        """Two-step lookahead: pick cell1 that maximizes (step1 + step2) value.

        1. Score all cells by one-step EFE (pragmatic + epistemic)
        2. Take top n_candidates as step-1 candidates
        3. For each candidate: simulate belief update, find best step-2 cell
        4. Return step-1 cell of the best (step1 + step2) pair
        """
        # Score all cells for step 1
        step1_scores = []
        for cell in range(self.schema.n_cells):
            if cell in self.excluded_cells:
                continue
            s = self.score_cell(cell)
            step1_scores.append((s["total"], cell))
        step1_scores.sort(reverse=True)

        candidates = step1_scores[:n_candidates]

        best_total = -np.inf
        best_cell1 = candidates[0][1] if candidates else 0

        for step1_score, cell1 in candidates:
            # Simulate: what would beliefs look like after observing cell1?
            mean1, _ = self.model.predict(cell1)
            hypothetical_outcome = mean1  # expected outcome under current belief

            # Create temporary updated model state
            snapshot = self.model.snapshot()
            mu_tmp = snapshot["mu_w"].copy()
            sigma_tmp = snapshot["sigma_w"].copy()
            x1 = self.model._features(cell1)

            sigma_inv = np.linalg.inv(sigma_tmp)
            sigma_inv_new = sigma_inv + np.outer(x1, x1) / self.model.noise_variance
            sigma_new = np.linalg.inv(sigma_inv_new)
            mu_new = sigma_new @ (
                sigma_inv @ mu_tmp + x1 * hypothetical_outcome / self.model.noise_variance
            )

            # Find best step-2 cell under updated beliefs
            best_step2 = -np.inf
            for cell2 in range(self.schema.n_cells):
                if cell2 in self.excluded_cells:
                    continue
                x2 = self.model._features(cell2)
                mean2 = float(x2 @ mu_new)
                epistemic2 = float(x2 @ sigma_new @ x2)
                pragmatic2 = -(mean2 - self.preferred_outcome) ** 2
                step2_score = pragmatic2 + epistemic2
                if step2_score > best_step2:
                    best_step2 = step2_score

            total = step1_score + best_step2
            if total > best_total:
                best_total = total
                best_cell1 = cell1

        return best_cell1

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
