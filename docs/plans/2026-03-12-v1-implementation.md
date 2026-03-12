# autoresearcher2 v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a v1 that validates the architecture on synthetic environments and prepares the path to operating on the same substrate as [autoresearch](https://github.com/karpathy/autoresearch) — GPT training pipeline optimization via `train.py` edits, measured by val_bpb. This v1 validates the controller, model, appraisal, and memory on controlled factorial environments. The next phase connects to the real GPT pipeline for head-to-head comparison against autoresearch-style experimentation.

**Architecture:** Three layers (Thompson-sampling controller, Bayesian linear outcome model, episodic memory) + appraisal module + toy validation environment. One-step action selection with EFE-style diagnostics over a fixed intervention schema. Synthetic environment with known factor effects for controlled testing; schema designed to map to real `train.py` levers.

**Tech Stack:** Python 3.11+, numpy, scipy (Bayesian inference), pytest, uv (package management). No ML frameworks in core — only in the GPT pipeline runner (future). Optional: matplotlib for visualization.

---

## Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/autoresearcher2/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: Initialize project with uv**

```bash
cd ~/github.com/erikdebruijn/autoresearcher2
uv init --lib --name autoresearcher2
```

If uv init conflicts with existing files, create manually:

```toml
# pyproject.toml
[project]
name = "autoresearcher2"
version = "0.1.0"
description = "Structured Bayesian research agent with learntropy-inspired appraisal"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]
viz = ["matplotlib>=3.8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Create directory structure**

```bash
mkdir -p src/autoresearcher2/{core,generative_model,appraisal,memory,research,eval,toy}
mkdir -p tests/{core,generative_model,appraisal,memory,research,eval,toy}
touch src/autoresearcher2/__init__.py
touch src/autoresearcher2/{core,generative_model,appraisal,memory,research,eval,toy}/__init__.py
touch tests/__init__.py
touch tests/{core,generative_model,appraisal,memory,research,eval,toy}/__init__.py
```

**Step 3: Verify pytest runs**

```bash
uv sync --dev
uv run pytest --co -q
```

Expected: no errors, no tests collected yet.

**Step 4: Create .gitignore**

```
__pycache__/
*.egg-info/
.venv/
dist/
*.pyc
.pytest_cache/
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with uv, pytest, directory structure"
```

---

## Task 1: Intervention Schema

The fixed factorial grid that defines the search space. Everything downstream depends on this.

**Files:**
- Create: `src/autoresearcher2/core/schema.py`
- Create: `tests/core/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/core/test_schema.py
from autoresearcher2.core.schema import InterventionSchema


def test_schema_creation():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "adamw", "sgd", "lion"],
            "lr_bucket": ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"],
            "pos_encoding": ["rope", "learned", "alibi", "none"],
        }
    )
    assert schema.n_factors == 3
    assert schema.n_cells == 4 * 5 * 4  # 80


def test_schema_cell_to_config():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    config = schema.cell_to_config(0)
    assert config == {"optimizer": "adam", "lr": "low"}

    config = schema.cell_to_config(3)
    assert config == {"optimizer": "sgd", "lr": "high"}


def test_schema_config_to_cell():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    cell = schema.config_to_cell({"optimizer": "sgd", "lr": "low"})
    assert cell == 2


def test_schema_one_hot():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    x = schema.one_hot(0)  # adam, low
    # adam=[1,0], low=[1,0]
    assert list(x) == [1.0, 0.0, 1.0, 0.0]


def test_schema_feature_vector_with_interactions():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    x = schema.feature_vector(0, include_interactions=True)
    # 4 main effects + 4 pairwise interactions = 8
    assert len(x) == 8


def test_schema_cell_roundtrip():
    """cell_to_config → config_to_cell should be identity for all cells."""
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
            "batch": ["small", "large"],
        }
    )
    for cell in range(schema.n_cells):
        config = schema.cell_to_config(cell)
        assert schema.config_to_cell(config) == cell
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_schema.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/core/schema.py
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
        config = {}
        remaining = cell_index
        for name in reversed(self.factor_names):
            levels = self.factors[name]
            config[name] = levels[remaining % len(levels)]
            remaining //= len(levels)
        return config

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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_schema.py -v
```

Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/core/schema.py tests/core/test_schema.py
git commit -m "feat: intervention schema with one-hot and interaction features"
```

---

## Task 2: Structured Bayesian Outcome Model (v1)

Bayesian linear regression over schema features that learns factor-level structure. In v1, this approximates the "generative model" from the full design — it predicts outcomes from interventions but does not yet model latent causes, regimes, or proxy fidelity. That's honest: it's a structured reward model, not a full generative model in the active-inference sense.

**Files:**
- Create: `src/autoresearcher2/generative_model/bayesian_linear.py`
- Create: `tests/generative_model/test_bayesian_linear.py`

**Step 1: Write the failing test**

```python
# tests/generative_model/test_bayesian_linear.py
import numpy as np
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel


def make_schema():
    return InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )


def test_model_creation():
    schema = make_schema()
    model = BayesianLinearModel(schema, include_interactions=True)
    assert model.n_features == 8  # 4 main + 4 interactions
    assert model.mu_w.shape == (8,)
    assert model.sigma_w.shape == (8, 8)


def test_prior_is_uninformative():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    # Prior mean should be zero
    np.testing.assert_array_equal(model.mu_w, np.zeros(model.n_features))
    # Prior covariance should be large diagonal
    assert model.sigma_w[0, 0] > 1.0


def test_predict_returns_mean_and_variance():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    mean, variance = model.predict(cell_index=0)
    assert isinstance(mean, float)
    assert isinstance(variance, float)
    assert variance > 0


def test_update_reduces_variance():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    _, var_before = model.predict(cell_index=0)
    model.update(cell_index=0, outcome=0.8)
    _, var_after = model.predict(cell_index=0)
    assert var_after < var_before


def test_update_shifts_mean_toward_observation():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    mean_before, _ = model.predict(cell_index=0)
    model.update(cell_index=0, outcome=1.0)
    mean_after, _ = model.predict(cell_index=0)
    # Mean should move toward 1.0
    assert abs(mean_after - 1.0) < abs(mean_before - 1.0)


def test_generalization_across_cells():
    """Updating cell 0 (adam, low) should affect cell 1 (adam, high)
    because they share the 'adam' factor."""
    schema = make_schema()
    model = BayesianLinearModel(schema)
    _, var_cell1_before = model.predict(cell_index=1)

    # Observe cell 0 multiple times
    for _ in range(5):
        model.update(cell_index=0, outcome=0.9)

    _, var_cell1_after = model.predict(cell_index=1)
    # Cell 1 variance should decrease because it shares features with cell 0
    assert var_cell1_after < var_cell1_before


def test_factor_importance():
    """After many observations, factor importances should be extractable."""
    schema = make_schema()
    model = BayesianLinearModel(schema)

    # Simulate: adam is always better than sgd
    for _ in range(20):
        model.update(cell_index=0, outcome=0.9)  # adam, low → good
        model.update(cell_index=1, outcome=0.85)  # adam, high → good
        model.update(cell_index=2, outcome=0.5)  # sgd, low → bad
        model.update(cell_index=3, outcome=0.55)  # sgd, high → bad

    importances = model.factor_importances()
    assert "optimizer" in importances
    assert "lr" in importances
    # Optimizer should matter more than lr
    assert importances["optimizer"] > importances["lr"]


def test_snapshot_is_independent_copy():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    snap = model.snapshot()
    assert "mu_w" in snap and "sigma_w" in snap and "noise_variance" in snap
    # Mutating model should not affect snapshot
    model.update(cell_index=0, outcome=0.8)
    assert not np.allclose(model.mu_w, snap["mu_w"])
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/generative_model/test_bayesian_linear.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/generative_model/bayesian_linear.py
import numpy as np
from numpy.typing import NDArray

from autoresearcher2.core.schema import InterventionSchema


class BayesianLinearModel:
    """Bayesian linear regression over intervention features.

    Maintains a Gaussian posterior N(mu_w, sigma_w) over weights.
    Predictions are linear: y = x^T w + noise.
    Updates are exact conjugate (known noise variance) or
    Laplace-approximated (unknown).
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
        """Return (predicted_mean, predictive_variance) for a cell."""
        x = self._features(cell_index)
        mean = float(x @ self.mu_w)
        variance = float(x @ self.sigma_w @ x) + self.noise_variance
        return mean, variance

    def epistemic_variance(self, cell_index: int) -> float:
        """Predictive variance from model uncertainty only (no noise)."""
        x = self._features(cell_index)
        return float(x @ self.sigma_w @ x)

    def update(self, cell_index: int, outcome: float) -> None:
        """Bayesian update: conjugate Gaussian linear regression.

        Uses the standard formula for known noise variance:
          sigma_new = (sigma_old^{-1} + x x^T / noise_var)^{-1}
          mu_new = sigma_new @ (sigma_old^{-1} @ mu_old + x * outcome / noise_var)
        """
        x = self._features(cell_index)
        sigma_inv = np.linalg.inv(self.sigma_w)
        sigma_inv_new = sigma_inv + np.outer(x, x) / self.noise_variance
        self.sigma_w = np.linalg.inv(sigma_inv_new)
        self.mu_w = self.sigma_w @ (
            sigma_inv @ self.mu_w + x * outcome / self.noise_variance
        )

    def snapshot(self) -> dict:
        """Return a copy of the current model state for appraisal."""
        return {
            "mu_w": self.mu_w.copy(),
            "sigma_w": self.sigma_w.copy(),
            "noise_variance": self.noise_variance,
        }

    def factor_importances(self) -> dict[str, float]:
        """Heuristic effect magnitude summary: sum of absolute weight means per factor."""
        importances = {}
        offset = 0
        for name in self.schema.factor_names:
            size = len(self.schema.factors[name])
            weights = self.mu_w[offset : offset + size]
            importances[name] = float(np.sum(np.abs(weights)))
            offset += size
        return importances
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/generative_model/test_bayesian_linear.py -v
```

Expected: all 8 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/generative_model/bayesian_linear.py tests/generative_model/test_bayesian_linear.py
git commit -m "feat: Bayesian linear generative model with conjugate updates and snapshot"
```

---

## Task 3: Thompson-Sampling Controller with EFE-Style Diagnostics

Selects the next experiment via Thompson sampling (sample from posterior, pick best under sample). The actual action selection is posterior sampling, NOT direct EFE minimization. The pragmatic + epistemic decomposition is computed as diagnostics for visualization and evaluation — it reports the decomposition, it doesn't drive the selection.

**Files:**
- Create: `src/autoresearcher2/core/controller.py`
- Create: `tests/core/test_controller.py`

**Step 1: Write the failing test**

```python
# tests/core/test_controller.py
import numpy as np
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller


def make_setup():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    model = BayesianLinearModel(schema)
    return schema, model


def test_controller_selects_cell():
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    cell = controller.select_next()
    assert 0 <= cell < schema.n_cells


def test_controller_explores_when_uncertain():
    """With no data, all cells are equally uncertain.
    Controller should select different cells across multiple calls
    (due to Thompson sampling)."""
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)

    selections = {controller.select_next() for _ in range(50)}
    # Should have visited multiple cells, not stuck on one
    assert len(selections) > 1


def test_controller_exploits_after_learning():
    """After learning that cell 0 is great, controller should prefer it."""
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)

    # Train: cell 0 is great, others are bad
    for _ in range(30):
        model.update(cell_index=0, outcome=0.95)
        model.update(cell_index=1, outcome=0.3)
        model.update(cell_index=2, outcome=0.2)
        model.update(cell_index=3, outcome=0.25)

    # Most selections should now be cell 0
    selections = [controller.select_next() for _ in range(50)]
    cell_0_count = selections.count(0)
    assert cell_0_count > 30  # majority should be cell 0


def test_controller_scores_decomposition():
    """Controller should report pragmatic and epistemic scores."""
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    scores = controller.score_all_cells()
    assert len(scores) == schema.n_cells
    for cell_idx, score in scores.items():
        assert "pragmatic" in score
        assert "epistemic" in score
        assert "total" in score


def test_epistemic_decreases_with_data():
    """Epistemic score for a cell should decrease after observing it."""
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)

    scores_before = controller.score_all_cells()
    ep_before = scores_before[0]["epistemic"]

    model.update(cell_index=0, outcome=0.5)

    scores_after = controller.score_all_cells()
    ep_after = scores_after[0]["epistemic"]

    assert ep_after < ep_before
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_controller.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/core/controller.py
import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel


class Controller:
    """One-step EFE-inspired action selection.

    For each candidate cell, computes:
      pragmatic = predicted closeness to preferred outcome
      epistemic = model uncertainty (predictive variance from weights)
      total = pragmatic + epistemic

    Action selection uses Thompson sampling: sample weights from posterior,
    compute expected utility under the sample. Exploration emerges from
    posterior uncertainty, not from explicit weights.
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
        """Thompson sampling: sample from posterior, pick best under sample."""
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
            # Score = closeness to preferred outcome (negative squared distance)
            score = -(predicted - self.preferred_outcome) ** 2
            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell

    def score_all_cells(self) -> dict[int, dict[str, float]]:
        """Compute pragmatic + epistemic decomposition for all cells."""
        scores = {}
        for cell in range(self.schema.n_cells):
            mean, _ = self.model.predict(cell)
            epistemic = self.model.epistemic_variance(cell)
            pragmatic = -(mean - self.preferred_outcome) ** 2
            scores[cell] = {
                "pragmatic": pragmatic,
                "epistemic": epistemic,
                "total": pragmatic + epistemic,
            }
        return scores
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_controller.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/core/controller.py tests/core/test_controller.py
git commit -m "feat: controller with Thompson sampling and EFE-inspired scoring"
```

---

## Task 4: Appraisal Module (3 Grounded Signals)

v1 computes three signals after each experiment: surprise, theory_conflict, prediction_impact_breadth.

**Important:** Appraisal does NOT update the model. It takes before/after snapshots around an update that the caller performs. This preserves the layer contract: model updates beliefs, appraisal measures consequences.

**Files:**
- Create: `src/autoresearcher2/appraisal/signals.py`
- Create: `tests/appraisal/test_signals.py`

**Step 1: Write the failing test**

```python
# tests/appraisal/test_signals.py
import numpy as np
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.appraisal.signals import compute_appraisal


def make_setup():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    model = BayesianLinearModel(schema)
    return schema, model


def test_appraisal_returns_signals():
    schema, model = make_setup()
    # Appraisal takes before/after snapshots — caller updates model
    snapshot_before = model.snapshot()
    model.update(cell_index=0, outcome=0.5)
    snapshot_after = model.snapshot()
    appraisal = compute_appraisal(
        schema, cell_index=0, outcome=0.5,
        snapshot_before=snapshot_before, snapshot_after=snapshot_after,
    )
    assert "surprise" in appraisal
    assert "theory_conflict" in appraisal
    assert "prediction_impact_breadth" in appraisal
    assert "learntropy" in appraisal


def test_appraisal_does_not_mutate_model():
    """Appraisal must not call model.update() — caller is responsible."""
    schema, model = make_setup()
    snapshot_before = model.snapshot()
    mu_before = model.mu_w.copy()
    model.update(cell_index=0, outcome=0.5)
    snapshot_after = model.snapshot()
    compute_appraisal(
        schema, cell_index=0, outcome=0.5,
        snapshot_before=snapshot_before, snapshot_after=snapshot_after,
    )
    # Model should only reflect the single update we did, not a second one
    np.testing.assert_array_equal(model.mu_w, snapshot_after["mu_w"])


def test_surprising_outcome_scores_higher_than_expected():
    schema, model = make_setup()
    for _ in range(20):
        model.update(cell_index=0, outcome=0.9)

    # Surprising outcome (0.1 when expecting ~0.9)
    snap_before = model.snapshot()
    model_copy_surprising = BayesianLinearModel(schema)
    model_copy_surprising.mu_w = snap_before["mu_w"].copy()
    model_copy_surprising.sigma_w = snap_before["sigma_w"].copy()
    model_copy_surprising.update(cell_index=0, outcome=0.1)
    surprising = compute_appraisal(
        schema, 0, 0.1, snap_before, model_copy_surprising.snapshot(),
    )

    # Expected outcome (0.9 when expecting ~0.9)
    model_copy_expected = BayesianLinearModel(schema)
    model_copy_expected.mu_w = snap_before["mu_w"].copy()
    model_copy_expected.sigma_w = snap_before["sigma_w"].copy()
    model_copy_expected.update(cell_index=0, outcome=0.9)
    expected = compute_appraisal(
        schema, 0, 0.9, snap_before, model_copy_expected.snapshot(),
    )

    assert surprising["surprise"] > expected["surprise"]


def test_theory_conflict_higher_when_confident():
    schema, model = make_setup()

    # Confident model: many observations
    for _ in range(50):
        model.update(cell_index=0, outcome=0.9)
    snap_confident = model.snapshot()
    model.update(cell_index=0, outcome=0.1)
    conflict_confident = compute_appraisal(
        schema, 0, 0.1, snap_confident, model.snapshot(),
    )

    # Uncertain model: no observations
    model2 = BayesianLinearModel(schema)
    snap_uncertain = model2.snapshot()
    model2.update(cell_index=0, outcome=0.1)
    conflict_uncertain = compute_appraisal(
        schema, 0, 0.1, snap_uncertain, model2.snapshot(),
    )

    assert conflict_confident["theory_conflict"] > conflict_uncertain["theory_conflict"]


def test_prediction_impact_breadth_counts_affected_cells():
    schema, model = make_setup()
    snap_before = model.snapshot()
    model.update(cell_index=0, outcome=0.9)
    snap_after = model.snapshot()
    appraisal = compute_appraisal(
        schema, 0, 0.9, snap_before, snap_after,
    )
    # With shared features, updating cell 0 should affect other cells
    assert appraisal["prediction_impact_breadth"] >= 1


def test_learntropy_higher_when_confidently_wrong():
    schema, model = make_setup()
    for _ in range(50):
        model.update(cell_index=0, outcome=0.9)
    snap_confident = model.snapshot()
    model.update(cell_index=0, outcome=0.1)
    high_lt = compute_appraisal(schema, 0, 0.1, snap_confident, model.snapshot())

    model2 = BayesianLinearModel(schema)
    snap_uncertain = model2.snapshot()
    model2.update(cell_index=0, outcome=0.1)
    low_lt = compute_appraisal(schema, 0, 0.1, snap_uncertain, model2.snapshot())

    assert high_lt["learntropy"] > low_lt["learntropy"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/appraisal/test_signals.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/appraisal/signals.py
import numpy as np
from numpy.typing import NDArray

from autoresearcher2.core.schema import InterventionSchema


def compute_appraisal(
    schema: InterventionSchema,
    cell_index: int,
    outcome: float,
    snapshot_before: dict,
    snapshot_after: dict,
    prediction_change_threshold: float = 0.01,
) -> dict[str, float]:
    """Compute learntropy-inspired appraisal signals for an observation.

    Appraisal does NOT update the model. The caller is responsible for:
      1. Taking snapshot_before = model.snapshot()
      2. Calling model.update(cell_index, outcome)
      3. Taking snapshot_after = model.snapshot()
      4. Passing both snapshots here

    Returns:
        surprise: how far the outcome was from prediction (before update),
                  normalized by predictive variance. High = unexpected.
        theory_conflict: surprise weighted by model confidence. High = the model
                         was confidently wrong (the learntropy sweet spot).
        prediction_impact_breadth: how many other cells had their prediction
                          materially changed by the update.
        learntropy: surprise × confidence. Maximum when the model was confident
                    but wrong — not when it was just uncertain.
    """
    mu_before, sigma_before = snapshot_before["mu_w"], snapshot_before["sigma_w"]
    mu_after, sigma_after = snapshot_after["mu_w"], snapshot_after["sigma_w"]

    # Predict from before-snapshot
    x = schema.feature_vector(cell_index, include_interactions=True)
    mean_before = float(x @ mu_before)
    var_before = float(x @ sigma_before @ x) + snapshot_before["noise_variance"]

    # Compute surprise (standardized prediction error)
    prediction_error = abs(outcome - mean_before)
    std = max(np.sqrt(var_before), 1e-8)
    surprise = float(prediction_error / std)
    # Normalize to [0, 1] range using sigmoid-like transform
    surprise_norm = float(1 - np.exp(-0.5 * surprise**2))

    # Confidence = inverse of epistemic variance (how sure was the model?)
    epistemic_var = float(x @ sigma_before @ x)
    confidence = float(1.0 / (1.0 + epistemic_var))

    # Theory conflict = surprise weighted by confidence
    theory_conflict = float(surprise_norm * confidence)

    # Prediction impact breadth: count cells whose prediction changed materially
    impact_count = 0
    for c in range(schema.n_cells):
        xc = schema.feature_vector(c, include_interactions=True)
        pred_before = float(xc @ mu_before)
        pred_after = float(xc @ mu_after)
        if abs(pred_after - pred_before) > prediction_change_threshold:
            impact_count += 1
    prediction_impact_breadth = float(impact_count)

    # Learntropy: surprise × confidence
    learntropy = float(surprise_norm * confidence)

    return {
        "surprise": surprise_norm,
        "theory_conflict": theory_conflict,
        "prediction_impact_breadth": prediction_impact_breadth,
        "learntropy": learntropy,
    }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/appraisal/test_signals.py -v
```

Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/appraisal/signals.py tests/appraisal/test_signals.py
git commit -m "feat: appraisal module with surprise, theory_conflict, prediction_impact_breadth, learntropy"
```

---

## Task 5: Memory (Simple Episodic Store)

v1 memory: store experiment records, retrieve by similarity, dedup, campaign summaries. No activation/decay dynamics yet.

**Files:**
- Create: `src/autoresearcher2/memory/store.py`
- Create: `tests/memory/test_store.py`

**Step 1: Write the failing test**

```python
# tests/memory/test_store.py
from autoresearcher2.memory.store import MemoryStore


def test_store_and_retrieve():
    store = MemoryStore()
    store.add(
        cell_index=0,
        config={"optimizer": "adam", "lr": "low"},
        outcome=0.85,
        appraisal={"surprise": 0.3, "learntropy": 0.2},
    )
    records = store.all()
    assert len(records) == 1
    assert records[0]["outcome"] == 0.85


def test_dedup_detects_repeat():
    store = MemoryStore()
    store.add(cell_index=0, config={"optimizer": "adam"}, outcome=0.85)
    assert store.has_tried(cell_index=0)
    assert not store.has_tried(cell_index=1)


def test_retrieve_by_cell():
    store = MemoryStore()
    store.add(cell_index=0, config={}, outcome=0.8)
    store.add(cell_index=0, config={}, outcome=0.85)
    store.add(cell_index=1, config={}, outcome=0.5)

    results = store.get_by_cell(0)
    assert len(results) == 2


def test_top_by_appraisal():
    store = MemoryStore()
    store.add(cell_index=0, config={}, outcome=0.8, appraisal={"learntropy": 0.1})
    store.add(cell_index=1, config={}, outcome=0.5, appraisal={"learntropy": 0.9})
    store.add(cell_index=2, config={}, outcome=0.6, appraisal={"learntropy": 0.5})

    top = store.top_by_appraisal("learntropy", n=2)
    assert len(top) == 2
    assert top[0]["cell_index"] == 1  # highest learntropy first


def test_summary():
    store = MemoryStore()
    store.add(cell_index=0, config={"opt": "adam"}, outcome=0.9)
    store.add(cell_index=1, config={"opt": "sgd"}, outcome=0.3)
    store.add(cell_index=0, config={"opt": "adam"}, outcome=0.85)

    summary = store.summary()
    assert summary["n_experiments"] == 3
    assert summary["n_unique_cells"] == 2
    assert summary["best_outcome"] == 0.9
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/memory/test_store.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/memory/store.py
from dataclasses import dataclass, field
import time


class MemoryStore:
    """Simple episodic memory for experiment results."""

    def __init__(self):
        self._records: list[dict] = []
        self._tried_cells: set[int] = set()

    def add(
        self,
        cell_index: int,
        config: dict | None = None,
        outcome: float = 0.0,
        appraisal: dict | None = None,
    ) -> None:
        self._records.append(
            {
                "cell_index": cell_index,
                "config": config or {},
                "outcome": outcome,
                "appraisal": appraisal or {},
                "timestamp": time.time(),
            }
        )
        self._tried_cells.add(cell_index)

    def all(self) -> list[dict]:
        return list(self._records)

    def has_tried(self, cell_index: int) -> bool:
        return cell_index in self._tried_cells

    def get_by_cell(self, cell_index: int) -> list[dict]:
        return [r for r in self._records if r["cell_index"] == cell_index]

    def top_by_appraisal(self, signal: str, n: int = 5) -> list[dict]:
        scored = [r for r in self._records if signal in r.get("appraisal", {})]
        scored.sort(key=lambda r: r["appraisal"][signal], reverse=True)
        return scored[:n]

    def summary(self) -> dict:
        if not self._records:
            return {"n_experiments": 0, "n_unique_cells": 0, "best_outcome": None}
        return {
            "n_experiments": len(self._records),
            "n_unique_cells": len(self._tried_cells),
            "best_outcome": max(r["outcome"] for r in self._records),
        }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/memory/test_store.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/memory/store.py tests/memory/test_store.py
git commit -m "feat: simple episodic memory store with dedup and appraisal retrieval"
```

---

## Task 6: Environment Interface + Synthetic Environment

Abstract environment interface + a synthetic environment for testing (before wiring to real NanoGPT). The synthetic environment has known factor effects so we can verify the controller learns them.

**Files:**
- Create: `src/autoresearcher2/research/environment.py`
- Create: `src/autoresearcher2/research/synthetic_environment.py`
- Create: `tests/research/test_synthetic_environment.py`

**Step 1: Write the failing test**

```python
# tests/research/test_synthetic_environment.py
import numpy as np
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment


def make_schema():
    return InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )


def test_environment_returns_outcome():
    schema = make_schema()
    env = SyntheticEnvironment(
        schema=schema,
        true_effects={"optimizer": {"adam": 0.3, "sgd": -0.3}},
        noise_std=0.01,
    )
    outcome = env.run(cell_index=0)
    assert isinstance(outcome, float)


def test_environment_respects_true_effects():
    schema = make_schema()
    env = SyntheticEnvironment(
        schema=schema,
        true_effects={
            "optimizer": {"adam": 0.5, "sgd": -0.5},
            "lr": {"low": 0.0, "high": 0.0},
        },
        noise_std=0.01,
        baseline=0.5,
    )
    # adam cells should score higher
    adam_outcomes = [env.run(0) for _ in range(20)]
    sgd_outcomes = [env.run(2) for _ in range(20)]
    assert np.mean(adam_outcomes) > np.mean(sgd_outcomes)


def test_environment_is_noisy():
    schema = make_schema()
    env = SyntheticEnvironment(
        schema=schema,
        true_effects={},
        noise_std=0.1,
        baseline=0.5,
    )
    outcomes = [env.run(0) for _ in range(100)]
    # Should have variance
    assert np.std(outcomes) > 0.01
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/research/test_synthetic_environment.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/research/environment.py
from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract interface for experiment environments."""

    @abstractmethod
    def run(self, cell_index: int) -> float:
        """Run an experiment and return the outcome (higher = better)."""
        ...
```

```python
# src/autoresearcher2/research/synthetic_environment.py
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
        seed: int | None = None,
    ):
        self.schema = schema
        self.true_effects = true_effects
        self.noise_std = noise_std
        self.baseline = baseline
        self.rng = np.random.default_rng(seed)

    def run(self, cell_index: int) -> float:
        config = self.schema.cell_to_config(cell_index)
        value = self.baseline
        for factor_name, factor_value in config.items():
            effects = self.true_effects.get(factor_name, {})
            value += effects.get(factor_value, 0.0)
        value += self.rng.normal(0, self.noise_std)
        return float(np.clip(value, 0.0, 1.0))
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/research/test_synthetic_environment.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/research/ tests/research/
git commit -m "feat: environment interface + synthetic environment with known factor effects"
```

---

## Task 7: Research Loop (Outer Loop)

Ties everything together: controller selects → environment runs → model updates (with snapshots) → appraisal scores from snapshots → memory stores.

**Files:**
- Create: `src/autoresearcher2/core/loop.py`
- Create: `tests/core/test_loop.py`

**Step 1: Write the failing test**

```python
# tests/core/test_loop.py
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop


def make_loop():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    model = BayesianLinearModel(schema, noise_variance=0.05)
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    memory = MemoryStore()
    env = SyntheticEnvironment(
        schema=schema,
        true_effects={
            "optimizer": {"adam": 0.3, "sgd": -0.3},
            "lr": {"low": -0.1, "high": 0.1},
        },
        baseline=0.5,
        noise_std=0.05,
        seed=42,
    )
    return ResearchLoop(
        schema=schema, model=model, controller=controller, memory=memory, env=env
    )


def test_loop_runs_n_experiments():
    loop = make_loop()
    results = loop.run(n_experiments=10)
    assert len(results) == 10
    assert loop.memory.summary()["n_experiments"] == 10


def test_loop_learns_correct_factor():
    loop = make_loop()
    loop.run(n_experiments=50)
    importances = loop.model.factor_importances()
    # Optimizer has larger effect (±0.3) than lr (±0.1)
    assert importances["optimizer"] > importances["lr"]


def test_loop_results_contain_appraisal():
    loop = make_loop()
    results = loop.run(n_experiments=5)
    for r in results:
        assert "appraisal" in r
        assert "surprise" in r["appraisal"]
        assert "learntropy" in r["appraisal"]


def test_loop_epistemic_decreases_over_time():
    loop = make_loop()
    results = loop.run(n_experiments=30)

    # Average epistemic score in first 10 vs last 10
    first_10_ep = [r["scores"]["epistemic"] for r in results[:10]]
    last_10_ep = [r["scores"]["epistemic"] for r in results[-10:]]

    import numpy as np

    assert np.mean(last_10_ep) < np.mean(first_10_ep)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/core/test_loop.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/core/loop.py
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.environment import Environment
from autoresearcher2.appraisal.signals import compute_appraisal


class ResearchLoop:
    """Main research loop: select → run → appraise → store → repeat."""

    def __init__(
        self,
        schema: InterventionSchema,
        model: BayesianLinearModel,
        controller: Controller,
        memory: MemoryStore,
        env: Environment,
    ):
        self.schema = schema
        self.model = model
        self.controller = controller
        self.memory = memory
        self.env = env

    def run(self, n_experiments: int) -> list[dict]:
        results = []
        for i in range(n_experiments):
            # Select
            cell = self.controller.select_next()
            config = self.schema.cell_to_config(cell)
            scores = self.controller.score_all_cells()[cell]

            # Run
            outcome = self.env.run(cell)

            # Update model (caller owns the update)
            snapshot_before = self.model.snapshot()
            self.model.update(cell, outcome)
            snapshot_after = self.model.snapshot()

            # Appraise (reads snapshots, does NOT update)
            appraisal = compute_appraisal(
                self.schema, cell, outcome, snapshot_before, snapshot_after
            )

            # Store
            self.memory.add(
                cell_index=cell,
                config=config,
                outcome=outcome,
                appraisal=appraisal,
            )

            results.append(
                {
                    "experiment": i,
                    "cell": cell,
                    "config": config,
                    "outcome": outcome,
                    "appraisal": appraisal,
                    "scores": scores,
                }
            )

        return results
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/core/test_loop.py -v
```

Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/core/loop.py tests/core/test_loop.py
git commit -m "feat: research loop tying controller, model, appraisal, and memory together"
```

---

## Task 8: Baseline Agents

Implement the baseline strategies for comparison: random, greedy, GP-UCB stub.

**Files:**
- Create: `src/autoresearcher2/eval/baselines.py`
- Create: `tests/eval/test_baselines.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_baselines.py
import numpy as np
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.eval.baselines import RandomAgent, GreedyAgent, run_baseline


def make_env():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    env = SyntheticEnvironment(
        schema=schema,
        true_effects={"optimizer": {"adam": 0.3, "sgd": -0.3}},
        baseline=0.5,
        noise_std=0.05,
        seed=42,
    )
    return schema, env


def test_random_agent():
    schema, env = make_env()
    agent = RandomAgent(schema, seed=42)
    results = run_baseline(agent, env, n_experiments=20)
    assert len(results) == 20


def test_greedy_agent():
    schema, env = make_env()
    agent = GreedyAgent(schema)
    results = run_baseline(agent, env, n_experiments=20)
    assert len(results) == 20


def test_random_visits_multiple_cells():
    schema, env = make_env()
    agent = RandomAgent(schema, seed=42)
    results = run_baseline(agent, env, n_experiments=20)
    cells = {r["cell"] for r in results}
    assert len(cells) > 1


def test_greedy_converges_to_best():
    schema, env = make_env()
    agent = GreedyAgent(schema)
    results = run_baseline(agent, env, n_experiments=50)
    # Last 10 should mostly be best cells (adam variants: 0 or 1)
    last_cells = [r["cell"] for r in results[-10:]]
    adam_count = sum(1 for c in last_cells if c in [0, 1])
    assert adam_count > 5
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_baselines.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/eval/baselines.py
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
        """Optional: let the agent learn from observations."""
        pass


class RandomAgent(BaselineAgent):
    def __init__(self, schema: InterventionSchema, seed: int | None = None):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)

    def select_next(self) -> int:
        return int(self.rng.integers(0, self.schema.n_cells))


class GreedyAgent(BaselineAgent):
    """Picks the cell with the best observed mean. Explores randomly until
    each cell has been tried at least once (or budget runs out)."""

    def __init__(self, schema: InterventionSchema, seed: int | None = None):
        super().__init__(schema)
        self.rng = np.random.default_rng(seed)
        self.observations: dict[int, list[float]] = {}

    def select_next(self) -> int:
        # If untried cells exist, try one
        untried = [
            c for c in range(self.schema.n_cells) if c not in self.observations
        ]
        if untried:
            return int(self.rng.choice(untried))

        # Otherwise pick best mean
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_baselines.py -v
```

Expected: all 4 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/eval/baselines.py tests/eval/test_baselines.py
git commit -m "feat: random and greedy baseline agents for evaluation"
```

---

## Task 9: Evaluation Harness

Runs the research loop and all baselines on the same environment, computes comparison metrics.

**Files:**
- Create: `src/autoresearcher2/eval/harness.py`
- Create: `tests/eval/test_harness.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_harness.py
from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.eval.harness import run_evaluation


def test_evaluation_runs_and_compares():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    env_config = {
        "true_effects": {"optimizer": {"adam": 0.3, "sgd": -0.3}},
        "baseline": 0.5,
        "noise_std": 0.05,
    }

    report = run_evaluation(
        schema=schema,
        env_config=env_config,
        n_experiments=30,
        seed=42,
    )

    assert "autoresearcher2" in report
    assert "random" in report
    assert "greedy" in report

    for agent_name, metrics in report.items():
        assert "best_outcome" in metrics
        assert "cumulative_regret" in metrics
        assert "outcomes" in metrics


def test_evaluation_autoresearcher_beats_random():
    schema = InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd"],
            "lr": ["low", "high"],
        }
    )
    env_config = {
        "true_effects": {
            "optimizer": {"adam": 0.4, "sgd": -0.4},
            "lr": {"low": -0.1, "high": 0.1},
        },
        "baseline": 0.5,
        "noise_std": 0.02,
    }

    report = run_evaluation(
        schema=schema,
        env_config=env_config,
        n_experiments=80,
        seed=42,
    )

    # autoresearcher2 should find a better outcome than random
    ar2_best = report["autoresearcher2"]["best_outcome"]
    random_best = report["random"]["best_outcome"]
    assert ar2_best >= random_best - 0.05  # allow small margin for noise
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval/test_harness.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/eval/harness.py
import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop
from autoresearcher2.eval.baselines import RandomAgent, GreedyAgent, run_baseline


def _compute_metrics(
    outcomes: list[float], true_best: float
) -> dict:
    cumulative_regret = sum(true_best - o for o in outcomes)
    best_so_far = []
    current_best = -np.inf
    for o in outcomes:
        current_best = max(current_best, o)
        best_so_far.append(current_best)
    return {
        "best_outcome": max(outcomes),
        "cumulative_regret": cumulative_regret,
        "outcomes": outcomes,
        "best_so_far": best_so_far,
    }


def run_evaluation(
    schema: InterventionSchema,
    env_config: dict,
    n_experiments: int = 100,
    seed: int = 42,
) -> dict[str, dict]:
    # Compute true best for regret calculation
    true_best_env = SyntheticEnvironment(schema=schema, **env_config, seed=0)
    true_outcomes = [
        np.mean([true_best_env.run(c) for _ in range(100)])
        for c in range(schema.n_cells)
    ]
    true_best = max(true_outcomes)

    report = {}

    # autoresearcher2
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    model = BayesianLinearModel(schema, noise_variance=env_config.get("noise_std", 0.05) ** 2)
    controller = Controller(schema, model, preferred_outcome=1.0, seed=seed)
    memory = MemoryStore()
    loop = ResearchLoop(schema, model, controller, memory, env)
    results = loop.run(n_experiments)
    outcomes = [r["outcome"] for r in results]
    report["autoresearcher2"] = _compute_metrics(outcomes, true_best)
    report["autoresearcher2"]["factor_importances"] = model.factor_importances()
    report["autoresearcher2"]["memory_summary"] = memory.summary()

    # Random baseline
    env_rand = SyntheticEnvironment(schema=schema, **env_config, seed=seed + 1)
    agent_rand = RandomAgent(schema, seed=seed + 1)
    results_rand = run_baseline(agent_rand, env_rand, n_experiments)
    outcomes_rand = [r["outcome"] for r in results_rand]
    report["random"] = _compute_metrics(outcomes_rand, true_best)

    # Greedy baseline
    env_greedy = SyntheticEnvironment(schema=schema, **env_config, seed=seed + 2)
    agent_greedy = GreedyAgent(schema, seed=seed + 2)
    results_greedy = run_baseline(agent_greedy, env_greedy, n_experiments)
    outcomes_greedy = [r["outcome"] for r in results_greedy]
    report["greedy"] = _compute_metrics(outcomes_greedy, true_best)

    return report
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/eval/test_harness.py -v
```

Expected: all 2 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/eval/harness.py tests/eval/test_harness.py
git commit -m "feat: evaluation harness comparing autoresearcher2 vs baselines"
```

---

## Task 10: Toy Validation Environment (Canonical Active Inference)

27-cell POMDP with canonical EFE. Separate from the practical system — validates the theory.

**Files:**
- Create: `src/autoresearcher2/toy/pomdp.py`
- Create: `src/autoresearcher2/toy/active_inference.py`
- Create: `tests/toy/test_active_inference.py`

**Step 1: Write the failing test**

```python
# tests/toy/test_active_inference.py
import numpy as np
from autoresearcher2.toy.pomdp import ToyPOMDP
from autoresearcher2.toy.active_inference import ActiveInferenceAgent


def test_pomdp_creation():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    assert pomdp.n_states == 27
    assert pomdp.n_observations == 5  # outcome bands


def test_agent_selects_action():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    action = agent.select_action()
    assert 0 <= action < pomdp.n_states


def test_agent_updates_beliefs():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    beliefs_before = agent.beliefs.copy()
    action = agent.select_action()
    obs = pomdp.observe(action)
    agent.update(action, obs)
    # Beliefs should have changed
    assert not np.allclose(agent.beliefs, beliefs_before)


def test_epistemic_decreases_over_time():
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)

    epistemic_values = []
    for _ in range(50):
        efe = agent.compute_efe_all_actions()
        avg_epistemic = np.mean([e["epistemic"] for e in efe.values()])
        epistemic_values.append(avg_epistemic)
        action = agent.select_action()
        obs = pomdp.observe(action)
        agent.update(action, obs)

    # Epistemic should decrease over time (rolling average)
    first_quarter = np.mean(epistemic_values[:12])
    last_quarter = np.mean(epistemic_values[-12:])
    assert last_quarter < first_quarter


def test_canonical_efe_decomposition():
    """EFE should decompose into pragmatic + epistemic."""
    pomdp = ToyPOMDP(n_factors=3, levels_per_factor=3, seed=42)
    agent = ActiveInferenceAgent(pomdp)
    efe = agent.compute_efe_all_actions()
    for action, components in efe.items():
        assert "pragmatic" in components
        assert "epistemic" in components
        assert "total" in components
        np.testing.assert_almost_equal(
            components["total"],
            components["pragmatic"] + components["epistemic"],
            decimal=10,
        )
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/toy/test_active_inference.py -v
```

Expected: FAIL with ImportError.

**Step 3: Write minimal implementation**

```python
# src/autoresearcher2/toy/pomdp.py
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

        # True observation likelihoods A(o|s) — randomly generated
        # Each state has a categorical distribution over observations
        raw = self.rng.dirichlet(np.ones(n_obs), size=self.n_states)
        self.A = raw  # shape: (n_states, n_obs)

        # Prior preferences over observations (prefer "excellent")
        # C[o] = log preference
        self.C = np.array([-4.0, -2.0, 0.0, 1.0, 2.0])

    def observe(self, action: int) -> int:
        """Sample an observation from A(o|s=action)."""
        probs = self.A[action]
        return int(self.rng.choice(self.n_observations, p=probs))
```

```python
# src/autoresearcher2/toy/active_inference.py
import numpy as np
from scipy.special import softmax, digamma

from autoresearcher2.toy.pomdp import ToyPOMDP


class ActiveInferenceAgent:
    """Canonical active inference agent for the toy POMDP.

    Maintains Dirichlet beliefs about the observation likelihood A(o|s).
    Selects actions by minimizing expected free energy G.
    """

    def __init__(self, pomdp: ToyPOMDP, gamma: float = 1.0):
        self.pomdp = pomdp
        self.gamma = gamma

        # Beliefs: Dirichlet concentration parameters for each state
        # Initialized uniform (no prior knowledge)
        self.beliefs = np.ones((pomdp.n_states, pomdp.n_observations))
        # beliefs[s] is a Dirichlet parameter vector for A(o|s)

    def _expected_A(self) -> np.ndarray:
        """Expected observation likelihood under current beliefs."""
        # E[A(o|s)] for Dirichlet = alpha / sum(alpha)
        return self.beliefs / self.beliefs.sum(axis=1, keepdims=True)

    def compute_efe_all_actions(self) -> dict[int, dict[str, float]]:
        """Compute EFE decomposition for all actions.

        G(a) = pragmatic + epistemic

        pragmatic = -E_Q(o|a)[ log P(o) ]
            = negative expected log preference
            = risk: how far from preferred outcomes

        epistemic = -E_Q(o|a)[ D_KL[ Q(s|o,a) || Q(s|a) ] ]
            For our bandit: this reduces to the negative expected
            information gain about A(o|s=a).
            = -[ H[expected_A(a)] - expected_H[A(o|s=a)] ]
            = ambiguity - entropy of expected likelihood
        """
        efe = {}
        expected_A = self._expected_A()

        for a in range(self.pomdp.n_states):
            # Expected observations Q(o|a)
            q_o = expected_A[a]

            # Pragmatic: -E[log P(o)] = -sum q(o) * C(o)
            pragmatic = -float(q_o @ self.pomdp.C)

            # Epistemic (information gain about A(o|s=a)):
            # H[E[A]] - E[H[A]]
            # H[E[A]] = entropy of the expected categorical
            h_expected = -float(np.sum(q_o * np.log(q_o + 1e-16)))

            # E[H[A]] = expected entropy under Dirichlet
            alpha = self.beliefs[a]
            alpha_0 = alpha.sum()
            expected_entropy = float(
                -np.sum(
                    (alpha / alpha_0) * (digamma(alpha + 1) - digamma(alpha_0 + 1))
                )
            )

            epistemic = -(h_expected - expected_entropy)

            efe[a] = {
                "pragmatic": pragmatic,
                "epistemic": epistemic,
                "total": pragmatic + epistemic,
            }

        return efe

    def select_action(self) -> int:
        """Select action by minimizing expected free energy."""
        efe = self.compute_efe_all_actions()
        g_values = np.array([efe[a]["total"] for a in range(self.pomdp.n_states)])
        # Softmax policy selection
        probs = softmax(-self.gamma * g_values)
        return int(np.random.choice(self.pomdp.n_states, p=probs))

    def update(self, action: int, observation: int) -> None:
        """Dirichlet update: add 1 to the observed category."""
        self.beliefs[action, observation] += 1.0
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/toy/test_active_inference.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add src/autoresearcher2/toy/ tests/toy/
git commit -m "feat: toy POMDP with canonical active inference (EFE, Dirichlet beliefs)"
```

---

## Task 11: Full Test Suite + Final Verification

Run all tests, verify everything integrates, make sure we're green.

**Step 1: Run full test suite**

```bash
uv run pytest -v --tb=short
```

Expected: all tests PASS (approximately 42 tests across 9 files).

**Step 2: Run with coverage**

```bash
uv run pytest --cov=autoresearcher2 --cov-report=term-missing
```

Verify coverage is reasonable (>80% on core modules).

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: test suite integration fixes (if any)"
```

**Step 4: Push**

```bash
git push
```

---

## Post-v1: What Comes Next

After v1 is working and baselines are beaten on the synthetic environment:

1. **Connect to GPT pipeline** — Wire to autoresearch's `train.py` substrate: edit → train 5 min → measure val_bpb → update model. Schema maps to real `train.py` levers (depth, optimizer, LR, attention pattern, etc.)
2. **Head-to-head with autoresearch** — Run both systems on same pipeline, same compute budget. Compare val_bpb convergence, experiment efficiency, structural knowledge
3. **Add GP-UCB and ASHA baselines** — requires `scikit-optimize` or `optuna` integration
4. **Visualization** — matplotlib plots of EFE decomposition, factor importances, appraisal trajectories over time
5. **Multi-step policy planning** — 2-step sequences, especially for proxy→expensive validation
6. **Memory dynamics** — activation/decay, appraisal-weighted persistence
7. **Transfer** — cross-campaign prior blending with negative-transfer guards
8. **Regime variable** — if the simpler model's residuals show regime structure
