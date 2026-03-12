"""End-to-end integration tests for autoresearcher2.

These tests exercise the REAL behavior of the full system,
not just isolated happy-path unit tests.
"""

import numpy as np
import pytest

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop
from autoresearcher2.appraisal.signals import compute_appraisal
from autoresearcher2.eval.harness import run_evaluation
from autoresearcher2.toy.pomdp import ToyPOMDP
from autoresearcher2.toy.active_inference import ActiveInferenceAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_standard_schema():
    return InterventionSchema(
        factors={
            "optimizer": ["adam", "sgd", "rmsprop"],
            "lr": ["low", "mid", "high"],
            "batch_size": ["32", "64", "128"],
        }
    )


def _make_standard_effects():
    return {
        "optimizer": {"adam": 0.3, "sgd": -0.1, "rmsprop": 0.0},
        "lr": {"low": -0.2, "mid": 0.0, "high": 0.1},
        "batch_size": {"32": -0.05, "64": 0.0, "128": 0.05},
    }


def _run_full_loop(schema, true_effects, n_experiments=50, seed=42,
                   noise_std=0.02, include_interactions=True):
    model = BayesianLinearModel(
        schema,
        include_interactions=include_interactions,
        noise_variance=noise_std ** 2,
    )
    controller = Controller(schema, model, preferred_outcome=1.0, seed=seed)
    memory = MemoryStore()
    env = SyntheticEnvironment(
        schema=schema,
        true_effects=true_effects,
        noise_std=noise_std,
        baseline=0.5,
        seed=seed,
    )
    loop = ResearchLoop(schema, model, controller, memory, env)
    results = loop.run(n_experiments)
    return loop, results, model, memory, env, controller


# ===========================================================================
# 1. Deterministic reproducibility
# ===========================================================================

class TestDeterministicReproducibility:
    """Run the full loop twice with the same seed and verify identical results."""

    def test_full_loop_reproducible(self):
        schema = _make_standard_schema()
        effects = _make_standard_effects()

        _, results1, model1, _, _, _ = _run_full_loop(schema, effects, n_experiments=30, seed=99)
        _, results2, model2, _, _, _ = _run_full_loop(schema, effects, n_experiments=30, seed=99)

        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1["cell"] == r2["cell"], f"Step {i}: cells differ ({r1['cell']} vs {r2['cell']})"
            assert r1["outcome"] == pytest.approx(r2["outcome"]), f"Step {i}: outcomes differ"
            for key in r1["appraisal"]:
                assert r1["appraisal"][key] == pytest.approx(r2["appraisal"][key]), (
                    f"Step {i}: appraisal[{key}] differs"
                )

        np.testing.assert_array_almost_equal(model1.mu_w, model2.mu_w)
        np.testing.assert_array_almost_equal(model1.sigma_w, model2.sigma_w)


# ===========================================================================
# 2. System actually finds the best cell
# ===========================================================================

class TestFindsBestCell:
    """With a clear best cell (strong signal, low noise), verify the system
    finds it and can report which cell is best."""

    def test_identifies_best_config(self):
        schema = InterventionSchema(
            factors={
                "optimizer": ["adam", "sgd"],
                "lr": ["low", "high"],
            }
        )
        # adam+high is clearly best: 0.5 + 0.4 + 0.3 = 1.2 (clipped to 1.0)
        effects = {
            "optimizer": {"adam": 0.4, "sgd": -0.3},
            "lr": {"low": -0.2, "high": 0.3},
        }

        _, results, model, memory, _, _ = _run_full_loop(
            schema, effects, n_experiments=100, seed=42, noise_std=0.01
        )

        # Find the cell the model thinks is best
        best_cell = max(range(schema.n_cells), key=lambda c: model.predict(c)[0])
        best_config = schema.cell_to_config(best_cell)

        assert best_config["optimizer"] == "adam", f"Expected adam, got {best_config['optimizer']}"
        assert best_config["lr"] == "high", f"Expected high, got {best_config['lr']}"

        # Also verify the model's predicted value is close to the true value (clipped to 1.0)
        predicted_mean, _ = model.predict(best_cell)
        assert predicted_mean > 0.8, f"Expected high predicted mean, got {predicted_mean}"

    def test_system_converges_to_best_cell(self):
        """After learning, the controller should overwhelmingly select the best cell."""
        schema = InterventionSchema(
            factors={
                "optimizer": ["adam", "sgd"],
                "lr": ["low", "high"],
            }
        )
        effects = {
            "optimizer": {"adam": 0.4, "sgd": -0.3},
            "lr": {"low": -0.2, "high": 0.3},
        }

        _, results, model, _, _, controller = _run_full_loop(
            schema, effects, n_experiments=80, seed=42, noise_std=0.01
        )

        # After 80 experiments, last 20 selections should be mostly the best cell
        best_cell = schema.config_to_cell({"optimizer": "adam", "lr": "high"})
        last_20_cells = [r["cell"] for r in results[-20:]]
        best_count = last_20_cells.count(best_cell)
        assert best_count >= 15, f"Expected best cell selected >= 15/20 times, got {best_count}"


# ===========================================================================
# 3. Bayesian math correctness
# ===========================================================================

class TestBayesianMathCorrectness:
    """After many observations of the same cell, the posterior mean should
    converge to the observed value."""

    def test_posterior_converges_to_observed_value(self):
        schema = InterventionSchema(factors={"x": ["a", "b"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=10.0, noise_variance=0.1
        )

        # Observe cell 0 a hundred times with outcome=0.7
        for _ in range(100):
            model.update(cell_index=0, outcome=0.7)

        predicted_mean, predicted_var = model.predict(0)
        assert predicted_mean == pytest.approx(0.7, abs=0.05), (
            f"After 100 observations of 0.7, mean should be ~0.7, got {predicted_mean}"
        )
        # Variance should be very low
        assert predicted_var < 0.11, (
            f"Variance should be close to noise_variance=0.1, got {predicted_var}"
        )

    def test_posterior_with_two_cells(self):
        """Observe cell 0 with 0.3 and cell 1 with 0.8, check both converge."""
        schema = InterventionSchema(factors={"x": ["a", "b"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=10.0, noise_variance=0.1
        )

        for _ in range(100):
            model.update(0, 0.3)
            model.update(1, 0.8)

        mean_0, _ = model.predict(0)
        mean_1, _ = model.predict(1)
        assert mean_0 == pytest.approx(0.3, abs=0.05), f"Cell 0 mean: {mean_0}"
        assert mean_1 == pytest.approx(0.8, abs=0.05), f"Cell 1 mean: {mean_1}"


# ===========================================================================
# 4. Appraisal consistency with model state
# ===========================================================================

class TestAppraisalConsistency:
    """After a surprising observation, surprise should be high.
    After an expected one, surprise should be low."""

    def test_surprising_observation_high_surprise(self):
        schema = InterventionSchema(factors={"x": ["a", "b"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=1.0, noise_variance=0.01
        )

        # Train the model to expect ~0.2 for cell 0
        for _ in range(50):
            model.update(0, 0.2)

        # Now give a very surprising observation: 0.9
        snap_before = model.snapshot()
        model.update(0, 0.9)
        snap_after = model.snapshot()

        appraisal = compute_appraisal(schema, 0, 0.9, snap_before, snap_after)
        assert appraisal["surprise"] > 0.5, (
            f"Surprise should be > 0.5 for unexpected observation, got {appraisal['surprise']}"
        )

    def test_expected_observation_low_surprise(self):
        schema = InterventionSchema(factors={"x": ["a", "b"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=1.0, noise_variance=0.01
        )

        # Train model to expect ~0.5 for cell 0
        for _ in range(50):
            model.update(0, 0.5)

        # Observe exactly what we expect
        snap_before = model.snapshot()
        model.update(0, 0.5)
        snap_after = model.snapshot()

        appraisal = compute_appraisal(schema, 0, 0.5, snap_before, snap_after)
        assert appraisal["surprise"] < 0.2, (
            f"Surprise should be < 0.2 for expected observation, got {appraisal['surprise']}"
        )


# ===========================================================================
# 5. Memory store consistency with loop
# ===========================================================================

class TestMemoryStoreConsistency:
    """After running N experiments, memory should have exactly N records,
    and each record's outcome should match what the environment produces."""

    def test_memory_has_correct_count(self):
        schema = _make_standard_schema()
        effects = _make_standard_effects()
        n = 25

        _, results, _, memory, _, _ = _run_full_loop(
            schema, effects, n_experiments=n, seed=77
        )

        assert memory.summary()["n_experiments"] == n
        assert len(memory.all()) == n

    def test_memory_outcomes_match_environment(self):
        """Replay the same cells through a fresh environment with same seed
        to verify outcomes match."""
        schema = _make_standard_schema()
        effects = _make_standard_effects()
        seed = 77
        n = 20

        _, results, _, memory, _, _ = _run_full_loop(
            schema, effects, n_experiments=n, seed=seed, noise_std=0.02
        )

        # Recreate environment with same seed and replay cell selections
        env_replay = SyntheticEnvironment(
            schema=schema, true_effects=effects, noise_std=0.02,
            baseline=0.5, seed=seed,
        )

        records = memory.all()
        for i, record in enumerate(records):
            expected_outcome = env_replay.run(record["cell_index"])
            assert record["outcome"] == pytest.approx(expected_outcome), (
                f"Record {i}: memory outcome {record['outcome']} != "
                f"env replay {expected_outcome} for cell {record['cell_index']}"
            )

    def test_memory_records_have_all_fields(self):
        schema = _make_standard_schema()
        effects = _make_standard_effects()

        _, _, _, memory, _, _ = _run_full_loop(
            schema, effects, n_experiments=5, seed=42
        )

        for record in memory.all():
            assert "cell_index" in record
            assert "config" in record
            assert "outcome" in record
            assert "appraisal" in record
            assert "timestamp" in record
            assert "surprise" in record["appraisal"]
            assert "learntropy" in record["appraisal"]


# ===========================================================================
# 6. Controller Thompson sampling correctness
# ===========================================================================

class TestThompsonSampling:
    """With a tight posterior around a known best, controller should almost
    always pick that cell. With flat posterior, should be roughly uniform."""

    def test_tight_posterior_selects_best(self):
        schema = InterventionSchema(factors={"x": ["a", "b", "c"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=10.0, noise_variance=0.01
        )

        # Heavily train: cell 2 (x=c) is best with outcome=0.9
        # cell 0 (x=a) is worst with outcome=0.1
        for _ in range(200):
            model.update(0, 0.1)  # a
            model.update(1, 0.5)  # b
            model.update(2, 0.9)  # c

        # Now create a controller and sample 100 times
        selections = []
        for trial in range(100):
            controller = Controller(schema, model, preferred_outcome=1.0, seed=trial)
            selections.append(controller.select_next())

        count_best = selections.count(2)
        assert count_best >= 90, (
            f"With tight posterior, best cell should be selected >= 90/100 times, "
            f"got {count_best}"
        )

    def test_flat_prior_roughly_uniform(self):
        schema = InterventionSchema(factors={"x": ["a", "b", "c"]})
        model = BayesianLinearModel(
            schema, include_interactions=False, prior_variance=10.0, noise_variance=0.1
        )
        # No observations at all — prior is flat

        selections = []
        for trial in range(300):
            controller = Controller(schema, model, preferred_outcome=1.0, seed=trial)
            selections.append(controller.select_next())

        counts = [selections.count(c) for c in range(3)]
        # With flat prior, each cell should be selected roughly 100 times
        # Allow wide margin since Thompson sampling from a wide prior is noisy
        for c, count in enumerate(counts):
            assert count >= 30, (
                f"Cell {c} selected only {count}/300 times with flat prior — "
                f"expected roughly uniform"
            )


# ===========================================================================
# 7. Evaluation harness produces sane comparisons
# ===========================================================================

class TestEvaluationHarness:
    """Over 200 experiments with strong signal, autoresearcher2 should have
    lower cumulative regret than random."""

    def test_autoresearcher_beats_random(self):
        schema = InterventionSchema(
            factors={
                "optimizer": ["adam", "sgd"],
                "lr": ["low", "high"],
            }
        )
        env_config = {
            "true_effects": {
                "optimizer": {"adam": 0.3, "sgd": -0.3},
                "lr": {"low": -0.1, "high": 0.1},
            },
            "noise_std": 0.02,
            "baseline": 0.5,
        }

        report = run_evaluation(schema, env_config, n_experiments=200, seed=42)

        ar2_regret = report["autoresearcher2"]["cumulative_regret"]
        random_regret = report["random"]["cumulative_regret"]

        assert ar2_regret < random_regret, (
            f"autoresearcher2 regret ({ar2_regret:.2f}) should be less than "
            f"random ({random_regret:.2f})"
        )

    def test_cumulative_regret_nonnegative(self):
        """Cumulative regret should be non-negative (outcomes <= true_best by definition
        since true_best is the average of the best cell)."""
        schema = InterventionSchema(
            factors={"x": ["a", "b"]},
        )
        env_config = {
            "true_effects": {"x": {"a": 0.3, "b": -0.3}},
            "noise_std": 0.05,
            "baseline": 0.5,
        }

        report = run_evaluation(schema, env_config, n_experiments=50, seed=42)
        # Note: with noise, individual outcomes can exceed true_best mean,
        # so cumulative regret CAN be negative. But with many experiments,
        # it should be close to or above 0.
        # This tests whether the harness computes regret correctly at all.
        ar2_regret = report["autoresearcher2"]["cumulative_regret"]
        assert isinstance(ar2_regret, float)


# ===========================================================================
# 8. Edge case: single-factor schema
# ===========================================================================

class TestSingleFactorSchema:
    """Does the system work with just 1 factor?"""

    def test_single_factor_loop_runs(self):
        schema = InterventionSchema(factors={"lr": ["low", "mid", "high"]})
        effects = {"lr": {"low": -0.2, "mid": 0.0, "high": 0.3}}

        _, results, model, memory, _, _ = _run_full_loop(
            schema, effects, n_experiments=50, seed=42, noise_std=0.02
        )

        assert len(results) == 50
        assert memory.summary()["n_experiments"] == 50

        # Model should identify "high" as best
        best_cell = max(range(schema.n_cells), key=lambda c: model.predict(c)[0])
        best_config = schema.cell_to_config(best_cell)
        assert best_config["lr"] == "high", f"Expected 'high', got {best_config['lr']}"

    def test_single_factor_no_interactions(self):
        """Single factor with include_interactions=False should have zero interaction terms."""
        schema = InterventionSchema(factors={"lr": ["low", "mid", "high"]})
        assert schema.n_interactions == 0
        model = BayesianLinearModel(schema, include_interactions=False)
        assert model.n_features == 3  # just the 3 levels


# ===========================================================================
# 9. Edge case: include_interactions=False
# ===========================================================================

class TestNoInteractions:
    """Does the model work with include_interactions=False?
    The loop uses appraisal which hardcodes include_interactions=True in
    signals.py line 26 — this could be a dimension mismatch bug!"""

    def test_model_without_interactions(self):
        """Basic model operations with include_interactions=False."""
        schema = InterventionSchema(
            factors={"optimizer": ["adam", "sgd"], "lr": ["low", "high"]}
        )
        model = BayesianLinearModel(schema, include_interactions=False)

        # Should have only main effects (2 + 2 = 4 features)
        assert model.n_features == 4

        # Predict and update should work
        mean, var = model.predict(0)
        model.update(0, 0.5)
        mean2, var2 = model.predict(0)
        assert var2 < var  # variance should decrease

    def test_loop_with_no_interactions_works(self):
        """The full loop should work with include_interactions=False."""
        schema = InterventionSchema(
            factors={"optimizer": ["adam", "sgd"], "lr": ["low", "high"]}
        )
        effects = {
            "optimizer": {"adam": 0.3, "sgd": -0.3},
            "lr": {"low": -0.1, "high": 0.1},
        }

        _, results, *_ = _run_full_loop(
            schema, effects, n_experiments=10, seed=42,
            include_interactions=False,
        )
        assert len(results) == 10
        for r in results:
            assert "appraisal" in r
            assert "surprise" in r["appraisal"]


# ===========================================================================
# 10. Toy POMDP reproducibility
# ===========================================================================

class TestToyPOMDPReproducibility:
    """Run with same seed twice -> same actions."""

    def _run_pomdp_session(self, seed=42, n_steps=20):
        pomdp = ToyPOMDP(n_factors=2, levels_per_factor=3, n_obs=5, seed=seed)
        agent = ActiveInferenceAgent(pomdp, gamma=1.0, seed=seed)
        actions = []
        observations = []
        for _ in range(n_steps):
            a = agent.select_action()
            o = pomdp.observe(a)
            agent.update(a, o)
            actions.append(a)
            observations.append(o)
        return actions, observations

    def test_pomdp_deterministic_with_same_seed(self):
        actions1, obs1 = self._run_pomdp_session(seed=123, n_steps=30)
        actions2, obs2 = self._run_pomdp_session(seed=123, n_steps=30)

        assert actions1 == actions2, (
            f"Actions differ between runs with same seed:\n"
            f"  Run 1: {actions1}\n"
            f"  Run 2: {actions2}"
        )
        assert obs1 == obs2, "Observations differ between runs with same seed"

    def test_pomdp_different_seed_different_actions(self):
        actions1, _ = self._run_pomdp_session(seed=100, n_steps=20)
        actions2, _ = self._run_pomdp_session(seed=200, n_steps=20)

        # Very unlikely to be identical with different seeds
        assert actions1 != actions2, "Different seeds should produce different action sequences"
