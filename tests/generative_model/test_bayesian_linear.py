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
    np.testing.assert_array_equal(model.mu_w, np.zeros(model.n_features))
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
    assert abs(mean_after - 1.0) < abs(mean_before - 1.0)


def test_generalization_across_cells():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    _, var_cell1_before = model.predict(cell_index=1)
    for _ in range(5):
        model.update(cell_index=0, outcome=0.9)
    _, var_cell1_after = model.predict(cell_index=1)
    assert var_cell1_after < var_cell1_before


def test_factor_importance():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    for _ in range(20):
        model.update(cell_index=0, outcome=0.9)
        model.update(cell_index=1, outcome=0.85)
        model.update(cell_index=2, outcome=0.5)
        model.update(cell_index=3, outcome=0.55)
    importances = model.factor_importances()
    assert "optimizer" in importances
    assert "lr" in importances
    assert importances["optimizer"] > importances["lr"]


def test_snapshot_is_independent_copy():
    schema = make_schema()
    model = BayesianLinearModel(schema)
    snap = model.snapshot()
    assert "mu_w" in snap and "sigma_w" in snap and "noise_variance" in snap
    model.update(cell_index=0, outcome=0.8)
    assert not np.allclose(model.mu_w, snap["mu_w"])
