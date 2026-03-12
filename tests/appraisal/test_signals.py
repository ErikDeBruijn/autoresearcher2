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
    schema, model = make_setup()
    snapshot_before = model.snapshot()
    mu_before = model.mu_w.copy()
    model.update(cell_index=0, outcome=0.5)
    snapshot_after = model.snapshot()
    compute_appraisal(
        schema, cell_index=0, outcome=0.5,
        snapshot_before=snapshot_before, snapshot_after=snapshot_after,
    )
    np.testing.assert_array_equal(model.mu_w, snapshot_after["mu_w"])


def test_surprising_outcome_scores_higher_than_expected():
    schema, model = make_setup()
    for _ in range(20):
        model.update(cell_index=0, outcome=0.9)

    snap_before = model.snapshot()
    model_copy_surprising = BayesianLinearModel(schema)
    model_copy_surprising.mu_w = snap_before["mu_w"].copy()
    model_copy_surprising.sigma_w = snap_before["sigma_w"].copy()
    model_copy_surprising.update(cell_index=0, outcome=0.1)
    surprising = compute_appraisal(
        schema, 0, 0.1, snap_before, model_copy_surprising.snapshot(),
    )

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
    for _ in range(50):
        model.update(cell_index=0, outcome=0.9)
    snap_confident = model.snapshot()
    model.update(cell_index=0, outcome=0.1)
    conflict_confident = compute_appraisal(
        schema, 0, 0.1, snap_confident, model.snapshot(),
    )

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
