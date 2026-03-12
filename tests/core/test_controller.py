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
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    selections = {controller.select_next() for _ in range(50)}
    assert len(selections) > 1


def test_controller_exploits_after_learning():
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    for _ in range(30):
        model.update(cell_index=0, outcome=0.95)
        model.update(cell_index=1, outcome=0.3)
        model.update(cell_index=2, outcome=0.2)
        model.update(cell_index=3, outcome=0.25)
    selections = [controller.select_next() for _ in range(50)]
    cell_0_count = selections.count(0)
    assert cell_0_count > 30


def test_controller_scores_decomposition():
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    scores = controller.score_all_cells()
    assert len(scores) == schema.n_cells
    for cell_idx, score in scores.items():
        assert "pragmatic" in score
        assert "epistemic" in score
        assert "total" in score


def test_epistemic_decreases_with_data():
    schema, model = make_setup()
    controller = Controller(schema, model, preferred_outcome=1.0, seed=42)
    scores_before = controller.score_all_cells()
    ep_before = scores_before[0]["epistemic"]
    model.update(cell_index=0, outcome=0.5)
    scores_after = controller.score_all_cells()
    ep_after = scores_after[0]["epistemic"]
    assert ep_after < ep_before
