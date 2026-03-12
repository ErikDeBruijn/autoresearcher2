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
    loop.run(n_experiments=100)
    importances = loop.model.factor_importances()
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

    first_10_ep = [r["scores"]["epistemic"] for r in results[:10]]
    last_10_ep = [r["scores"]["epistemic"] for r in results[-10:]]

    import numpy as np
    assert np.mean(last_10_ep) < np.mean(first_10_ep)
