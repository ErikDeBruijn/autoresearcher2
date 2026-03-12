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
    assert np.std(outcomes) > 0.01
