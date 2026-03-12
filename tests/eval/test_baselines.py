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
    last_cells = [r["cell"] for r in results[-10:]]
    adam_count = sum(1 for c in last_cells if c in [0, 1])
    assert adam_count > 5
