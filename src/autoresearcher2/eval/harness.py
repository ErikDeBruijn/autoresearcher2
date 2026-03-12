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
    true_best_env = SyntheticEnvironment(schema=schema, **env_config, seed=0)
    true_outcomes = [
        np.mean([true_best_env.run(c) for _ in range(100)])
        for c in range(schema.n_cells)
    ]
    true_best = max(true_outcomes)

    report = {}

    # autoresearcher2
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    model = BayesianLinearModel(
        schema, noise_variance=env_config.get("noise_std", 0.05) ** 2
    )
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
