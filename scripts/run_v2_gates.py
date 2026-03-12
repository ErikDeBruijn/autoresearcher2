"""v2 Exit Gate Tests

Gate A: structured model > tabular model (structure matters)
Gate B: structured model stays strong on real interaction environment

Usage:
    uv run python scripts/run_v2_gates.py
"""

import json
import time
from pathlib import Path

import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop
from autoresearcher2.eval.baselines import (
    RandomAgent,
    GreedyAgent,
    TabularAgent,
    run_baseline,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS = 20
N_EXPERIMENTS = 50

SCHEMA = InterventionSchema(
    factors={
        "optimizer": ["adam", "adamw", "sgd"],
        "lr": ["1e-4", "3e-4", "1e-3"],
        "batch_size": ["32", "64", "128"],
    }
)

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def compute_true_best(schema, env_config, n_samples=500):
    env = SyntheticEnvironment(schema=schema, **env_config, seed=0)
    cell_means = []
    for c in range(schema.n_cells):
        outcomes = [env.run(c) for _ in range(n_samples)]
        cell_means.append(np.mean(outcomes))
    best_cell = int(np.argmax(cell_means))
    return float(max(cell_means)), best_cell


def run_autoresearcher2(schema, env_config, seed, n_experiments):
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    model = BayesianLinearModel(
        schema, noise_variance=env_config["noise_std"] ** 2
    )
    controller = Controller(schema, model, preferred_outcome=1.0, seed=seed)
    memory = MemoryStore()
    loop = ResearchLoop(schema, model, controller, memory, env)
    results = loop.run(n_experiments)
    outcomes = [r["outcome"] for r in results]
    return outcomes


def run_agent(agent_cls, schema, env_config, seed, n_experiments):
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    agent = agent_cls(schema, seed=seed)
    results = run_baseline(agent, env, n_experiments)
    return [r["outcome"] for r in results]


def cumulative_regret(outcomes, true_best):
    return sum(true_best - o for o in outcomes)


# ---------------------------------------------------------------------------
# Gate A: structured > tabular
# ---------------------------------------------------------------------------


def run_gate_a():
    print("=" * 60)
    print("GATE A: Structure Matters (structured > tabular)")
    print("=" * 60)

    env_config = {
        "true_effects": {
            "optimizer": {"adam": 0.2, "adamw": 0.25, "sgd": -0.2},
            "lr": {"1e-4": -0.1, "3e-4": 0.15, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        "baseline": 0.5,
        "noise_std": 0.03,
    }

    true_best, best_cell = compute_true_best(SCHEMA, env_config)
    print(f"True best: {true_best:.4f} at cell {best_cell}")

    ar2_regrets = []
    tab_regrets = []
    random_regrets = []

    for seed in range(N_SEEDS):
        ar2_outcomes = run_autoresearcher2(
            SCHEMA, env_config, seed, N_EXPERIMENTS
        )
        tab_outcomes = run_agent(
            TabularAgent, SCHEMA, env_config, seed + 3000, N_EXPERIMENTS
        )
        rand_outcomes = run_agent(
            RandomAgent, SCHEMA, env_config, seed + 1000, N_EXPERIMENTS
        )

        ar2_regrets.append(cumulative_regret(ar2_outcomes, true_best))
        tab_regrets.append(cumulative_regret(tab_outcomes, true_best))
        random_regrets.append(cumulative_regret(rand_outcomes, true_best))

    ar2_mean = np.mean(ar2_regrets)
    tab_mean = np.mean(tab_regrets)
    rand_mean = np.mean(random_regrets)

    print(f"\n  autoresearcher2: regret {ar2_mean:.1f} ± {np.std(ar2_regrets):.1f}")
    print(f"  tabular (UCB1):  regret {tab_mean:.1f} ± {np.std(tab_regrets):.1f}")
    print(f"  random:          regret {rand_mean:.1f} ± {np.std(random_regrets):.1f}")

    passes = ar2_mean < tab_mean
    print(f"\n  structured < tabular: {'PASS' if passes else 'FAIL'} ({ar2_mean:.1f} vs {tab_mean:.1f})")

    return {
        "gate": "A",
        "name": "structure_matters",
        "passes": passes,
        "ar2_regret": float(ar2_mean),
        "tabular_regret": float(tab_mean),
        "random_regret": float(rand_mean),
    }


# ---------------------------------------------------------------------------
# Gate B: interaction environment
# ---------------------------------------------------------------------------


def run_gate_b():
    print("\n" + "=" * 60)
    print("GATE B: Interaction Matters (structured stays strong)")
    print("=" * 60)

    # Real interaction: adamw + 3e-4 has a synergy bonus of +0.15
    # This means the true best is NOT just sum of best main effects
    # Without interaction: best main = adamw(0.25) + 3e-4(0.15) + 64(0.05) = 0.95
    # With interaction: adamw+3e-4 gets +0.15 extra = 1.10 (clipped to 1.0)
    # But adam+1e-3 also gets an interaction: +0.10
    # This creates a landscape where main effects alone are misleading
    env_config = {
        "true_effects": {
            "optimizer": {"adam": 0.1, "adamw": 0.1, "sgd": -0.2},
            "lr": {"1e-4": -0.05, "3e-4": 0.05, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        "interactions": {
            ("optimizer", "adamw", "lr", "3e-4"): 0.20,
            ("optimizer", "adam", "lr", "1e-3"): 0.10,
            ("optimizer", "sgd", "lr", "1e-4"): -0.10,
        },
        "baseline": 0.5,
        "noise_std": 0.03,
    }

    true_best, best_cell = compute_true_best(SCHEMA, env_config)
    best_config = SCHEMA.cell_to_config(best_cell)
    print(f"True best: {true_best:.4f} at cell {best_cell} = {best_config}")

    ar2_regrets = []
    tab_regrets = []
    greedy_regrets = []
    random_regrets = []

    for seed in range(N_SEEDS):
        ar2_outcomes = run_autoresearcher2(
            SCHEMA, env_config, seed, N_EXPERIMENTS
        )
        tab_outcomes = run_agent(
            TabularAgent, SCHEMA, env_config, seed + 3000, N_EXPERIMENTS
        )
        greedy_outcomes = run_agent(
            GreedyAgent, SCHEMA, env_config, seed + 2000, N_EXPERIMENTS
        )
        rand_outcomes = run_agent(
            RandomAgent, SCHEMA, env_config, seed + 1000, N_EXPERIMENTS
        )

        ar2_regrets.append(cumulative_regret(ar2_outcomes, true_best))
        tab_regrets.append(cumulative_regret(tab_outcomes, true_best))
        greedy_regrets.append(cumulative_regret(greedy_outcomes, true_best))
        random_regrets.append(cumulative_regret(rand_outcomes, true_best))

    ar2_mean = np.mean(ar2_regrets)
    tab_mean = np.mean(tab_regrets)
    greedy_mean = np.mean(greedy_regrets)
    rand_mean = np.mean(random_regrets)

    print(f"\n  autoresearcher2: regret {ar2_mean:.1f} ± {np.std(ar2_regrets):.1f}")
    print(f"  tabular (UCB1):  regret {tab_mean:.1f} ± {np.std(tab_regrets):.1f}")
    print(f"  greedy:          regret {greedy_mean:.1f} ± {np.std(greedy_regrets):.1f}")
    print(f"  random:          regret {rand_mean:.1f} ± {np.std(random_regrets):.1f}")

    beats_random = ar2_mean < rand_mean
    beats_tabular = ar2_mean < tab_mean
    competitive_greedy = ar2_mean <= greedy_mean * 1.5

    print(f"\n  beats random:       {'PASS' if beats_random else 'FAIL'} ({ar2_mean:.1f} vs {rand_mean:.1f})")
    print(f"  beats tabular:      {'PASS' if beats_tabular else 'FAIL'} ({ar2_mean:.1f} vs {tab_mean:.1f})")
    print(f"  competitive greedy: {'PASS' if competitive_greedy else 'FAIL'} ({ar2_mean:.1f} vs {greedy_mean:.1f})")

    passes = beats_random and competitive_greedy
    print(f"\n  GATE B overall: {'PASS' if passes else 'FAIL'}")

    return {
        "gate": "B",
        "name": "interaction_environment",
        "passes": passes,
        "ar2_regret": float(ar2_mean),
        "tabular_regret": float(tab_mean),
        "greedy_regret": float(greedy_mean),
        "random_regret": float(rand_mean),
        "true_best": float(true_best),
        "best_config": best_config,
        "beats_random": beats_random,
        "beats_tabular": beats_tabular,
        "competitive_greedy": competitive_greedy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start = time.time()

    artifacts_dir = Path("artifacts/v2_gates")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["gate_a"] = run_gate_a()
    results["gate_b"] = run_gate_b()

    output_path = artifacts_dir / "gate_ab_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Gate A (structure matters):    {'PASS' if results['gate_a']['passes'] else 'FAIL'}")
    print(f"  Gate B (interaction env):      {'PASS' if results['gate_b']['passes'] else 'FAIL'}")
    print(f"  Gate C (real substrate):       not tested here — see scripts/run_gate_c_smoke.py")
    print(f"\nResults: {output_path}")
    print(f"Completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
