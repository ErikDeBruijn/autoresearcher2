"""v1 Synthetic Validation Runner

Runs autoresearcher2, random, and greedy across multiple synthetic environments
and seeds. Writes JSON results to artifacts/v1_validation/ and prints summary.

Usage:
    uv run python scripts/run_v1_synthetic_validation.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.synthetic_environment import SyntheticEnvironment
from autoresearcher2.core.loop import ResearchLoop
from autoresearcher2.eval.baselines import RandomAgent, GreedyAgent, run_baseline

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

ENVIRONMENTS = {
    "env_a_main_effects": {
        "description": "Main effects dominant, low noise",
        "true_effects": {
            "optimizer": {"adam": 0.2, "adamw": 0.25, "sgd": -0.2},
            "lr": {"1e-4": -0.1, "3e-4": 0.15, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        "baseline": 0.5,
        "noise_std": 0.03,
    },
    "env_b_interaction": {
        "description": "Main effects + interaction (adamw + 3e-4 synergy)",
        "true_effects": {
            "optimizer": {"adam": 0.15, "adamw": 0.15, "sgd": -0.2},
            "lr": {"1e-4": -0.1, "3e-4": 0.1, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        # Note: v1 SyntheticEnvironment doesn't support interaction terms directly.
        # We simulate it by making adamw+3e-4 the clear best through main effects.
        # True interaction testing requires extending the environment (post-v1).
        "baseline": 0.5,
        "noise_std": 0.03,
    },
    "env_c_high_noise": {
        "description": "Same structure, high noise — harder to learn",
        "true_effects": {
            "optimizer": {"adam": 0.2, "adamw": 0.25, "sgd": -0.2},
            "lr": {"1e-4": -0.1, "3e-4": 0.15, "1e-3": 0.0},
            "batch_size": {"32": -0.05, "64": 0.05, "128": 0.0},
        },
        "baseline": 0.5,
        "noise_std": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def compute_true_best(schema, env_config, n_samples=500):
    """Estimate the true best outcome by averaging many samples per cell."""
    env = SyntheticEnvironment(schema=schema, **env_config, seed=0)
    cell_means = []
    for c in range(schema.n_cells):
        outcomes = [env.run(c) for _ in range(n_samples)]
        cell_means.append(np.mean(outcomes))
    best_cell = int(np.argmax(cell_means))
    return float(max(cell_means)), best_cell, cell_means


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
    cells_visited = [r["cell"] for r in results]
    epistemic_scores = [r["scores"]["epistemic"] for r in results]
    appraisal_events = [r["appraisal"] for r in results]

    return {
        "outcomes": outcomes,
        "cells_visited": cells_visited,
        "epistemic_scores": epistemic_scores,
        "appraisal_events": appraisal_events,
        "factor_importances": model.factor_importances(),
        "n_unique_cells": len(set(cells_visited)),
    }


def run_random_agent(schema, env_config, seed, n_experiments):
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    agent = RandomAgent(schema, seed=seed)
    results = run_baseline(agent, env, n_experiments)
    outcomes = [r["outcome"] for r in results]
    cells_visited = [r["cell"] for r in results]
    return {
        "outcomes": outcomes,
        "cells_visited": cells_visited,
        "n_unique_cells": len(set(cells_visited)),
    }


def run_greedy_agent(schema, env_config, seed, n_experiments):
    env = SyntheticEnvironment(schema=schema, **env_config, seed=seed)
    agent = GreedyAgent(schema, seed=seed)
    results = run_baseline(agent, env, n_experiments)
    outcomes = [r["outcome"] for r in results]
    cells_visited = [r["cell"] for r in results]
    return {
        "outcomes": outcomes,
        "cells_visited": cells_visited,
        "n_unique_cells": len(set(cells_visited)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(outcomes, true_best):
    cumulative_regret = sum(true_best - o for o in outcomes)
    best_so_far = []
    current_best = -np.inf
    for o in outcomes:
        current_best = max(current_best, o)
        best_so_far.append(current_best)
    return {
        "best_outcome": float(max(outcomes)),
        "mean_outcome": float(np.mean(outcomes)),
        "cumulative_regret": float(cumulative_regret),
        "best_so_far": [float(x) for x in best_so_far],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_environment(env_name, env_config, schema, n_seeds, n_experiments):
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Description: {env_config['description']}")
    print(f"Seeds: {n_seeds}, Budget: {n_experiments}")
    print(f"Schema: {schema.n_cells} cells, {schema.n_factors} factors")
    print(f"{'='*60}")

    # Stripped env_config for passing to SyntheticEnvironment
    # (remove 'description' key)
    clean_config = {k: v for k, v in env_config.items() if k != "description"}

    true_best, best_cell, cell_means = compute_true_best(schema, clean_config)
    best_config = schema.cell_to_config(best_cell)
    print(f"True best: {true_best:.4f} at cell {best_cell} = {best_config}")

    agent_results = {"autoresearcher2": [], "random": [], "greedy": []}

    for seed in range(n_seeds):
        ar2 = run_autoresearcher2(schema, clean_config, seed, n_experiments)
        rand = run_random_agent(schema, clean_config, seed + 1000, n_experiments)
        grdy = run_greedy_agent(schema, clean_config, seed + 2000, n_experiments)

        agent_results["autoresearcher2"].append(ar2)
        agent_results["random"].append(rand)
        agent_results["greedy"].append(grdy)

    # Aggregate metrics
    report = {
        "env_name": env_name,
        "env_config": env_config,
        "schema_config": {
            "factors": schema.factors,
            "n_cells": schema.n_cells,
        },
        "true_best": true_best,
        "best_cell": best_cell,
        "best_config": best_config,
        "n_seeds": n_seeds,
        "n_experiments": n_experiments,
        "agents": {},
    }

    for agent_name, runs in agent_results.items():
        all_outcomes = [r["outcomes"] for r in runs]
        metrics_per_seed = [
            compute_metrics(outcomes, true_best) for outcomes in all_outcomes
        ]

        best_outcomes = [m["best_outcome"] for m in metrics_per_seed]
        regrets = [m["cumulative_regret"] for m in metrics_per_seed]
        unique_cells = [r["n_unique_cells"] for r in runs]

        agent_report = {
            "mean_best_outcome": float(np.mean(best_outcomes)),
            "std_best_outcome": float(np.std(best_outcomes)),
            "mean_cumulative_regret": float(np.mean(regrets)),
            "std_cumulative_regret": float(np.std(regrets)),
            "mean_unique_cells": float(np.mean(unique_cells)),
        }

        # Autoresearcher2-specific metrics
        if agent_name == "autoresearcher2":
            # Factor importances (averaged over seeds)
            all_importances = [r["factor_importances"] for r in runs]
            avg_importances = {}
            for key in all_importances[0]:
                vals = [imp[key] for imp in all_importances]
                avg_importances[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
            agent_report["factor_importances"] = avg_importances

            # Factor importance rank correctness
            # Check: does the strongest ground-truth factor get highest importance?
            gt_effects = clean_config["true_effects"]
            gt_ranges = {}
            for fname, levels in gt_effects.items():
                vals = list(levels.values())
                gt_ranges[fname] = max(vals) - min(vals)
            gt_top_factor = max(gt_ranges, key=gt_ranges.get)

            n_correct = 0
            for imp in all_importances:
                learned_top = max(imp, key=imp.get)
                if learned_top == gt_top_factor:
                    n_correct += 1
            agent_report["factor_rank_accuracy"] = float(n_correct / n_seeds)
            agent_report["gt_top_factor"] = gt_top_factor

            # Epistemic decrease check
            all_epistemic = [r["epistemic_scores"] for r in runs]
            first_quarter = [
                float(np.mean(ep[: n_experiments // 4])) for ep in all_epistemic
            ]
            last_quarter = [
                float(np.mean(ep[-n_experiments // 4 :])) for ep in all_epistemic
            ]
            n_decreased = sum(
                1 for f, l in zip(first_quarter, last_quarter) if l < f
            )
            agent_report["epistemic_decreased_pct"] = float(
                n_decreased / n_seeds
            )

            # Top appraisal events (from seed 0 for manual inspection)
            seed0_appraisals = runs[0]["appraisal_events"]
            seed0_cells = runs[0]["cells_visited"]
            seed0_outcomes = runs[0]["outcomes"]
            events_with_context = []
            for i, (ap, cell, outcome) in enumerate(
                zip(seed0_appraisals, seed0_cells, seed0_outcomes)
            ):
                events_with_context.append(
                    {
                        "experiment": i,
                        "cell": cell,
                        "config": schema.cell_to_config(cell),
                        "outcome": float(outcome),
                        "surprise": float(ap["surprise"]),
                        "theory_conflict": float(ap["theory_conflict"]),
                        "learntropy": float(ap["learntropy"]),
                        "prediction_impact_breadth": float(
                            ap["prediction_impact_breadth"]
                        ),
                    }
                )
            top_learntropy = sorted(
                events_with_context, key=lambda x: x["learntropy"], reverse=True
            )[:10]
            agent_report["top_learntropy_events_seed0"] = top_learntropy

        report["agents"][agent_name] = agent_report

    # Print summary
    print()
    for agent_name, ar in report["agents"].items():
        bo = ar["mean_best_outcome"]
        bo_std = ar["std_best_outcome"]
        reg = ar["mean_cumulative_regret"]
        reg_std = ar["std_cumulative_regret"]
        uc = ar["mean_unique_cells"]
        print(f"  {agent_name}:")
        print(f"    mean_best_outcome: {bo:.4f} +/- {bo_std:.4f}")
        print(f"    mean_regret:       {reg:.2f} +/- {reg_std:.2f}")
        print(f"    mean_unique_cells: {uc:.1f}")
        if "factor_rank_accuracy" in ar:
            print(f"    factor_rank_accuracy: {ar['factor_rank_accuracy']:.0%}")
            print(f"    epistemic_decreased:  {ar['epistemic_decreased_pct']:.0%}")
        print()

    return report


def main():
    start = time.time()

    artifacts_dir = Path("artifacts/v1_validation")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}

    for env_name, env_config in ENVIRONMENTS.items():
        report = run_environment(
            env_name, env_config, SCHEMA, N_SEEDS, N_EXPERIMENTS
        )
        all_reports[env_name] = report

    # Write full results
    output_path = artifacts_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nFull results written to: {output_path}")

    # Print exit criteria summary
    print(f"\n{'='*60}")
    print("EXIT CRITERIA CHECK")
    print(f"{'='*60}")

    all_pass = True

    for env_name, report in all_reports.items():
        ar2 = report["agents"]["autoresearcher2"]
        rand = report["agents"]["random"]
        greedy = report["agents"]["greedy"]

        beats_random = ar2["mean_cumulative_regret"] < rand["mean_cumulative_regret"]
        competitive_greedy = ar2["mean_cumulative_regret"] <= greedy["mean_cumulative_regret"] * 1.2
        factor_correct = ar2.get("factor_rank_accuracy", 0) >= 0.8
        epistemic_drops = ar2.get("epistemic_decreased_pct", 0) >= 0.8

        print(f"\n{env_name}:")
        print(f"  beats_random:        {'PASS' if beats_random else 'FAIL'} (regret {ar2['mean_cumulative_regret']:.1f} vs {rand['mean_cumulative_regret']:.1f})")
        print(f"  competitive_greedy:  {'PASS' if competitive_greedy else 'FAIL'} (regret {ar2['mean_cumulative_regret']:.1f} vs {greedy['mean_cumulative_regret']:.1f})")
        print(f"  factor_rank>=80%:    {'PASS' if factor_correct else 'FAIL'} ({ar2.get('factor_rank_accuracy', 0):.0%})")
        print(f"  epistemic_drops>=80%: {'PASS' if epistemic_drops else 'FAIL'} ({ar2.get('epistemic_decreased_pct', 0):.0%})")

        if not all([beats_random, competitive_greedy, factor_correct, epistemic_drops]):
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("OVERALL: ALL EXIT CRITERIA PASS")
    else:
        print("OVERALL: SOME EXIT CRITERIA FAILED — v1 is nog niet geslaagd")
    print(f"{'='*60}")
    print(f"\nCompleted in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
