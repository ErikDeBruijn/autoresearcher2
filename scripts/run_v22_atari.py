"""v2.2 Cross-Domain: apply autoresearcher2 to Atari game optimization.

Tests whether the same research methodology (Bayesian Thompson sampling,
appraisal signals, structured experimentation) works on a fundamentally
different domain than LLM training.

If the agent can optimize both GPT training AND Atari game scores →
evidence for fluid intelligence (domain-general research capability).

Schema:
    game         × [Breakout, SpaceInvaders, Pong]
    learning_rate × [1e-4, 5e-4, 1e-3]
    network_size  × [small, medium, large]

    = 27 cells (same grid size as GPT domain)

Usage:
    uv run python scripts/run_v22_atari.py
    uv run python scripts/run_v22_atari.py --n-experiments 10

Runs on GPU 0 (parallel with GPT experiments on GPU 1).
Expected runtime: ~30-60 min per experiment × 20 = ~10-20 hours
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.appraisal.signals import compute_appraisal
from autoresearcher2.research.atari_environment import AtariEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA = InterventionSchema(
    factors={
        "game": ["Breakout", "SpaceInvaders", "Pong"],
        "learning_rate": ["1e-4", "5e-4", "1e-3"],
        "network_size": ["small", "medium", "large"],
    }
)

N_EXPERIMENTS = 20
GPU_DEVICE = "0"
TOTAL_TIMESTEPS = 1_000_000
SEED = 42
RUN_ID = uuid.uuid4().hex[:8]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="v2.2 Cross-Domain Atari experiment")
    parser.add_argument("--n-experiments", type=int, default=N_EXPERIMENTS)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--gpu", type=str, default=GPU_DEVICE)
    args = parser.parse_args()

    n_experiments = args.n_experiments

    start = time.time()
    log("=" * 60)
    log(f"v2.2 CROSS-DOMAIN ATARI — run_id={RUN_ID}")
    log("=" * 60)
    log(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    log(f"Budget: {n_experiments} experiments")
    log(f"GPU: {args.gpu}, Timesteps: {args.total_timesteps}")

    env = AtariEnvironment(
        schema=SCHEMA,
        cuda_device=args.gpu,
        total_timesteps=args.total_timesteps,
    )
    model = BayesianLinearModel(SCHEMA, noise_variance=0.01, prior_variance=1.0)
    controller = Controller(SCHEMA, model, preferred_outcome=0.7, seed=SEED)
    memory = MemoryStore()
    results = []

    for i in range(n_experiments):
        cell = controller.select_next_lookahead()
        config = SCHEMA.cell_to_config(cell)
        scores = controller.score_cell(cell)
        log(f"  [{i+1}/{n_experiments}] cell {cell} {config} "
            f"(prag={scores['pragmatic']:.3f} epist={scores['epistemic']:.3f})")

        exp_start = time.time()
        try:
            outcome = env.run(cell)
            wall_time = time.time() - exp_start
            metadata = getattr(env, "last_run_metadata", {})

            snapshot_before = model.snapshot()
            model.update(cell, outcome)
            snapshot_after = model.snapshot()
            appraisal = compute_appraisal(SCHEMA, cell, outcome,
                                          snapshot_before, snapshot_after)
            memory.add(cell, outcome, appraisal)

            result = {
                "experiment": i,
                "cell": cell,
                "config": config,
                "outcome": outcome,
                "raw_reward": metadata.get("raw_mean_reward"),
                "wall_time_s": round(wall_time, 1),
                "training_time_s": metadata.get("training_time_s"),
                "fps": metadata.get("fps"),
                "scores": scores,
                "appraisal": {k: float(v) for k, v in appraisal.items()},
                "error": None,
            }
            log(f"    → reward={metadata.get('raw_mean_reward', '?')} "
                f"outcome={outcome:.4f} surprise={appraisal['surprise']:.4f}")
        except Exception as e:
            wall_time = time.time() - exp_start
            result = {
                "experiment": i,
                "cell": cell,
                "config": config,
                "outcome": None,
                "raw_reward": None,
                "wall_time_s": round(wall_time, 1),
                "scores": scores,
                "error": str(e),
            }
            log(f"    → FAILED: {e}")

        results.append(result)
        _save_results(results, n_experiments)

    # Summary
    total_time = time.time() - start
    successful = [r for r in results if r["error"] is None]
    log(f"\n{'=' * 60}")
    log(f"DONE — {total_time/60:.1f} min total")
    log(f"  {len(successful)}/{len(results)} experiments succeeded")

    if successful:
        best = max(successful, key=lambda r: r["outcome"])
        log(f"  Best: cell {best['cell']} {best['config']}")
        log(f"    reward={best['raw_reward']:.1f} outcome={best['outcome']:.4f}")

        # Factor importances
        importances = model.factor_importances()
        log("\n  Factor importances:")
        for factor, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            log(f"    {factor}: {imp:.4f}")

        # Best per game
        log("\n  Best per game:")
        for game in SCHEMA.factors["game"]:
            game_results = [r for r in successful if r["config"]["game"] == game]
            if game_results:
                best_game = max(game_results, key=lambda r: r["outcome"])
                log(f"    {game}: reward={best_game['raw_reward']:.1f} "
                    f"lr={best_game['config']['learning_rate']} "
                    f"net={best_game['config']['network_size']}")

    log(f"\nResults: artifacts/v22_atari/{RUN_ID}.json")


def _save_results(results: list[dict], n_experiments: int):
    out_dir = Path("artifacts/v22_atari")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "run_id": RUN_ID,
        "version": "v2.2",
        "schema": SCHEMA.factors,
        "n_experiments": n_experiments,
        "seed": SEED,
        "gpu_device": GPU_DEVICE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "results": results,
        "n_successful": sum(1 for r in results if r["error"] is None),
    }
    with open(out_dir / f"{RUN_ID}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
