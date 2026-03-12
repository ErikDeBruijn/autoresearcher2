"""Run autoresearch-style baseline on real train.py.

Same schema, same hardware, same budget as the autoresearcher2 LLM loops.
The only difference: no Bayesian model, no factor importances, no appraisal.
The LLM sees a flat results log and suggests what to try next.

Usage:
    uv run python scripts/run_autoresearch_baseline.py

Expected runtime: ~6 min per experiment × 20 = ~2 hours
"""

import json
import time
from pathlib import Path

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.llm.autoresearch_baseline import AutoresearchLLMAgent
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment

# ---------------------------------------------------------------------------
# Configuration — same as the LLM-augmented loops for fair comparison
# ---------------------------------------------------------------------------

SCHEMA = InterventionSchema(
    factors={
        "DEPTH": ["6", "8", "10"],
        "MATRIX_LR": ["0.02", "0.04", "0.08"],
        "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
    }
)

N_EXPERIMENTS = 20
GPU_DEVICE = "1"


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    start = time.time()
    log("=" * 60)
    log("autoresearch-style baseline — flat LLM on real train.py")
    log("=" * 60)
    log(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    log(f"Budget: {N_EXPERIMENTS} experiments")
    log(f"GPU: {GPU_DEVICE}")
    log(f"Factors: {SCHEMA.factors}")
    log("No Bayesian model, no factor importances, no appraisal")
    log("")

    env = TrainPyEnvironment(schema=SCHEMA, cuda_device=GPU_DEVICE)
    agent = AutoresearchLLMAgent(schema=SCHEMA, seed=42)

    all_results = []

    for i in range(N_EXPERIMENTS):
        exp_start = time.time()
        log(f"\n--- Experiment {i+1}/{N_EXPERIMENTS} ---")

        cell = agent.select_next()
        config = SCHEMA.cell_to_config(cell)
        log(f"  Selected: cell {cell} = {config}")

        try:
            outcome = env.run(cell)
            val_bpb = 2.0 - outcome
            wall_time = time.time() - exp_start
            log(f"  Result: val_bpb={val_bpb:.6f} ({wall_time:.0f}s)")

            agent.observe(cell, outcome)

            all_results.append({
                "experiment": i,
                "cell": cell,
                "config": config,
                "val_bpb": val_bpb,
                "outcome": outcome,
                "wall_time_s": round(wall_time, 1),
                "label": "autoresearch_baseline",
                "error": None,
            })
        except Exception as e:
            wall_time = time.time() - exp_start
            log(f"  FAILED: {e} ({wall_time:.0f}s)")
            all_results.append({
                "experiment": i,
                "cell": cell,
                "config": config,
                "val_bpb": None,
                "outcome": None,
                "wall_time_s": round(wall_time, 1),
                "label": "autoresearch_baseline",
                "error": str(e),
            })

        # Show progress
        successful = [r for r in all_results if r["error"] is None]
        if successful:
            best = min(r["val_bpb"] for r in successful)
            log(f"  Best so far: {best:.6f} ({len(successful)} successful)")

    # Summary
    total_time = time.time() - start
    log(f"\n{'=' * 60}")
    log("FINAL SUMMARY — autoresearch-style baseline")
    log(f"{'=' * 60}")

    successful = [r for r in all_results if r["error"] is None]
    if successful:
        best = min(successful, key=lambda r: r["val_bpb"])
        log(f"  Best val_bpb: {best['val_bpb']:.6f} at {best['config']}")
        log(f"  Worst val_bpb: {max(r['val_bpb'] for r in successful):.6f}")
        log(f"  Mean val_bpb: {sum(r['val_bpb'] for r in successful)/len(successful):.6f}")
        log("")
        log("  All runs (sorted by val_bpb):")
        for r in sorted(successful, key=lambda r: r["val_bpb"]):
            log(f"    {r['config']} → {r['val_bpb']:.6f}")

    log(f"\n  Unique cells visited: {len(set(r['cell'] for r in all_results))}")
    log(f"  Successful: {len(successful)}/{len(all_results)}")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    artifacts_dir = Path("artifacts/autoresearch_baseline")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "schema": SCHEMA.factors,
            "n_experiments": N_EXPERIMENTS,
            "approach": "autoresearch_baseline",
            "description": "LLM-only with flat results log, no Bayesian model",
            "results": all_results,
            "total_time_s": round(total_time, 1),
        }, f, indent=2, default=str)
    log(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
