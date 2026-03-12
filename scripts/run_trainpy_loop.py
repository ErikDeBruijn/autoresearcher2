"""First real autoresearcher2 loop on train.py.

27-cell schema (3 factors × 3 levels), real training runs,
Bayesian model updates after each experiment. Same schema size
as synthetic validation — now on real GPU training.

Usage:
    uv run python scripts/run_trainpy_loop.py

Expected runtime: ~6 min per experiment × 15 = ~1.5 hours
"""

import json
import time
from pathlib import Path

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment
from autoresearcher2.core.loop import ResearchLoop

# ---------------------------------------------------------------------------
# Configuration — deliberately tiny
# ---------------------------------------------------------------------------

SCHEMA = InterventionSchema(
    factors={
        "DEPTH": ["6", "8", "10"],
        "MATRIX_LR": ["0.02", "0.04", "0.08"],
        "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
    }
)

N_EXPERIMENTS = 15  # ~1.5 hours, covers 15 of 27 cells

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start = time.time()
    print("=" * 60)
    print("autoresearcher2 — Real train.py Loop")
    print("=" * 60)
    print(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    print(f"Budget: {N_EXPERIMENTS} experiments (~{N_EXPERIMENTS * 6} min)")
    print(f"Factors: {SCHEMA.factors}")
    print()

    env = TrainPyEnvironment(schema=SCHEMA, cuda_device="0")

    # Higher noise variance for real experiments (noisy GPU training)
    model = BayesianLinearModel(
        schema=SCHEMA,
        noise_variance=0.01,  # expect ~0.01 variance in transformed val_bpb
        prior_variance=1.0,
    )
    controller = Controller(
        schema=SCHEMA,
        model=model,
        preferred_outcome=1.0,  # outcome = 2.0 - val_bpb, so 1.0 means val_bpb=1.0
        seed=42,
    )
    memory = MemoryStore()
    loop = ResearchLoop(SCHEMA, model, controller, memory, env)

    # Run loop with detailed logging
    all_results = []
    for i in range(N_EXPERIMENTS):
        exp_start = time.time()
        print(f"\n--- Experiment {i+1}/{N_EXPERIMENTS} ---")

        # Select
        cell = controller.select_next()
        config = SCHEMA.cell_to_config(cell)
        scores = controller.score_cell(cell)
        print(f"  Selected: cell {cell} = {config}")
        print(f"  Scores: pragmatic={scores['pragmatic']:.4f} epistemic={scores['epistemic']:.4f}")

        # Run
        try:
            outcome = env.run(cell)
            val_bpb = 2.0 - outcome
            print(f"  Result: val_bpb={val_bpb:.6f} (outcome={outcome:.6f})")
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            # Log failure but continue
            all_results.append({
                "experiment": i,
                "cell": cell,
                "config": config,
                "error": str(e),
                "wall_time_s": round(time.time() - exp_start, 1),
            })
            continue

        # Snapshot, update, appraise
        snapshot_before = model.snapshot()
        model.update(cell, outcome)
        snapshot_after = model.snapshot()

        from autoresearcher2.appraisal.signals import compute_appraisal
        appraisal = compute_appraisal(
            SCHEMA, cell, outcome, snapshot_before, snapshot_after
        )
        memory.add(cell, outcome, appraisal)

        print(f"  Appraisal: surprise={appraisal['surprise']:.4f} "
              f"learntropy={appraisal['learntropy']:.4f}")

        all_results.append({
            "experiment": i,
            "cell": cell,
            "config": config,
            "val_bpb": val_bpb,
            "outcome": outcome,
            "scores": scores,
            "appraisal": {k: float(v) for k, v in appraisal.items()},
            "wall_time_s": round(time.time() - exp_start, 1),
        })

        # Show factor importances so far
        importances = model.factor_importances()
        print(f"  Factor importances: {importances}")

    # Summary
    total_time = time.time() - start
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    successful = [r for r in all_results if "val_bpb" in r]
    if successful:
        best = min(successful, key=lambda r: r["val_bpb"])
        print(f"  Best val_bpb: {best['val_bpb']:.6f} at {best['config']}")
        print(f"  Worst val_bpb: {max(r['val_bpb'] for r in successful):.6f}")

        print(f"\n  All runs:")
        for r in all_results:
            if "val_bpb" in r:
                print(f"    exp {r['experiment']}: cell {r['cell']} {r['config']} "
                      f"→ val_bpb={r['val_bpb']:.6f} ({r['wall_time_s']}s)")
            else:
                print(f"    exp {r['experiment']}: cell {r['cell']} {r['config']} "
                      f"→ FAILED ({r.get('wall_time_s', '?')}s)")

    print(f"\n  Factor importances: {model.factor_importances()}")
    print(f"  Unique cells visited: {len(set(r['cell'] for r in all_results))}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    artifacts_dir = Path("artifacts/trainpy_loop")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "first_real_loop.json"
    with open(output_path, "w") as f:
        json.dump({
            "schema": SCHEMA.factors,
            "n_experiments": N_EXPERIMENTS,
            "results": all_results,
            "factor_importances": model.factor_importances(),
            "total_time_s": round(total_time, 1),
        }, f, indent=2, default=str)
    print(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
