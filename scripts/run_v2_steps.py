"""v2.0 research steps experiment runner.

Runs the v2.0 research loop where the LLM proposes mixed research steps
(experiments, analyses, hypotheses, schema changes) instead of just configs.

Usage:
    uv run python scripts/run_v2_steps.py [--n-iterations 10]

Expected: each iteration produces 0-3 experiments plus analyses/hypotheses.
The LLM decides the mix based on what it learns.
"""

import argparse
import json
import time
import uuid
from pathlib import Path

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment
from autoresearcher2.core.research_loop_v2 import ResearchLoopV2

SCHEMA = InterventionSchema(
    factors={
        "DEPTH": ["6", "8", "10"],
        "MATRIX_LR": ["0.02", "0.04", "0.08"],
        "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
    }
)

GPU_DEVICE = "1"
SEED = 42
RUN_ID = uuid.uuid4().hex[:8]
ARTIFACTS_DIR = Path("artifacts/v2-steps")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="v2.0 research steps experiment")
    parser.add_argument("--n-iterations", type=int, default=10,
                        help="Number of LLM consultation rounds (each produces 1-3 steps)")
    parser.add_argument("--gpu", type=str, default=GPU_DEVICE)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Starting v2.0 research steps run {RUN_ID}")
    log(f"Schema: {SCHEMA.n_cells} cells, iterations: {args.n_iterations}")

    # Initialize components
    model = BayesianLinearModel(SCHEMA, noise_variance=0.01)
    controller = Controller(SCHEMA, model, preferred_outcome=1.0, seed=SEED)
    memory = MemoryStore()
    env = TrainPyEnvironment(
        schema=SCHEMA,
        ssh_host="root@dllm-experiment.home",
        ssh_key="~/.ssh/pve03_key",
        cuda_device=args.gpu,
    )

    loop = ResearchLoopV2(
        schema=SCHEMA,
        model=model,
        controller=controller,
        memory=memory,
        env=env,
    )

    # Run
    start = time.time()
    results = loop.run(args.n_iterations)
    elapsed = time.time() - start

    # Report
    log(f"\n=== v2.0 Run Complete ===")
    log(f"Iterations: {args.n_iterations}")
    log(f"Experiments completed: {len(results)}")
    log(f"Total steps executed: {len(loop.step_log)}")
    log(f"Wall time: {elapsed/60:.1f} min")

    if results:
        outcomes = [r["outcome"] for r in results]
        val_bpbs = [2.0 - o for o in outcomes]
        log(f"Best val_bpb: {min(val_bpbs):.6f}")
        log(f"Mean val_bpb: {sum(val_bpbs)/len(val_bpbs):.6f}")

    # Step type breakdown
    type_counts = {}
    for entry in loop.step_log:
        t = entry["step"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    log(f"Step breakdown: {type_counts}")

    # Save results
    output = {
        "run_id": RUN_ID,
        "version": "v2.0-steps",
        "n_iterations": args.n_iterations,
        "schema": {n: list(l) for n, l in SCHEMA.factors.items()},
        "results": results,
        "step_log": loop.step_log,
        "analysis_log": loop.router.analysis_log,
        "pending_schema_changes": loop.router.pending_schema_changes,
        "wall_time_s": elapsed,
    }

    out_path = ARTIFACTS_DIR / f"{RUN_ID}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
