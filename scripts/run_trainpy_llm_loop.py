"""LLM-augmented autoresearcher2 loop on real train.py.

Uses both GPU 0 and GPU 1 in parallel when possible.
Every 3rd experiment: Claude analyzes results and suggests next config.
Other experiments: Thompson sampling from the Bayesian model.

This is the full system: structured Bayesian model + LLM proposal
generation + real GPU training + learntropy appraisal.

Usage:
    uv run python scripts/run_trainpy_llm_loop.py

Expected runtime: ~6 min per experiment × N_EXPERIMENTS
With 2 GPUs: pairs of experiments run in parallel where possible.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from autoresearcher2.core.schema import InterventionSchema
from autoresearcher2.generative_model.bayesian_linear import BayesianLinearModel
from autoresearcher2.core.controller import Controller
from autoresearcher2.memory.store import MemoryStore
from autoresearcher2.appraisal.signals import compute_appraisal
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment
from autoresearcher2.llm.proposal import propose_experiments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA = InterventionSchema(
    factors={
        "DEPTH": ["6", "8", "10"],
        "MATRIX_LR": ["0.02", "0.04", "0.08"],
        "WEIGHT_DECAY": ["0.1", "0.2", "0.4"],
    }
)

N_EXPERIMENTS = 20
LLM_EVERY = 3  # ask Claude for suggestions every N experiments

# GPU devices for parallel runs (adjust based on availability)
GPU_DEVICES = ["0"]  # Use only GPU 0; GPU 1 may be in use by other experiments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str):
    """Unbuffered print with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_single_experiment(
    env: TrainPyEnvironment,
    cell: int,
    config: dict,
    label: str,
) -> dict:
    """Run one train.py experiment. Returns result dict."""
    start = time.time()
    try:
        outcome = env.run(cell)
        val_bpb = 2.0 - outcome
        wall_time = time.time() - start
        return {
            "cell": cell,
            "config": config,
            "outcome": outcome,
            "val_bpb": val_bpb,
            "wall_time_s": round(wall_time, 1),
            "label": label,
            "error": None,
        }
    except Exception as e:
        wall_time = time.time() - start
        return {
            "cell": cell,
            "config": config,
            "outcome": None,
            "val_bpb": None,
            "wall_time_s": round(wall_time, 1),
            "label": label,
            "error": str(e),
        }


def update_model_and_appraise(model, schema, memory, result):
    """Update Bayesian model and compute appraisal for a successful result."""
    cell = result["cell"]
    outcome = result["outcome"]
    snapshot_before = model.snapshot()
    model.update(cell, outcome)
    snapshot_after = model.snapshot()
    appraisal = compute_appraisal(schema, cell, outcome, snapshot_before, snapshot_after)
    memory.add(cell, outcome, appraisal)
    return appraisal


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    start = time.time()
    log("=" * 60)
    log("autoresearcher2 — LLM-Augmented Real train.py Loop")
    log("=" * 60)
    log(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    log(f"Budget: {N_EXPERIMENTS} experiments")
    log(f"GPUs: {GPU_DEVICES} (parallel when possible)")
    log(f"LLM consultation: every {LLM_EVERY} experiments")
    log(f"Factors: {SCHEMA.factors}")
    log("")

    # Create two environments, one per GPU
    envs = {
        gpu: TrainPyEnvironment(schema=SCHEMA, cuda_device=gpu)
        for gpu in GPU_DEVICES
    }

    model = BayesianLinearModel(
        schema=SCHEMA,
        noise_variance=0.005,  # real training has moderate noise
        prior_variance=1.0,
    )
    controller = Controller(
        schema=SCHEMA,
        model=model,
        preferred_outcome=1.0,
        seed=42,
    )
    memory = MemoryStore()

    all_results = []
    llm_suggestions = []
    experiment_idx = 0

    while experiment_idx < N_EXPERIMENTS:
        # Decide source: LLM suggestion or Thompson sampling
        use_llm = (
            experiment_idx > 0
            and experiment_idx % LLM_EVERY == 0
            and len(all_results) >= 3
        )

        if use_llm:
            log(f"\n--- LLM consultation (after {len(all_results)} experiments) ---")
            history = [
                {
                    "config": r["config"],
                    "val_bpb": r["val_bpb"],
                    "outcome": r["outcome"],
                    "cell_index": r["cell"],
                    "appraisal": r.get("appraisal", {}),
                }
                for r in all_results
                if r["error"] is None
            ]
            suggestions = propose_experiments(
                SCHEMA, history, model.factor_importances()
            )
            if suggestions:
                for s in suggestions:
                    log(f"  LLM suggests: cell {s['cell']} {s['config']} — {s['reasoning']}")
                llm_suggestions = suggestions
            else:
                log("  LLM returned no suggestions, using Thompson sampling")

        # Determine how many experiments to run this round
        # Try to run 2 in parallel if we have 2 GPUs and enough budget
        n_this_round = min(2, N_EXPERIMENTS - experiment_idx)

        cells_to_run = []
        labels = []

        for i in range(n_this_round):
            if llm_suggestions:
                s = llm_suggestions.pop(0)
                cells_to_run.append(s["cell"])
                labels.append(f"llm_suggestion")
            else:
                cell = controller.select_next_lookahead()
                cells_to_run.append(cell)
                labels.append("lookahead")

        # Run experiments (parallel if 2 GPUs, sequential otherwise)
        round_results = []

        if len(cells_to_run) >= 2 and len(GPU_DEVICES) >= 2:
            # Parallel execution on 2 GPUs
            log(f"\n--- Experiments {experiment_idx+1}-{experiment_idx+2} (parallel) ---")
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for i, (cell, label, gpu) in enumerate(
                    zip(cells_to_run, labels, GPU_DEVICES)
                ):
                    config = SCHEMA.cell_to_config(cell)
                    exp_num = experiment_idx + i + 1
                    scores = controller.score_cell(cell)
                    log(f"  GPU {gpu}: exp {exp_num} cell {cell} {config} [{label}]")
                    log(f"    pragmatic={scores['pragmatic']:.4f} epistemic={scores['epistemic']:.4f}")
                    futures[executor.submit(
                        run_single_experiment, envs[gpu], cell, config, label
                    )] = i

                for future in as_completed(futures):
                    result = future.result()
                    round_results.append((futures[future], result))

            # Sort by original order
            round_results.sort(key=lambda x: x[0])
            round_results = [r for _, r in round_results]
        else:
            # Sequential
            for cell, label in zip(cells_to_run, labels):
                config = SCHEMA.cell_to_config(cell)
                exp_num = experiment_idx + 1
                scores = controller.score_cell(cell)
                log(f"\n--- Experiment {exp_num}/{N_EXPERIMENTS} [{label}] ---")
                log(f"  Selected: cell {cell} = {config}")
                log(f"  Scores: pragmatic={scores['pragmatic']:.4f} epistemic={scores['epistemic']:.4f}")
                result = run_single_experiment(
                    envs[GPU_DEVICES[0]], cell, config, label
                )
                round_results.append(result)

        # Process results sequentially (no race conditions on model update)
        for result in round_results:
            if result["error"] is None:
                appraisal = update_model_and_appraise(model, SCHEMA, memory, result)
                result["appraisal"] = {k: float(v) for k, v in appraisal.items()}
                log(f"  cell {result['cell']} → val_bpb={result['val_bpb']:.6f} "
                    f"surprise={appraisal['surprise']:.4f} "
                    f"learntropy={appraisal['learntropy']:.4f} "
                    f"({result['wall_time_s']}s)")
            else:
                log(f"  cell {result['cell']} → FAILED: {result['error']} ({result['wall_time_s']}s)")

            result["experiment"] = experiment_idx
            all_results.append(result)
            experiment_idx += 1

        # Show factor importances periodically
        if experiment_idx % 4 == 0 or experiment_idx == N_EXPERIMENTS:
            importances = model.factor_importances()
            log(f"  Factor importances: {importances}")

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - start
    log(f"\n{'=' * 60}")
    log("FINAL SUMMARY")
    log(f"{'=' * 60}")

    successful = [r for r in all_results if r["error"] is None]
    failed = [r for r in all_results if r["error"] is not None]

    if successful:
        best = min(successful, key=lambda r: r["val_bpb"])
        log(f"  Best val_bpb: {best['val_bpb']:.6f} at {best['config']} [{best['label']}]")
        log(f"  Worst val_bpb: {max(r['val_bpb'] for r in successful):.6f}")
        log(f"  Mean val_bpb: {sum(r['val_bpb'] for r in successful)/len(successful):.6f}")
        log("")

        # Show all runs ordered by val_bpb
        log("  All runs (sorted by val_bpb):")
        for r in sorted(successful, key=lambda r: r["val_bpb"]):
            src = "LLM" if r["label"] == "llm_suggestion" else "LA"
            log(f"    [{src}] {r['config']} → {r['val_bpb']:.6f}")

    if failed:
        log(f"\n  Failed runs: {len(failed)}")

    llm_runs = [r for r in successful if r["label"] == "llm_suggestion"]
    la_runs = [r for r in successful if r["label"] == "lookahead"]

    if llm_runs:
        llm_mean = sum(r["val_bpb"] for r in llm_runs) / len(llm_runs)
        log(f"\n  LLM suggestions mean val_bpb: {llm_mean:.6f} ({len(llm_runs)} runs)")
    if la_runs:
        la_mean = sum(r["val_bpb"] for r in la_runs) / len(la_runs)
        log(f"  Lookahead mean val_bpb: {la_mean:.6f} ({len(la_runs)} runs)")

    log(f"\n  Factor importances: {model.factor_importances()}")
    log(f"  Unique cells visited: {len(set(r['cell'] for r in all_results))}")
    log(f"  Successful: {len(successful)}/{len(all_results)}")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    artifacts_dir = Path("artifacts/trainpy_llm_loop")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "schema": SCHEMA.factors,
            "n_experiments": N_EXPERIMENTS,
            "llm_every": LLM_EVERY,
            "gpu_devices": GPU_DEVICES,
            "results": all_results,
            "factor_importances": model.factor_importances(),
            "total_time_s": round(total_time, 1),
        }, f, indent=2, default=str)
    log(f"\n  Results: {output_path}")


if __name__ == "__main__":
    main()
