"""v2.1 Domain Transfer: test whether optimal configs transfer across datasets.

Runs the full autoresearcher2 approach (Bayesian + LLM) on multiple datasets
using the same 27-cell schema (DEPTH × MATRIX_LR × WEIGHT_DECAY).

Evidence test:
  - If top-3 configs are the same across datasets → crystallized (configs are general)
  - If top-3 differ → agent needs to re-explore per domain (dataset-specific)
  - Either result is informative

Usage:
    uv run python scripts/run_v21_transfer.py
    uv run python scripts/run_v21_transfer.py --dataset wikipedia
    uv run python scripts/run_v21_transfer.py --datasets climbmix wikipedia code

Expected runtime: ~6 min/experiment × 20 × 3 datasets = ~6 hours
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
from autoresearcher2.research.trainpy_environment import TrainPyEnvironment

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

ALL_DATASETS = ["climbmix", "wikipedia", "code"]
N_EXPERIMENTS = 20
GPU_DEVICE = "1"
SEED = 42
RUN_ID = uuid.uuid4().hex[:8]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_dataset(dataset: str, env: TrainPyEnvironment) -> list[dict]:
    """Run the full approach on one dataset."""
    log(f"--- Dataset: {dataset.upper()} ---")
    model = BayesianLinearModel(SCHEMA, noise_variance=0.005, prior_variance=1.0)
    controller = Controller(SCHEMA, model, preferred_outcome=1.0, seed=SEED)
    memory = MemoryStore()
    results = []

    for i in range(N_EXPERIMENTS):
        cell = controller.select_next_lookahead()
        config = SCHEMA.cell_to_config(cell)
        scores = controller.score_cell(cell)
        log(f"  [{i+1}/{N_EXPERIMENTS}] cell {cell} {config} "
            f"(prag={scores['pragmatic']:.3f} epist={scores['epistemic']:.3f})")

        start = time.time()
        try:
            outcome = env.run(cell)
            val_bpb = 2.0 - outcome
            wall_time = time.time() - start
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
                "val_bpb": val_bpb,
                "wall_time_s": round(wall_time, 1),
                "tokens_M": metadata.get("total_tokens_M"),
                "scores": scores,
                "appraisal": {k: float(v) for k, v in appraisal.items()},
                "error": None,
            }
            log(f"    → val_bpb={val_bpb:.6f} surprise={appraisal['surprise']:.4f}")
        except Exception as e:
            wall_time = time.time() - start
            result = {
                "experiment": i,
                "cell": cell,
                "config": config,
                "outcome": None,
                "val_bpb": None,
                "wall_time_s": round(wall_time, 1),
                "scores": scores,
                "error": str(e),
            }
            log(f"    → FAILED: {e}")

        results.append(result)

    return results


def analyze_transfer(all_results: dict[str, list[dict]]):
    """Compare cell rankings across datasets."""
    log("\n" + "=" * 60)
    log("TRANSFER ANALYSIS")
    log("=" * 60)

    # Get cell rankings per dataset (by mean outcome)
    rankings = {}
    for dataset, results in all_results.items():
        successful = [r for r in results if r["error"] is None]
        if not successful:
            continue

        # Compute mean outcome per cell
        cell_outcomes: dict[int, list[float]] = {}
        for r in successful:
            cell_outcomes.setdefault(r["cell"], []).append(r["outcome"])
        cell_means = {c: np.mean(o) for c, o in cell_outcomes.items()}

        # Rank cells by mean outcome (descending)
        ranked = sorted(cell_means.items(), key=lambda x: x[1], reverse=True)
        rankings[dataset] = ranked

        best = min(r["val_bpb"] for r in successful)
        log(f"\n  {dataset}: best_bpb={best:.6f}, {len(successful)}/{len(results)} ok")
        log(f"    Top-3 configs:")
        for cell, mean_out in ranked[:3]:
            config = SCHEMA.cell_to_config(cell)
            log(f"      cell {cell}: {config} (mean_outcome={mean_out:.4f})")

    # Compute pairwise rank correlation
    if len(rankings) >= 2:
        datasets = list(rankings.keys())
        log("\n  Rank Correlation (Spearman):")
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                d1, d2 = datasets[i], datasets[j]
                cells_1 = {c for c, _ in rankings[d1]}
                cells_2 = {c for c, _ in rankings[d2]}
                common = cells_1 & cells_2
                if len(common) < 3:
                    log(f"    {d1} vs {d2}: too few common cells ({len(common)})")
                    continue

                rank_1 = {c: rank for rank, (c, _) in enumerate(rankings[d1]) if c in common}
                rank_2 = {c: rank for rank, (c, _) in enumerate(rankings[d2]) if c in common}
                cells = sorted(common)
                r1 = [rank_1[c] for c in cells]
                r2 = [rank_2[c] for c in cells]

                # Spearman rank correlation
                n = len(cells)
                d_sq = sum((a - b) ** 2 for a, b in zip(r1, r2))
                rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0
                log(f"    {d1} vs {d2}: ρ = {rho:.3f} (n={n} common cells)")

        # Check top-3 overlap
        log("\n  Top-3 Overlap:")
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                d1, d2 = datasets[i], datasets[j]
                top3_1 = {c for c, _ in rankings[d1][:3]}
                top3_2 = {c for c, _ in rankings[d2][:3]}
                overlap = top3_1 & top3_2
                log(f"    {d1} vs {d2}: {len(overlap)}/3 overlap "
                    f"({'crystallized' if len(overlap) >= 2 else 'dataset-specific'})")


def main():
    parser = argparse.ArgumentParser(description="v2.1 Domain Transfer experiment")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        choices=ALL_DATASETS)
    parser.add_argument("--dataset", type=str, choices=ALL_DATASETS,
                        help="Run single dataset (shorthand)")
    parser.add_argument("--n-experiments", type=int, default=N_EXPERIMENTS)
    args = parser.parse_args()

    global N_EXPERIMENTS
    N_EXPERIMENTS = args.n_experiments

    datasets = [args.dataset] if args.dataset else args.datasets

    start = time.time()
    log("=" * 60)
    log(f"v2.1 DOMAIN TRANSFER — run_id={RUN_ID}")
    log("=" * 60)
    log(f"Datasets: {datasets}")
    log(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    log(f"Budget: {N_EXPERIMENTS} experiments per dataset")

    all_results = {}

    for dataset in datasets:
        dataset_start = time.time()
        env = TrainPyEnvironment(
            schema=SCHEMA,
            cuda_device=GPU_DEVICE,
            dataset=dataset,
        )
        results = run_dataset(dataset, env)
        dataset_time = time.time() - dataset_start

        all_results[dataset] = results

        # Save incrementally
        _save_results(all_results)
        log(f"\n  {dataset} done in {dataset_time/60:.1f} min")

    analyze_transfer(all_results)

    total_time = time.time() - start
    log(f"\nALL DONE — {total_time/60:.1f} min total")
    log(f"Results: artifacts/v21_transfer/{RUN_ID}.json")


def _save_results(all_results: dict):
    out_dir = Path("artifacts/v21_transfer")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "run_id": RUN_ID,
        "version": "v2.1",
        "schema": SCHEMA.factors,
        "n_experiments": N_EXPERIMENTS,
        "seed": SEED,
        "gpu_device": GPU_DEVICE,
        "datasets": {
            name: {
                "results": results,
                "n_successful": sum(1 for r in results if r["error"] is None),
            }
            for name, results in all_results.items()
        },
    }
    with open(out_dir / f"{RUN_ID}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
