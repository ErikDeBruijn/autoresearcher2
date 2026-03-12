"""Evidence-quality comparison run on real train.py.

Runs four approaches on the same schema, same hardware, same budget,
with full instrumentation per experiment. This is the first run
whose results can be used as evidence for or against the approach.

Approaches:
  1. random       — random cell selection (control)
  2. bayesian     — Bayesian Thompson sampling + two-step lookahead, no LLM
  3. autoresearch — LLM-only with flat results log (Karpathy-style)
  4. full         — Bayesian + LLM with appraisal context (autoresearcher2)

Each approach runs N_EXPERIMENTS experiments sequentially on 1 GPU.

Usage:
    uv run python scripts/run_evidence.py [--approach all|random|bayesian|autoresearch|full]

Expected runtime: ~6 min per experiment × 20 × 4 approaches = ~8 hours
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
from autoresearcher2.llm.proposal import propose_experiments
from autoresearcher2.llm.autoresearch_baseline import AutoresearchLLMAgent
from autoresearcher2.eval.baselines import RandomAgent

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
GPU_DEVICE = "1"
LLM_EVERY = 3  # for 'full' approach: consult LLM every N experiments
SEED = 42
RUN_ID = uuid.uuid4().hex[:8]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_experiment(env: TrainPyEnvironment, cell: int, config: dict) -> dict:
    """Run one train.py experiment with full instrumentation."""
    start = time.time()
    try:
        outcome = env.run(cell)
        val_bpb = 2.0 - outcome
        wall_time = time.time() - start
        metadata = getattr(env, "last_run_metadata", {})
        return {
            "cell": cell,
            "config": config,
            "outcome": outcome,
            "val_bpb": val_bpb,
            "wall_time_s": round(wall_time, 1),
            "tokens_M": metadata.get("total_tokens_M"),
            "num_steps": metadata.get("num_steps"),
            "mfu": metadata.get("steady_state_mfu"),
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
            "tokens_M": None,
            "num_steps": None,
            "mfu": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Approach runners
# ---------------------------------------------------------------------------


def run_random(env: TrainPyEnvironment) -> list[dict]:
    """Random baseline: pick random cells."""
    log("--- Approach: RANDOM ---")
    agent = RandomAgent(SCHEMA, seed=SEED)
    results = []
    for i in range(N_EXPERIMENTS):
        cell = agent.select_next()
        config = SCHEMA.cell_to_config(cell)
        log(f"  [{i+1}/{N_EXPERIMENTS}] cell {cell} {config}")
        result = run_experiment(env, cell, config)
        result["decision_id"] = f"random-{i}"
        result["trial_id"] = i
        result["source"] = "random"
        result["experiment"] = i
        if result["error"] is None:
            agent.observe(cell, result["outcome"])
            log(f"    → val_bpb={result['val_bpb']:.6f} tokens={result['tokens_M']}M")
        else:
            log(f"    → FAILED: {result['error']}")
        results.append(result)
    return results


def run_bayesian(env: TrainPyEnvironment) -> list[dict]:
    """Bayesian Thompson sampling + lookahead, no LLM."""
    log("--- Approach: BAYESIAN (no LLM) ---")
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

        result = run_experiment(env, cell, config)
        result["decision_id"] = f"bayesian-{i}"
        result["trial_id"] = i
        result["source"] = "lookahead"
        result["experiment"] = i
        result["scores"] = scores

        if result["error"] is None:
            snapshot_before = model.snapshot()
            model.update(cell, result["outcome"])
            snapshot_after = model.snapshot()
            appraisal = compute_appraisal(SCHEMA, cell, result["outcome"],
                                          snapshot_before, snapshot_after)
            memory.add(cell, result["outcome"], appraisal)
            result["appraisal"] = {k: float(v) for k, v in appraisal.items()}
            log(f"    → val_bpb={result['val_bpb']:.6f} tokens={result['tokens_M']}M "
                f"surprise={appraisal['surprise']:.4f}")
        else:
            log(f"    → FAILED: {result['error']}")

        results.append(result)

    return results


def run_autoresearch(env: TrainPyEnvironment) -> list[dict]:
    """Autoresearch-style: LLM with flat results log, no Bayesian model."""
    log("--- Approach: AUTORESEARCH BASELINE (flat LLM) ---")
    agent = AutoresearchLLMAgent(SCHEMA, seed=SEED)
    results = []

    for i in range(N_EXPERIMENTS):
        cell = agent.select_next()
        config = SCHEMA.cell_to_config(cell)
        is_llm = len(agent.history) > 0  # first pick is random
        source = "llm_flat" if is_llm else "random_init"
        log(f"  [{i+1}/{N_EXPERIMENTS}] cell {cell} {config} [{source}]")

        result = run_experiment(env, cell, config)
        result["decision_id"] = f"autoresearch-{i}"
        result["trial_id"] = i
        result["source"] = source
        result["experiment"] = i

        if result["error"] is None:
            agent.observe(cell, result["outcome"])
            log(f"    → val_bpb={result['val_bpb']:.6f} tokens={result['tokens_M']}M")
        else:
            log(f"    → FAILED: {result['error']}")

        results.append(result)

    return results


def run_full(env: TrainPyEnvironment) -> list[dict]:
    """Full autoresearcher2: Bayesian + LLM with appraisal context."""
    log("--- Approach: FULL (Bayesian + LLM) ---")
    model = BayesianLinearModel(SCHEMA, noise_variance=0.005, prior_variance=1.0)
    controller = Controller(SCHEMA, model, preferred_outcome=1.0, seed=SEED)
    memory = MemoryStore()
    results = []
    llm_suggestions = []

    for i in range(N_EXPERIMENTS):
        # LLM consultation
        use_llm = (i > 0 and i % LLM_EVERY == 0 and len(results) >= 3)
        if use_llm:
            log(f"  LLM consultation (after {len(results)} experiments)...")
            history = [
                {
                    "config": r["config"],
                    "val_bpb": r["val_bpb"],
                    "outcome": r["outcome"],
                    "cell_index": r["cell"],
                    "appraisal": r.get("appraisal", {}),
                }
                for r in results if r["error"] is None
            ]
            suggestions = propose_experiments(SCHEMA, history, model.factor_importances())
            if suggestions:
                for s in suggestions:
                    log(f"    LLM: cell {s['cell']} {s['config']} — {s['reasoning']}")
                llm_suggestions = suggestions
            else:
                log("    LLM returned no suggestions")

        # Select cell
        if llm_suggestions:
            s = llm_suggestions.pop(0)
            cell = s["cell"]
            source = "llm_augmented"
        else:
            cell = controller.select_next_lookahead()
            source = "lookahead"

        config = SCHEMA.cell_to_config(cell)
        scores = controller.score_cell(cell)
        log(f"  [{i+1}/{N_EXPERIMENTS}] cell {cell} {config} [{source}] "
            f"(prag={scores['pragmatic']:.3f} epist={scores['epistemic']:.3f})")

        result = run_experiment(env, cell, config)
        result["decision_id"] = f"full-{i}"
        result["trial_id"] = i
        result["source"] = source
        result["experiment"] = i
        result["scores"] = scores

        if result["error"] is None:
            snapshot_before = model.snapshot()
            model.update(cell, result["outcome"])
            snapshot_after = model.snapshot()
            appraisal = compute_appraisal(SCHEMA, cell, result["outcome"],
                                          snapshot_before, snapshot_after)
            memory.add(cell, result["outcome"], appraisal)
            result["appraisal"] = {k: float(v) for k, v in appraisal.items()}
            log(f"    → val_bpb={result['val_bpb']:.6f} tokens={result['tokens_M']}M "
                f"surprise={appraisal['surprise']:.4f}")
        else:
            log(f"    → FAILED: {result['error']}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

APPROACHES = {
    "random": run_random,
    "bayesian": run_bayesian,
    "autoresearch": run_autoresearch,
    "full": run_full,
}


def main():
    parser = argparse.ArgumentParser(description="Evidence-quality comparison run")
    parser.add_argument("--approach", default="all",
                        choices=["all"] + list(APPROACHES.keys()),
                        help="Which approach(es) to run")
    args = parser.parse_args()

    approaches_to_run = list(APPROACHES.keys()) if args.approach == "all" else [args.approach]

    start = time.time()
    log("=" * 60)
    log(f"EVIDENCE RUN — run_id={RUN_ID}")
    log("=" * 60)
    log(f"Schema: {SCHEMA.n_cells} cells, {SCHEMA.n_factors} factors")
    log(f"Budget: {N_EXPERIMENTS} experiments per approach")
    log(f"GPU: {GPU_DEVICE}, Seed: {SEED}")
    log(f"Approaches: {approaches_to_run}")
    log("")

    env = TrainPyEnvironment(schema=SCHEMA, cuda_device=GPU_DEVICE)
    all_results = {}

    for approach_name in approaches_to_run:
        approach_start = time.time()
        log(f"\n{'=' * 60}")
        runner = APPROACHES[approach_name]
        results = runner(env)
        approach_time = time.time() - approach_start

        successful = [r for r in results if r["error"] is None]
        if successful:
            best = min(r["val_bpb"] for r in successful)
            mean = sum(r["val_bpb"] for r in successful) / len(successful)
            tokens = [r["tokens_M"] for r in successful if r["tokens_M"]]
            mean_tokens = sum(tokens) / len(tokens) if tokens else None
            log(f"\n  {approach_name}: best={best:.6f} mean={mean:.6f} "
                f"tokens/exp={mean_tokens:.1f}M "
                f"({len(successful)}/{len(results)} ok, {approach_time/60:.1f} min)")

        all_results[approach_name] = {
            "results": results,
            "total_time_s": round(approach_time, 1),
        }

        # Save incrementally
        _save_results(all_results)

    total_time = time.time() - start
    log(f"\n{'=' * 60}")
    log(f"ALL DONE — {total_time/60:.1f} min total")
    log(f"{'=' * 60}")

    # Final comparison
    log("\nCOMPARISON:")
    log(f"{'Approach':<20} {'Best':>10} {'Mean':>10} {'Unique':>8} {'Tok/exp':>10}")
    log("-" * 60)
    for name, data in all_results.items():
        successful = [r for r in data["results"] if r["error"] is None]
        if successful:
            best = min(r["val_bpb"] for r in successful)
            mean = sum(r["val_bpb"] for r in successful) / len(successful)
            unique = len(set(r["cell"] for r in successful))
            tokens = [r["tokens_M"] for r in successful if r["tokens_M"]]
            mean_tok = f"{sum(tokens)/len(tokens):.1f}M" if tokens else "?"
            log(f"{name:<20} {best:>10.6f} {mean:>10.6f} {unique:>8} {mean_tok:>10}")

    _save_results(all_results)
    log(f"\nResults: artifacts/evidence/{RUN_ID}.json")


def _save_results(all_results: dict):
    artifacts_dir = Path("artifacts/evidence")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "run_id": RUN_ID,
        "schema": SCHEMA.factors,
        "n_experiments": N_EXPERIMENTS,
        "seed": SEED,
        "gpu_device": GPU_DEVICE,
        "llm_every": LLM_EVERY,
        "approaches": {
            name: {
                "results": data["results"],
                "total_time_s": data["total_time_s"],
            }
            for name, data in all_results.items()
        },
    }
    with open(artifacts_dir / f"{RUN_ID}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
