#!/usr/bin/env python3
"""Run v4 generator-critic loop on real GPT training (dllm-experiment VM).

Usage:
    # Initialize and run (first time)
    python scripts/run_v4_real.py --init --ssh-host root@dllm-experiment.home

    # Resume existing research
    python scripts/run_v4_real.py --ssh-host root@dllm-experiment.home

    # Dry run (no real execution)
    python scripts/run_v4_real.py --dry-run

This runs both planner and worker in a single process for simplicity.
For multi-GPU: run separate run_v4_worker.py instances per GPU.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.executors import make_trainpy_executor, make_dry_run_executor
from autoresearcher2.v3.llm_call import call_llm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("v4.real")


def seed_world_model(store):
    """Seed initial world model with known beliefs from v1.5 evidence run."""
    wm = store.load_world_model()
    if len(wm.beliefs) > 0:
        logger.info("World model already has beliefs, skipping seed")
        return

    # Known results from v1.5 evidence run
    wm.add_belief(
        claim="MATRIX_LR=0.04 produces best val_bpb at DEPTH=8",
        confidence=0.7,
        evidence_for=["v1.5_evidence_run"],
    )
    wm.add_belief(
        claim="DEPTH > 8 has diminishing returns for val_bpb",
        confidence=0.45,
        evidence_for=["v1.5_depth_10_run"],
    )
    wm.add_belief(
        claim="WEIGHT_DECAY has minimal effect on outcome",
        confidence=0.3,
        evidence_for=["v1.5_wd_02"],
        evidence_against=["v1.5_wd_01"],
    )
    wm.cost_beliefs = {
        "config_change": {"wall_time_s": 300, "compute_cost": 0.5},
        "probe": {"wall_time_s": 60, "compute_cost": 0.1},
    }

    store.save_world_model(wm, delta={"seeded": True}, reasoning="Initial beliefs from v1.5 evidence run")
    logger.info("Seeded world model with %d beliefs from v1.5", len(wm.beliefs))


def main():
    parser = argparse.ArgumentParser(description="v4 real-world research loop")
    parser.add_argument("--database", type=Path, default="research_v4.db")
    parser.add_argument("--ssh-host", type=str, default="root@dllm-experiment.home")
    parser.add_argument("--ssh-key", type=str, default="~/.ssh/pve03_key")
    parser.add_argument("--cuda-device", type=str, default="1")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-llm", action="store_true", help="Call claude locally instead of via SSH")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--planner-interval", type=float, default=60.0)
    parser.add_argument("--min-queue", type=int, default=5)
    parser.add_argument("--n-proposals", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=2)
    args = parser.parse_args()

    store = Store(args.database)
    if args.init:
        store.init()
        seed_world_model(store)
        logger.info("Initialized database at %s", args.database)

    # LLM call function (for planner)
    def llm_fn(prompt):
        return call_llm_json(prompt, ssh_host=args.ssh_host, local=args.local_llm)

    # Executor (for worker)
    if args.dry_run:
        execute_fn = make_dry_run_executor()
    else:
        execute_fn = make_trainpy_executor(
            ssh_host=args.ssh_host,
            ssh_key=args.ssh_key,
            cuda_device=args.cuda_device,
            local=args.local_llm,
        )

    planner = Planner(
        store, llm_call_fn=llm_fn,
        min_queue_size=args.min_queue,
        n_proposals=args.n_proposals,
        n_select=args.n_select,
    )
    worker = Worker(store, execute_fn=execute_fn)

    logger.info("Starting v4 research loop (dry_run=%s, cuda=%s)", args.dry_run, args.cuda_device)

    cycle = 0
    try:
        while args.max_cycles is None or cycle < args.max_cycles:
            cycle += 1
            logger.info("=== Cycle %d ===", cycle)

            # Plan
            summary = planner.tick()
            logger.info("Planner: %s", summary)

            # Execute all available work
            executed = 0
            while True:
                result = worker.tick()
                if result is None:
                    break
                executed += 1
                metrics = result.get("outcome_metrics") or {}
                logger.info("Worker: success=%s val_bpb=%s",
                           result.get("outcome_success"),
                           metrics.get("val_bpb", "n/a"))

            logger.info("Executed %d experiments this cycle", executed)

            # Status
            wm = store.load_world_model()
            logger.info("WM v%d: %d beliefs, %d tensions",
                       wm.version, len(wm.beliefs), len(wm.tensions))

            if args.max_cycles is None or cycle < args.max_cycles:
                if executed == 0:
                    time.sleep(args.planner_interval)
                # If we executed something, immediately loop (no sleep)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Final report
        logger.info("=== Final Report ===")
        logger.info("Observations: %d", len(store.list_observations()))
        for stage in ("backlog", "todo", "running", "done", "reviewed"):
            logger.info("  %s: %d", stage, store.count_proposals(stage))

        wm = store.load_world_model()
        logger.info("World model v%d: %d beliefs", wm.version, len(wm.beliefs))
        for b in wm.beliefs:
            logger.info("  [%.2f] %s", b["confidence"], b["claim"])

        history = store.get_world_model_history()
        logger.info("World model history: %d versions", len(history))

        store.close()


if __name__ == "__main__":
    main()
