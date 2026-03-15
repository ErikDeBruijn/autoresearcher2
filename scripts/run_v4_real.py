#!/usr/bin/env python3
"""Run v4 generator-critic loop on real GPT training (dllm-experiment VM).

Usage:
    # Initialize and run with 2 GPUs (first time)
    python scripts/run_v4_real.py --init --local-llm --cuda-devices 0,1

    # Resume existing research on 1 GPU
    python scripts/run_v4_real.py --local-llm --cuda-devices 1

    # Dry run (no real execution)
    python scripts/run_v4_real.py --dry-run
"""
import argparse
import logging
import socket
import sys
import time
import threading
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
    parser.add_argument("--cuda-device", type=str, default="1",
                        help="DEPRECATED: use --cuda-devices instead")
    parser.add_argument("--cuda-devices", type=str, default=None,
                        help="Comma-separated CUDA device IDs (e.g. '0,1')")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-llm", action="store_true", help="Call claude locally instead of via SSH")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--planner-interval", type=float, default=30.0)
    parser.add_argument("--worker-poll", type=float, default=5.0,
                        help="Seconds workers wait before rechecking for todo items")
    parser.add_argument("--stale-timeout", type=float, default=1800,
                        help="Seconds before a running proposal is considered stale (default: 30min)")
    parser.add_argument("--min-queue", type=int, default=5)
    parser.add_argument("--n-proposals", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=2)
    args = parser.parse_args()

    # Determine GPU devices
    if args.cuda_devices:
        cuda_devices = [d.strip() for d in args.cuda_devices.split(",")]
    else:
        cuda_devices = [args.cuda_device]

    hostname = socket.gethostname()

    store = Store(args.database)
    if args.init:
        store.init()
        seed_world_model(store)
        logger.info("Initialized database at %s", args.database)

    # On startup, ALL running proposals are orphaned (no live workers yet).
    # Use timeout_s=0 to reclaim them unconditionally.
    reclaimed = store.reclaim_stale_running(timeout_s=0)
    if reclaimed > 0:
        logger.info("Reclaimed %d orphaned running proposals back to todo", reclaimed)

    # LLM call function (for planner)
    def llm_fn(prompt):
        return call_llm_json(prompt, ssh_host=args.ssh_host, local=args.local_llm)

    # Planner gets its own Store connection for thread safety
    planner_store = Store(args.database)

    # Worker configs (each worker gets its own Store connection for thread safety)
    worker_configs = []
    for i, cuda_dev in enumerate(cuda_devices):
        worker_id = f"worker_{hostname}_{i}"
        if args.dry_run:
            execute_fn = make_dry_run_executor()
        else:
            execute_fn = make_trainpy_executor(
                ssh_host=args.ssh_host,
                ssh_key=args.ssh_key,
                cuda_device=cuda_dev,
                local=args.local_llm,
            )
        worker_configs.append((worker_id, execute_fn))

    worker_ids = [wid for wid, _ in worker_configs]
    logger.info("Starting v4 research loop (dry_run=%s, gpus=%s, workers=%s)",
                args.dry_run, cuda_devices, worker_ids)

    # Shared stop event for graceful shutdown
    stop = threading.Event()

    def run_planner():
        """Planner thread: continuously stock the queue for all active projects.

        Maintains a Planner instance per project. On each cycle, iterates
        all active projects and ticks their planner. Also ticks a default
        planner for proposals without a project (backwards compat).
        """
        from autoresearcher2.v3.generator import DomainConfig
        planners: dict[str | None, Planner] = {}
        cycle = 0

        def get_or_create_planner(project_id, domain=None):
            if project_id not in planners:
                planners[project_id] = Planner(
                    planner_store, llm_call_fn=llm_fn,
                    min_queue_size=args.min_queue,
                    n_proposals=args.n_proposals,
                    n_select=args.n_select,
                    project_id=project_id,
                    domain=domain,
                )
            return planners[project_id]

        while not stop.is_set():
            cycle += 1
            try:
                # Reclaim proposals stuck in 'running' longer than stale_timeout
                reclaimed = planner_store.reclaim_stale_running(timeout_s=args.stale_timeout)
                if reclaimed > 0:
                    logger.info("Reclaimed %d stale running proposals", reclaimed)

                # Tick planner for each active project
                active_projects = planner_store.list_projects(active_only=True)
                for proj in active_projects:
                    domain = None
                    if proj.get("domain_config"):
                        dc = proj["domain_config"]
                        domain = DomainConfig(
                            name=dc.get("name", proj["name"]),
                            description=dc.get("description", proj.get("description", "")),
                            intervention_types=dc.get("intervention_types", "config_change or probe"),
                            parameters=dc.get("parameters", ""),
                            diversity_hint=dc.get("diversity_hint", ""),
                        )
                    planner = get_or_create_planner(proj["id"], domain)
                    summary = planner.tick()
                    if any(v > 0 for v in summary.values()):
                        logger.info("Planner [%s] cycle %d: %s", proj["name"], cycle, summary)

                # Also tick default planner for unassigned proposals (backwards compat)
                default_planner = get_or_create_planner(None)
                summary = default_planner.tick()
                if any(v > 0 for v in summary.values()):
                    logger.info("Planner [default] cycle %d: %s", cycle, summary)

                # Log world model status for each project
                for proj in active_projects:
                    wm = planner_store.load_world_model(project_id=proj["id"])
                    if wm.version > 0 and len(wm.beliefs) > 0:
                        logger.info("[%s] WM v%d: %d beliefs, %d tensions",
                                   proj["name"], wm.version, len(wm.beliefs), len(wm.tensions))

                # Log default world model
                wm = planner_store.load_world_model()
                if wm.version > 0:
                    logger.info("WM v%d: %d beliefs, %d tensions",
                               wm.version, len(wm.beliefs), len(wm.tensions))

            except Exception:
                logger.error("Planner cycle %d failed", cycle, exc_info=True)

            if args.max_cycles and cycle >= args.max_cycles:
                logger.info("Planner reached max_cycles=%d", args.max_cycles)
                stop.set()
                break

            # Wait before next planning cycle, but wake up early if stopped
            stop.wait(timeout=args.planner_interval)

    def run_worker(worker_id, execute_fn):
        """Worker thread: continuously claim and execute experiments."""
        worker_store = Store(args.database)
        worker = Worker(worker_store, execute_fn=execute_fn, worker_id=worker_id)
        executed = 0
        try:
            while not stop.is_set():
                result = worker.tick()
                if result is None:
                    # No work available — wait briefly, then retry
                    stop.wait(timeout=args.worker_poll)
                    continue
                executed += 1
                metrics = result.get("outcome_metrics") or {}
                logger.info("[%s] success=%s val_bpb=%s",
                           worker_id,
                           result.get("outcome_success"),
                           metrics.get("val_bpb", "n/a"))
        finally:
            worker_store.close()
            logger.info("[%s] stopped after %d experiments", worker_id, executed)

    try:
        # Start all threads concurrently — planner and workers run in parallel
        threads = []

        planner_thread = threading.Thread(target=run_planner, name="planner", daemon=True)
        threads.append(planner_thread)
        planner_thread.start()

        for wid, exe_fn in worker_configs:
            t = threading.Thread(target=run_worker, args=(wid, exe_fn), name=wid, daemon=True)
            threads.append(t)
            t.start()

        # Wait for stop signal (from max_cycles or KeyboardInterrupt)
        while not stop.is_set():
            stop.wait(timeout=1.0)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stop.set()

    # Wait for threads to finish current work
    logger.info("Shutting down...")
    for t in threads:
        t.join(timeout=10)

    # Final report
    logger.info("=== Final Report ===")
    logger.info("Observations: %d", len(store.list_observations()))
    for stage in ("backlog", "todo", "running", "done", "reviewed"):
        logger.info("  %s: %d", stage, store.count_proposals(stage))

    wm = store.load_world_model()
    logger.info("World model v%d: %d beliefs", wm.version, len(wm.beliefs))
    for b in wm.beliefs:
        logger.info("  [%.2f] %s", b["confidence"], b["claim"])

    planner_store.close()
    store.close()


if __name__ == "__main__":
    main()
