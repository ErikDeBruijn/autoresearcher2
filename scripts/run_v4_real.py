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
import subprocess
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store, QUEUE_STAGES
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.executors import make_trainpy_executor, make_shell_executor, make_dispatch_executor, make_dry_run_executor
from autoresearcher2.v3.cost_tracker import with_cost_tracking
from autoresearcher2.v3.llm_call import call_llm_json

# NanoGPT-specific metric patterns for train.py output parsing
NANOGPT_METRIC_PATTERNS = {
    "val_bpb": r"val_bpb:\s+([\d.]+)",
    "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
    "num_steps": r"num_steps:\s+(\d+)",
    "num_params_M": r"num_params_M:\s+([\d.]+)",
}

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
    parser.add_argument("--ssh-host", type=str, default="root@dllm-experiment.local")
    parser.add_argument("--ssh-key", type=str, default="~/.ssh/pve03_key")
    parser.add_argument("--cuda-device", type=str, default="1",
                        help="DEPRECATED: use --cuda-devices instead")
    parser.add_argument("--cuda-devices", type=str, default=None,
                        help="Comma-separated CUDA device IDs (e.g. '0,1')")
    parser.add_argument("--cpu-workers", type=int, default=0,
                        help="Number of CPU-only workers for CPU-bound projects (e.g. Atari)")
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
    parser.add_argument("--n-select", type=int, default=3)
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

    # All projects run on GPU — both NanoGPT and Atari/RL use GPU workers.
    # Atari Breakout uses CnnPolicy which requires GPU for reasonable speed.
    gpu_project_ids = [proj["id"] for proj in store.list_projects(active_only=False)]
    # RL project IDs (for executor dispatch)
    rl_project_ids = [
        proj["id"] for proj in store.list_projects(active_only=False)
        if "atari" in proj["name"].lower() or "cartpole" in proj["name"].lower()
        or "rl " in proj["name"].lower() or "breakout" in proj["name"].lower()
    ]

    worker_configs = []  # (worker_id, execute_fn, project_ids, post_complete_fn)

    atari_base = "/root/github.com/atari-research"
    record_script = "/root/github.com/erikdebruijn/autoresearcher2/scripts/record_video.py"

    # Track best mean_reward per project for conditional video recording
    atari_best_reward: dict[str | None, float] = {}

    def atari_post_complete(proposal, obs):
        """Record gameplay video only when a new best reward is achieved."""
        metrics = obs.outcome_metrics or {}
        reward = metrics.get("mean_reward")
        if reward is None:
            return
        pid = getattr(proposal, "project_id", None)
        prev_best = atari_best_reward.get(pid)
        if prev_best is not None and reward <= prev_best:
            logger.info("Atari reward %.1f <= best %.1f, skipping video", reward, prev_best)
            return
        atari_best_reward[pid] = reward
        logger.info("New Atari record! reward=%.1f (prev=%.1f), recording video...",
                    reward, prev_best or 0)
        try:
            result = subprocess.run(
                ["bash", "-c",
                 f"cd {atari_base} && source .venv/bin/activate && "
                 f"python {record_script} 2>&1"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("Video recorded: %s", result.stdout[-200:])
                import re as _re
                m = _re.search(r"artifact_video:\s+(.+)", result.stdout)
                if m:
                    video_path = m.group(1).strip()
                    artifact_paths = dict(obs.artifact_paths or {})
                    artifact_paths["video"] = video_path
                    video_store = Store(args.database)
                    video_store.update_observation_artifacts(obs.id, artifact_paths)
                    video_store.close()
            else:
                logger.warning("Video recording failed: %s", result.stderr[-200:])
        except Exception:
            logger.warning("Video recording error", exc_info=True)

    # --- GPU workers: one per CUDA device ---
    # Each GPU worker handles both NanoGPT and Atari Breakout projects.
    # Atari Breakout uses CnnPolicy which needs GPU for reasonable speed.
    for i, cuda_dev in enumerate(cuda_devices):
        worker_id = f"{hostname}_GPU{cuda_dev}"
        post_complete = None
        if args.dry_run:
            execute_fn = make_dry_run_executor()
        else:
            # NanoGPT executor (cost tracking via wrapper)
            nanogpt_exec = with_cost_tracking(
                make_trainpy_executor(
                    ssh_host=args.ssh_host,
                    ssh_key=args.ssh_key,
                    cuda_device=cuda_dev,
                    local=args.local_llm,
                    metric_patterns=NANOGPT_METRIC_PATTERNS,
                ),
                cuda_device=cuda_dev,
            )

            # Atari Breakout executor (GPU-accelerated)
            atari_dir = f"{atari_base}_gpu{cuda_dev}"
            subprocess.run(
                ["bash", "-c",
                 f"mkdir -p {atari_dir} && "
                 f"test -L {atari_dir}/.venv || ln -s {atari_base}/.venv {atari_dir}/.venv && "
                 f"cp {atari_base}/train_atari.py {atari_dir}/train_atari.py"],
                capture_output=True, text=True, timeout=30,
            )
            atari_exec = with_cost_tracking(
                make_shell_executor(
                    command_template=(
                        f"cd {atari_dir} && "
                        "source .venv/bin/activate && "
                        f"CUDA_VISIBLE_DEVICES={cuda_dev} python train_atari.py 2>&1"
                    ),
                    metric_patterns={
                        "mean_reward": r"mean_reward:\s+([\d.]+)",
                        "std_reward": r"std_reward:\s+([\d.]+)",
                        "fps": r"fps:\s+([\d.]+)",
                    },
                    timeout=900,
                    work_dir=atari_dir,
                    base_script=f"{atari_base}/train_atari.py",
                ),
                cuda_device=cuda_dev,
            )

            # Dispatch: RL projects → atari executor, everything else → NanoGPT
            executors = {None: nanogpt_exec}
            for pid in gpu_project_ids:
                if pid in rl_project_ids:
                    executors[pid] = atari_exec
                else:
                    executors[pid] = nanogpt_exec
            execute_fn = make_dispatch_executor(executors)
            post_complete = atari_post_complete

        worker_configs.append((worker_id, execute_fn, gpu_project_ids, post_complete))

    worker_ids = [wid for wid, _, _, _ in worker_configs]
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
        from autoresearcher2.v3.generator import domain_config_from_project
        planners: dict[str | None, Planner] = {}
        cycle = 0

        PRIORITY_SLOTS = {
            "exclusive": {"n_proposals": 5, "n_select": 3},
            "high":      {"n_proposals": 5, "n_select": 3},
            "normal":    {"n_proposals": 3, "n_select": 2},
            "low":       {"n_proposals": 2, "n_select": 1},
        }

        def get_priority_slots(proj):
            priority = proj.get("priority", "auto")
            if priority == "paused":
                return None
            if priority in PRIORITY_SLOTS:
                return PRIORITY_SLOTS[priority]
            # auto: compute expected_gain
            gain = planner_store.compute_expected_gain(proj["id"])
            if gain >= 0.5:
                return {"n_proposals": 5, "n_select": 3}
            if gain >= 0.2:
                return {"n_proposals": 3, "n_select": 2}
            return {"n_proposals": 2, "n_select": 1}

        def get_or_create_planner(project_id, domain=None, n_proposals=None, n_select=None):
            if project_id not in planners:
                planners[project_id] = Planner(
                    planner_store, llm_call_fn=llm_fn,
                    min_queue_size=args.min_queue,
                    min_todo=len(cuda_devices),
                    n_proposals=n_proposals or args.n_proposals,
                    n_select=n_select or args.n_select,
                    project_id=project_id,
                    domain=domain,
                )
            else:
                # Update slots dynamically per cycle
                p = planners[project_id]
                if n_proposals:
                    p.n_proposals = n_proposals
                if n_select:
                    p.n_select = n_select
            return planners[project_id]

        def build_domain(proj):
            return domain_config_from_project(proj)

        while not stop.is_set():
            cycle += 1
            try:
                # Reclaim proposals stuck in 'running' longer than stale_timeout
                reclaimed = planner_store.reclaim_stale_running(timeout_s=args.stale_timeout)
                if reclaimed > 0:
                    logger.info("Reclaimed %d stale running proposals", reclaimed)

                # Tick planner for each active project, respecting priority
                active_projects = planner_store.list_projects(active_only=True)

                # Exclusive mode: if any project is exclusive, only run that one
                exclusive = [p for p in active_projects if p.get("priority") == "exclusive"]
                if exclusive:
                    active_projects = exclusive[:1]

                for proj in active_projects:
                    slots = get_priority_slots(proj)
                    if slots is None:
                        continue  # paused
                    domain = build_domain(proj)
                    planner = get_or_create_planner(
                        proj["id"], domain,
                        n_proposals=slots["n_proposals"],
                        n_select=slots["n_select"],
                    )
                    summary = planner.tick()
                    if any(v > 0 for v in summary.values()):
                        logger.info("Planner [%s] (%s) cycle %d: %s",
                                   proj["name"], proj.get("priority", "auto"), cycle, summary)

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

    def run_worker(worker_id, execute_fn, project_ids=None, post_complete_fn=None):
        """Worker thread: continuously claim and execute experiments."""
        worker_store = Store(args.database)
        worker = Worker(worker_store, execute_fn=execute_fn, worker_id=worker_id,
                        project_ids=project_ids, post_complete_fn=post_complete_fn)
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

        for wid, exe_fn, proj_ids, post_fn in worker_configs:
            t = threading.Thread(target=run_worker, args=(wid, exe_fn, proj_ids, post_fn), name=wid, daemon=True)
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
    for stage in QUEUE_STAGES:
        logger.info("  %s: %d", stage, store.count_proposals(stage))

    wm = store.load_world_model()
    logger.info("World model v%d: %d beliefs", wm.version, len(wm.beliefs))
    for b in wm.beliefs:
        logger.info("  [%.2f] %s", b["confidence"], b["claim"])

    planner_store.close()
    store.close()


if __name__ == "__main__":
    main()
