#!/usr/bin/env python3
"""Run the v3 worker loop (Act phase of OODA).

Usage:
    python scripts/run_v3_worker.py /path/to/workspace [--interval 30] [--max-ticks 10]

The worker:
1. Claims the highest-ranked todo item
2. Executes the intervention
3. Records the observation
4. Moves proposal to done
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.proposal import Proposal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("v3.worker")


def make_execute_fn(ssh_host: str | None = None):
    """Create execution function. Override this for your domain."""
    def execute(proposal: Proposal) -> dict:
        itype = proposal.intervention_type
        spec = proposal.intervention_spec

        if itype == "config_change":
            logger.info("Would execute config_change: %s", spec)
            # TODO: connect to actual training environment
            return {"metrics": {"placeholder": True}, "raw_log": "dry-run"}
        elif itype == "probe":
            logger.info("Would execute probe: %s", spec)
            return {"metrics": {"placeholder": True}, "raw_log": "dry-run probe"}
        else:
            logger.warning("Unknown intervention type: %s", itype)
            return {"metrics": {"placeholder": True}, "raw_log": f"dry-run {itype}"}

    return execute


def main():
    parser = argparse.ArgumentParser(description="v3 Worker loop")
    parser.add_argument("workspace", type=Path, help="Path to workspace directory")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between polls when idle")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N ticks")
    parser.add_argument("--worker-id", type=str, default="worker_0", help="Worker identifier")
    parser.add_argument("--ssh-host", type=str, default=None, help="SSH host for remote execution")
    parser.add_argument("--dry-run", action="store_true", help="Log interventions without executing")
    args = parser.parse_args()

    ws = Workspace(args.workspace)
    execute_fn = make_execute_fn(args.ssh_host)

    worker = Worker(ws, execute_fn=execute_fn, worker_id=args.worker_id)

    logger.info("Starting worker %s: interval=%ss", args.worker_id, args.interval)
    worker.run(poll_interval=args.interval, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
