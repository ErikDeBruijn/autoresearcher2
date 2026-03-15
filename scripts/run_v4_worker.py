#!/usr/bin/env python3
"""Run the v4 worker loop with SQLite backend.

Usage:
    python scripts/run_v4_worker.py /path/to/research.db [--interval 30]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("v4.worker")


def make_execute_fn(ssh_host=None):
    def execute(proposal):
        logger.info("Would execute %s: %s", proposal.intervention_type, proposal.intervention_spec)
        return {"metrics": {"placeholder": True}, "raw_log": "dry-run"}
    return execute


def main():
    parser = argparse.ArgumentParser(description="v4 Worker loop (SQLite)")
    parser.add_argument("database", type=Path, help="Path to SQLite database")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--worker-id", type=str, default="worker_0")
    parser.add_argument("--ssh-host", type=str, default=None)
    args = parser.parse_args()

    store = Store(args.database)
    execute_fn = make_execute_fn(args.ssh_host)
    worker = Worker(store, execute_fn=execute_fn, worker_id=args.worker_id)

    logger.info("Starting v4 worker %s (SQLite): %s", args.worker_id, args.database)
    try:
        worker.run(poll_interval=args.interval, max_ticks=args.max_ticks)
    finally:
        store.close()


if __name__ == "__main__":
    main()
