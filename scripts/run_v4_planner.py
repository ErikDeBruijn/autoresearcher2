#!/usr/bin/env python3
"""Run the v4 planner loop with SQLite backend.

Usage:
    python scripts/run_v4_planner.py /path/to/research.db [--interval 60]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.llm_call import call_llm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("v4.planner")


def main():
    parser = argparse.ArgumentParser(description="v4 Planner loop (SQLite)")
    parser.add_argument("database", type=Path, help="Path to SQLite database")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--min-queue", type=int, default=5)
    parser.add_argument("--n-proposals", type=int, default=5)
    parser.add_argument("--n-select", type=int, default=2)
    parser.add_argument("--ssh-host", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()

    store = Store(args.database)
    if args.init:
        store.init()
        logger.info("Database initialized at %s", args.database)

    def llm_fn(prompt):
        return call_llm_json(prompt, ssh_host=args.ssh_host)

    planner = Planner(
        store, llm_call_fn=llm_fn,
        min_queue_size=args.min_queue,
        n_proposals=args.n_proposals,
        n_select=args.n_select,
    )

    logger.info("Starting v4 planner (SQLite): %s", args.database)
    try:
        planner.run(poll_interval=args.interval, max_ticks=args.max_ticks)
    finally:
        store.close()


if __name__ == "__main__":
    main()
