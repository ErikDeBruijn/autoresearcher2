#!/usr/bin/env python3
"""Run the v3 planner loop (Orient + Decide phases of OODA).

Usage:
    python scripts/run_v3_planner.py /path/to/store.db [--interval 60] [--max-ticks 10]

The planner:
1. Orients on new observations (updates world model)
2. Generates proposals when the queue is low
3. Critiques and promotes the best proposals to todo
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
logger = logging.getLogger("v3.planner")


def make_llm_call_fn(ssh_host: str | None = None):
    """Create an LLM call function, optionally over SSH."""
    def llm_call(prompt: str) -> dict:
        return call_llm_json(prompt, ssh_host=ssh_host)
    return llm_call


def main():
    parser = argparse.ArgumentParser(description="v3 Planner loop")
    parser.add_argument("db_path", type=Path, help="Path to SQLite database file")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between ticks")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after N ticks")
    parser.add_argument("--min-queue", type=int, default=5, help="Min backlog+todo before generating")
    parser.add_argument("--n-proposals", type=int, default=5, help="Proposals per generation")
    parser.add_argument("--n-select", type=int, default=2, help="Proposals promoted per tick")
    parser.add_argument("--ssh-host", type=str, default=None, help="SSH host for remote LLM")
    parser.add_argument("--init", action="store_true", help="Initialize store if needed")
    args = parser.parse_args()

    store = Store(args.db_path)
    if args.init:
        store.init()
        logger.info("Store initialized at %s", args.db_path)

    llm_fn = make_llm_call_fn(args.ssh_host)
    planner = Planner(
        store,
        llm_call_fn=llm_fn,
        min_queue_size=args.min_queue,
        n_proposals=args.n_proposals,
        n_select=args.n_select,
    )

    logger.info("Starting planner: interval=%ss, min_queue=%d, n_proposals=%d, n_select=%d",
                args.interval, args.min_queue, args.n_proposals, args.n_select)
    planner.run(poll_interval=args.interval, max_ticks=args.max_ticks)


if __name__ == "__main__":
    main()
