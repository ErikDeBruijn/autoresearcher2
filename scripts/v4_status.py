#!/usr/bin/env python3
"""Inspect v4 research database status.

Usage:
    python scripts/v4_status.py /path/to/research.db [--verbose]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store, QUEUE_STAGES


def main():
    parser = argparse.ArgumentParser(description="v4 database status")
    parser.add_argument("database", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    store = Store(args.database)

    # Queue status
    print("=== Queue ===")
    for stage in QUEUE_STAGES:
        count = store.count_proposals(stage)
        print(f"  {stage}: {count}")
        if args.verbose and count > 0:
            for p in store.list_proposals(stage):
                rank = (p.critic or {}).get("rank", "-")
                print(f"    [{rank}] {p.id[:12]} {p.intervention_type}: {p.intent[:60]}")

    # Observations
    obs = store.list_observations()
    print(f"\n=== Observations: {len(obs)} ===")
    if args.verbose:
        for o in obs:
            metrics = o.outcome_metrics or {}
            print(f"  {o.id[:12]} {o.intervention_type} success={o.outcome_success} "
                  f"val_bpb={metrics.get('val_bpb', 'n/a')} {o.wall_time_s:.1f}s")

    # World model
    wm = store.load_world_model()
    print(f"\n=== World Model v{wm.version} ===")
    print(f"  Beliefs: {len(wm.beliefs)}")
    for b in wm.beliefs:
        print(f"    [{b['confidence']:.2f}] {b['claim']}")
    print(f"  Tensions: {len(wm.tensions)}")
    for t in wm.tensions:
        print(f"    [{t.get('salience', '?')}] {t.get('nature', '')[:60]}")
    if wm.cost_beliefs:
        print(f"  Cost beliefs:")
        for itype, costs in wm.cost_beliefs.items():
            print(f"    {itype}: {costs}")

    # History
    history = store.get_world_model_history()
    print(f"\n=== World Model History: {len(history)} versions ===")
    for h in history:
        trigger = h["trigger_obs_id"][:12] if h["trigger_obs_id"] else "seed"
        learntropy = h["delta"].get("learntropy", "n/a")
        print(f"  v{h['version']}: trigger={trigger} learntropy={learntropy}")

    store.close()


if __name__ == "__main__":
    main()
