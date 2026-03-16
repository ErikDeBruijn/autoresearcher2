#!/usr/bin/env python3
"""Migrate existing v4 database to v4.4 multi-project schema.

Creates a "NanoGPT" project and assigns all existing data to it.
Idempotent — safe to run multiple times.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store


def migrate(db_path: str):
    store = Store(db_path)
    store.init()  # This runs _migrate() which adds the new columns

    # Check if NanoGPT project already exists
    projects = store.list_projects()
    nanogpt = next((p for p in projects if p["name"] == "NanoGPT"), None)

    if nanogpt:
        pid = nanogpt["id"]
        print(f"NanoGPT project already exists: {pid}")
    else:
        pid = store.create_project(
            name="NanoGPT",
            description="NanoGPT training hyperparameter optimization on ClimbMix-400B",
            domain_config={
                "name": "NanoGPT training",
                "description": "We run NanoGPT training experiments.",
                "intervention_types": "config_change (modify training hyperparameters), probe (short training run to test hypothesis cheaply), or code_change (rewrite train.py with structural changes)",
                "parameters": "DEPTH, MATRIX_LR, WEIGHT_DECAY, num_steps, batch_size",
                "diversity_hint": "Mix of config_change (full run), probe (short run), and code_change (when the hypothesis requires structural code changes)",
            },
        )
        print(f"Created NanoGPT project: {pid}")

    # Assign unassigned queue items to NanoGPT
    cursor = store.conn.execute(
        "UPDATE queue SET project_id = ? WHERE project_id IS NULL", (pid,)
    )
    store.conn.commit()
    print(f"Assigned {cursor.rowcount} queue items to NanoGPT")

    # Assign unassigned observations
    cursor = store.conn.execute(
        "UPDATE observations SET project_id = ? WHERE project_id IS NULL", (pid,)
    )
    store.conn.commit()
    print(f"Assigned {cursor.rowcount} observations to NanoGPT")

    # Assign unassigned world model versions
    cursor = store.conn.execute(
        "UPDATE world_model SET project_id = ? WHERE project_id IS NULL", (pid,)
    )
    store.conn.commit()
    print(f"Assigned {cursor.rowcount} world model versions to NanoGPT")

    # Summary
    for stage in ("backlog", "todo", "running", "done", "reviewed"):
        count = store.count_proposals(stage, project_id=pid)
        print(f"  {stage}: {count}")

    wm = store.load_world_model(project_id=pid)
    print(f"  World model: v{wm.version}, {len(wm.beliefs)} beliefs")

    store.close()
    print("Migration complete.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "research_v4.db"
    migrate(db)
