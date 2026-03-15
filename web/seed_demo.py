#!/usr/bin/env python3
"""Seed demo data into research_v4.db for UI development."""
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation

DB_PATH = Path(__file__).parent.parent / "research_v4.db"


def seed():
    store = Store(DB_PATH)
    store.init()

    # Skip if already seeded
    if store.count_proposals("done") > 0:
        print("Already seeded, skipping")
        store.close()
        return

    base_time = time.time() - 3600 * 6  # 6 hours ago

    # --- Seed world model with beliefs ---
    wm = store.load_world_model()
    wm.add_belief("MATRIX_LR=0.04 produces best val_bpb at DEPTH=8", 0.75, ["obs_001", "obs_003"])
    wm.add_belief("DEPTH > 8 has diminishing returns for val_bpb", 0.45, ["obs_005"])
    wm.add_belief("WEIGHT_DECAY has minimal effect on outcome", 0.30, ["obs_002"], ["obs_006"])
    wm.add_belief("Batch size 64 is optimal for current GPU memory", 0.60, ["obs_004"])
    wm.add_tension(belief_ids=["B1", "B2"], nature="DEPTH=10 with optimal LR untested — could challenge B2")
    wm.cost_beliefs = {
        "config_change": {"wall_time_s": 300, "compute_cost": 0.5},
        "probe": {"wall_time_s": 60, "compute_cost": 0.1},
    }
    store.save_world_model(wm, delta={"seeded": True, "learntropy": 0.0}, reasoning="Initial beliefs from v1.5 evidence run")

    # --- Seed observations ---
    configs = [
        {"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
        {"DEPTH": "8", "MATRIX_LR": "0.02", "WEIGHT_DECAY": "0.1"},
        {"DEPTH": "10", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
        {"DEPTH": "6", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.2"},
        {"DEPTH": "8", "MATRIX_LR": "0.08", "WEIGHT_DECAY": "0.1"},
    ]
    val_bpbs = [1.033, 1.049, 1.041, 1.058, None]
    successes = [True, True, True, True, False]
    errors = [None, None, None, None, "OOM: CUDA out of memory"]
    workers = ["gpu-0", "gpu-1", "gpu-0", "gpu-1", "gpu-0"]

    for i, (cfg, vbpb, success, err, wid) in enumerate(zip(configs, val_bpbs, successes, errors, workers)):
        obs = Observation(
            intervention_type="config_change",
            intervention_spec=cfg,
            outcome_metrics={"val_bpb": vbpb} if vbpb else None,
            outcome_success=success,
            error=err,
            wall_time_s=random.uniform(200, 400),
            compute_cost=random.uniform(0.3, 0.7),
            worker_id=wid,
        )
        obs.created_at = base_time + i * 1200
        store.save_observation(obs)

    # Update world model with learntropy after observations
    wm2 = store.load_world_model()
    wm2.beliefs[0]["confidence"] = 0.85
    store.save_world_model(
        wm2,
        trigger_obs_id=obs.id,
        delta={"beliefs_revised": [{"id": "B1", "new_confidence": 0.85}], "learntropy": 0.42},
        reasoning="Observation confirmed LR=0.04 at DEPTH=8 is best so far",
    )

    # --- Seed proposals in various stages ---

    # Backlog (3 proposals)
    backlog_proposals = [
        ("Test LR=0.06 at DEPTH=8", "Mid-range between 0.04 and 0.08 is unexplored", "Narrow optimal LR range"),
        ("Probe WEIGHT_DECAY=0.3", "Current evidence is contradictory for WD", "Resolve tension T1"),
        ("Test DEPTH=12 with LR=0.04", "If B2 is wrong, deeper models could be better", "Challenge diminishing returns belief"),
    ]
    for intent, rationale, learning in backlog_proposals:
        p = Proposal(
            intent=intent, rationale=rationale, expected_learning=learning,
            intervention_type="config_change",
            intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.06"},
        )
        p.created_at = base_time + 5000
        store.save_proposal(p)

    # Todo (2 ranked proposals)
    for i, (intent, rationale) in enumerate([
        ("Replicate best result LR=0.04 DEPTH=8", "Validate reliability of best config"),
        ("Test batch_size=128 with gradient accumulation", "May improve throughput without OOM"),
    ]):
        p = Proposal(
            intent=intent, rationale=rationale, expected_learning="Validate reproducibility",
            intervention_type="config_change",
            intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04"},
        )
        p.set_critic_decision("accept", rank=i + 1, rationale="High value, low risk")
        p.promote("todo")
        p.created_at = base_time + 4000
        store.save_proposal(p)

    # Running (1 proposal)
    p = Proposal(
        intent="Test LR schedule: cosine annealing", rationale="Fixed LR may leave performance on the table",
        expected_learning="Whether LR scheduling improves convergence",
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04", "LR_SCHEDULE": "cosine"},
    )
    p.set_critic_decision("accept", rank=1, rationale="Low cost, high potential")
    p.promote("running")
    p.created_at = base_time + 3000
    store.save_proposal(p)

    # Done (3 completed proposals)
    for i, (intent, obs_id) in enumerate([
        ("Initial exploration: DEPTH=8 LR=0.04", "done"),
        ("Test DEPTH=10 LR=0.04", "done"),
        ("Probe DEPTH=6 baseline", "done"),
    ]):
        p = Proposal(
            intent=intent, rationale="Systematic exploration",
            expected_learning="Map landscape",
            intervention_type="config_change",
            intervention_spec=configs[i],
        )
        p.set_critic_decision("accept", rank=i + 1, rationale="Needed for baseline")
        p.promote("done")
        p.created_at = base_time + i * 1000
        store.save_proposal(p)

    # Reviewed (2 proposals)
    for intent in ["Test LR=0.02 (conservative)", "Test LR=0.08 (aggressive)"]:
        p = Proposal(
            intent=intent, rationale="Bracket the optimal LR",
            expected_learning="LR sensitivity",
            intervention_type="config_change",
            intervention_spec={"DEPTH": "8"},
        )
        p.set_critic_decision("accept", rank=1, rationale="Standard exploration")
        p.promote("done")
        p.status = "reviewed"
        p.created_at = base_time
        store.save_proposal(p)

    store.close()
    print(f"Seeded demo data into {DB_PATH}")


if __name__ == "__main__":
    seed()
