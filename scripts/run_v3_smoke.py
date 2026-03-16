#!/usr/bin/env python3
"""Smoke test: run planner + worker for a few cycles with mock execution.

Usage:
    python scripts/run_v3_smoke.py [--db /tmp/v3_smoke.db]

This validates the full OODA loop works end-to-end without requiring
a real LLM or GPU. Uses a mock LLM and mock execution function.
"""
import argparse
import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("v3.smoke")


class SmokeTestLLM:
    """Mock LLM for smoke testing."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": f"Observation {self.call_count} is informative",
                     "confidence": 0.6, "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": {"config_change": {"wall_time_s": 300}},
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {
                "proposals": [
                    {
                        "intent": f"Test config variant {self.call_count}",
                        "rationale": "Explore parameter space",
                        "expected_learning": "Effect on outcome metric",
                        "intervention_type": "config_change",
                        "intervention_spec": {"lr": f"0.0{self.call_count}"},
                        "estimated_cost": {"cost_to_test": "~5 min"},
                    },
                    {
                        "intent": f"Quick probe {self.call_count}",
                        "rationale": "Low cost exploration",
                        "expected_learning": "Rough estimate",
                        "intervention_type": "probe",
                        "intervention_spec": {"run_steps": 100},
                        "estimated_cost": {"cost_to_test": "~30s"},
                    },
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}


def mock_execute(proposal):
    """Mock execution that returns fake metrics."""
    import random
    return {
        "metrics": {"val_bpb": round(random.uniform(0.8, 1.5), 3)},
        "compute_cost": round(random.uniform(0.1, 1.0), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="v3 smoke test")
    parser.add_argument("--db", type=Path, default="/tmp/v3_smoke.db")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    store = Store(args.db)
    store.init()

    mock_llm = SmokeTestLLM()
    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=2)
    worker = Worker(store, execute_fn=mock_execute)

    for cycle in range(args.cycles):
        logger.info("=== Cycle %d ===", cycle + 1)

        summary = planner.tick()
        logger.info("Planner: %s", summary)

        executed = 0
        while True:
            result = worker.tick()
            if result is None:
                break
            executed += 1
            logger.info("Worker executed: success=%s metrics=%s",
                       result["outcome_success"], result.get("outcome_metrics"))

        logger.info("Executed %d items this cycle", executed)

    # Final status
    logger.info("=== Final Status ===")
    logger.info("Backlog: %d", store.count_proposals("backlog"))
    logger.info("Todo: %d", store.count_proposals("todo"))
    logger.info("Running: %d", store.count_proposals("running"))
    logger.info("Done: %d", store.count_proposals("done"))
    logger.info("Observations: %d", len(store.list_observations()))

    wm = store.load_world_model()
    logger.info("World model version: %d, beliefs: %d",
               wm.version, len(wm.beliefs))
    logger.info("LLM calls: %d", mock_llm.call_count)

    # Verify basic health
    assert store.count_proposals("done") > 0, "No proposals completed!"
    assert len(store.list_observations()) > 0, "No observations recorded!"
    assert wm.version > 0, "World model never updated!"
    logger.info("Smoke test PASSED")


if __name__ == "__main__":
    main()
