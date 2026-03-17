"""End-to-end test: full OODA cycle through SQLite Store (v4.0).

Same tests as test_e2e_cycle.py but using Store instead of Workspace.
"""
import pytest
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker


class E2EMockLLM:
    """Mock LLM that handles all three roles."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt[:80])
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": "latest result informative", "confidence": 0.6,
                     "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": {"config_change": {"wall_time_s": 60}},
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {
                "proposals": [
                    {
                        "intent": "Test hypothesis",
                        "rationale": "Based on world model",
                        "expected_learning": "Effect on outcome",
                        "intervention_type": "config_change",
                        "intervention_spec": {"x": "2"},
                        "estimated_cost": {"cost_to_test": "~1 min"},
                    },
                    {
                        "intent": "Quick probe",
                        "rationale": "Low cost exploration",
                        "expected_learning": "Rough estimate",
                        "intervention_type": "probe",
                        "intervention_spec": {"x": "3"},
                        "estimated_cost": {"cost_to_test": "~10s"},
                    },
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}


def test_full_ooda_cycle_sqlite(store):
    """Complete OODA cycle through SQLite backend."""
    mock_llm = E2EMockLLM()

    def mock_execute(proposal):
        return {"metrics": {"outcome": 0.85}, "compute_cost": 0.10}

    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=1)
    worker = Worker(store, execute_fn=mock_execute)

    # Generate + critique
    summary1 = planner.tick()
    assert summary1["generated"] == 2

    # Execute
    result = worker.tick()
    assert result is not None
    assert result["outcome_success"] is True
    assert store.count_proposals("done") == 1

    # Orient on result
    summary2 = planner.tick()
    assert summary2["oriented"] == 1

    # World model should have delta traceability
    history = store.get_world_model_history()
    assert len(history) >= 2  # seed + at least 1 update
    latest = history[-1]
    assert latest["trigger_obs_id"] is not None
    assert latest["delta"] != {}


def test_multi_cycle_sqlite(store):
    """Multiple cycles accumulate observations and world model versions."""
    mock_llm = E2EMockLLM()
    exec_count = [0]

    def mock_execute(proposal):
        exec_count[0] += 1
        return {"metrics": {"outcome": 0.85 + exec_count[0] * 0.01}}

    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=2)
    worker = Worker(store, execute_fn=mock_execute)

    for _ in range(3):
        planner.tick()
        while worker.tick() is not None:
            pass

    assert len(store.list_observations()) >= 3
    wm = store.load_world_model()
    assert wm.version > 1
    assert len(wm.beliefs) > 0


def test_world_model_delta_audit_trail(store):
    """Every orientation step creates a traceable world model version."""
    mock_llm = E2EMockLLM()
    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=2)
    worker = Worker(store, execute_fn=lambda p: {"metrics": {"x": 1}})

    # Generate, execute, orient
    planner.tick()
    worker.tick()
    worker.tick()
    planner.tick()

    history = store.get_world_model_history()
    # Should have: seed (v1) + orientation updates
    assert len(history) >= 2
    # Non-seed versions should have trigger_obs_id
    for entry in history[1:]:
        if entry["trigger_obs_id"] is not None:
            assert entry["delta"] != {}
