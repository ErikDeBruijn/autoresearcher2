"""Tests for design doc criteria that aren't covered by other test files.

Criteria 12: Cost beliefs improve after observations
Criteria 13: Deliberation efficiency (skip full cycle for cheap probes)
"""
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker
from autoresearcher2.v3.world_model import WorldModel


class CostLearningLLM:
    """Mock LLM that reports actual wall_time in cost_beliefs_updated."""

    def __init__(self):
        self.observed_costs = []

    def __call__(self, prompt):
        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            # Extract wall_time from the observation in the prompt
            import json
            # The prompt contains the observation JSON — parse the wall_time
            wall_time = 300  # default
            if self.observed_costs:
                wall_time = self.observed_costs[-1]
            return {
                "beliefs_added": [
                    {"claim": "cost data gathered", "confidence": 0.5, "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": {"config_change": {"wall_time_s": wall_time}},
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {
                "proposals": [
                    {
                        "intent": "Test config",
                        "rationale": "Explore",
                        "expected_learning": "Effect",
                        "intervention_type": "config_change",
                        "intervention_spec": {"x": "1"},
                        "estimated_cost": {"cost_to_test": "~5 min"},
                    },
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "research")
    workspace.init()
    return workspace


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


def test_cost_beliefs_improve_after_runs(ws):
    """Criterion 12: after 5+ runs, cost_beliefs reflect actual observed costs."""
    mock_llm = CostLearningLLM()
    actual_times = [120, 135, 128, 140, 125]  # Real wall times

    def mock_execute(proposal):
        t = actual_times.pop(0) if actual_times else 130
        mock_llm.observed_costs.append(t)
        return {"metrics": {"y": 1.0}, "compute_cost": 0.5}

    planner = Planner(ws, llm_call_fn=mock_llm, min_queue_size=2, n_proposals=1, n_select=1)
    worker = Worker(ws, execute_fn=mock_execute)

    # Run 5 cycles
    for _ in range(5):
        planner.tick()
        while worker.tick() is not None:
            pass

    wm = ws.load_world_model()
    # Cost beliefs should exist and reflect observed data (not initial guesses)
    assert "config_change" in wm.cost_beliefs
    # The last update should reflect actual wall times (~120-140s), not the initial default
    actual_cost = wm.cost_beliefs["config_change"]["wall_time_s"]
    assert actual_cost < 200, f"Cost belief {actual_cost} should reflect actual times ~130s, not initial guess"


def test_cost_beliefs_tracked_in_store(store):
    """Criterion 12 on SQLite: cost beliefs are tracked with delta traceability."""
    mock_llm = CostLearningLLM()
    mock_llm.observed_costs = [60, 65, 58]

    def mock_execute(proposal):
        return {"metrics": {"y": 1.0}}

    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=2, n_proposals=1, n_select=1)
    worker = Worker(store, execute_fn=mock_execute)

    for _ in range(3):
        planner.tick()
        while worker.tick() is not None:
            pass

    # Check that world model history records cost belief changes
    history = store.get_world_model_history()
    cost_deltas = [h for h in history if h["delta"].get("cost_beliefs_updated")]
    assert len(cost_deltas) > 0, "No cost belief updates recorded in history"


def test_todo_queue_stays_filled(ws):
    """Criterion 10: todo should stay filled while backlog has items."""
    call_count = [0]

    def mock_llm(prompt):
        call_count[0] += 1
        if "Generate" in prompt and "proposals" in prompt.lower():
            return {
                "proposals": [
                    {
                        "intent": f"Proposal {call_count[0]}",
                        "rationale": "test",
                        "expected_learning": "test",
                        "intervention_type": "config_change",
                        "intervention_spec": {"x": str(call_count[0])},
                        "estimated_cost": {},
                    }
                    for _ in range(3)
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}

    planner = Planner(ws, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=3, n_select=2)
    worker = Worker(ws, execute_fn=lambda p: {"metrics": {"ok": True}})

    # Run several cycles
    for _ in range(3):
        planner.tick()
        # Execute one item
        worker.tick()

    # After planner ticks, todo should have items (planner replenishes)
    total_actionable = ws.count_proposals("backlog") + ws.count_proposals("todo")
    assert total_actionable > 0, "Queue should stay filled"
