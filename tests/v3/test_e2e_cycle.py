"""End-to-end test: full OODA cycle through workspace.

Tests v3.2 criterion: observation → orientation → generator → critic →
worker → observation completes without manual intervention.
"""
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.worker import Worker


class E2EMockLLM:
    """Mock LLM that handles all three roles in the OODA cycle."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt[:80])

        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            return {
                "beliefs_added": [
                    {"claim": "config x=1 yields good results", "confidence": 0.7,
                     "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": {"config_change": {"wall_time_s": 60}},
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            return {
                "proposals": [
                    {
                        "intent": "Test hypothesis about x=2",
                        "rationale": "x=1 worked, try x=2",
                        "expected_learning": "Effect of x on outcome",
                        "intervention_type": "config_change",
                        "intervention_spec": {"x": "2"},
                        "estimated_cost": {"cost_to_test": "~1 min"},
                    },
                    {
                        "intent": "Quick probe x=3",
                        "rationale": "Explore wider range",
                        "expected_learning": "Boundary of x effect",
                        "intervention_type": "probe",
                        "intervention_spec": {"x": "3", "run_steps": 50},
                        "estimated_cost": {"cost_to_test": "~10s"},
                    },
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}  # Fallback: accept first n
        else:
            return {}


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "research")
    workspace.init()
    return workspace


def test_full_ooda_cycle(ws):
    """Complete cycle: generate → critique → execute → orient."""
    mock_llm = E2EMockLLM()

    # Mock execution function
    def mock_execute(proposal):
        return {"metrics": {"outcome": 0.85}, "compute_cost": 0.10}

    planner = Planner(ws, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=1)
    worker = Worker(ws, execute_fn=mock_execute)

    # Step 1: Planner generates proposals (queue is empty)
    summary1 = planner.tick()
    assert summary1["generated"] == 2
    assert ws.count_proposals("backlog") + ws.count_proposals("todo") == 2

    # Step 2: Worker executes a todo item
    worker_result = worker.tick()
    assert worker_result is not None
    assert worker_result["outcome_success"] is True
    assert ws.count_proposals("done") == 1

    # Step 3: Planner orients on the result, then generates more
    summary2 = planner.tick()
    assert summary2["oriented"] == 1  # Processed the new observation
    # World model should be updated
    wm = ws.load_world_model()
    assert wm.version > 0
    assert len(wm.beliefs) > 0


def test_multi_tick_convergence(ws):
    """Multiple planner+worker ticks produce a growing set of observations."""
    mock_llm = E2EMockLLM()
    execution_count = [0]

    def mock_execute(proposal):
        execution_count[0] += 1
        return {"metrics": {"outcome": 0.85 + execution_count[0] * 0.01}}

    planner = Planner(ws, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=2)
    worker = Worker(ws, execute_fn=mock_execute)

    # Run 3 cycles
    for _ in range(3):
        planner.tick()
        # Execute all available todos
        while worker.tick() is not None:
            pass

    observations = ws.list_observations()
    assert len(observations) >= 3
    # All observations should be successful
    assert all(o.outcome_success for o in observations)


def test_worker_idles_when_no_work(ws):
    """Worker returns None when planner hasn't produced work yet."""
    worker = Worker(ws, execute_fn=lambda p: {"metrics": {}})
    assert worker.tick() is None


def test_planner_worker_independence(ws):
    """Planner and worker operate on different stages without conflict."""
    mock_llm = E2EMockLLM()
    planner = Planner(ws, llm_call_fn=mock_llm, min_queue_size=3, n_proposals=2, n_select=1)
    worker = Worker(ws, execute_fn=lambda p: {"metrics": {"x": 1}})

    # Planner fills queue
    planner.tick()

    # Verify stages are populated correctly
    backlog = ws.count_proposals("backlog")
    todo = ws.count_proposals("todo")
    assert backlog + todo == 2

    # Worker claims from todo
    worker.tick()
    assert ws.count_proposals("running") == 0  # Already completed
    assert ws.count_proposals("done") == 1
