"""Tests for Planner — generator + critic + orientation loop."""
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.planner import Planner
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


class PlannerMockLLM:
    """Mock LLM that handles orientation, generator, and critic prompts."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt):
        self.call_count += 1

        if "NEW OBSERVATION" in prompt and "UPDATE INSTRUCTIONS" in prompt:
            # Orientation prompt
            return {
                "beliefs_added": [
                    {"claim": "Result suggests lr matters", "confidence": 0.6,
                     "evidence_for": ["obs_new"]},
                ],
                "cost_beliefs_updated": {"config_change": {"wall_time_s": 300}},
            }
        elif "Generate" in prompt and "proposals" in prompt.lower():
            # Generator prompt
            return {
                "proposals": [
                    {
                        "intent": f"Test hypothesis {self.call_count}",
                        "rationale": "Based on current world model tensions",
                        "expected_learning": "Whether our belief holds",
                        "intervention_type": "config_change",
                        "intervention_spec": {"DEPTH": "8", "MATRIX_LR": "0.04"},
                        "estimated_cost": {"cost_to_test": "~5 min"},
                    },
                    {
                        "intent": f"Quick probe {self.call_count}",
                        "rationale": "Low cost way to test",
                        "expected_learning": "Rough estimate",
                        "intervention_type": "probe",
                        "intervention_spec": {"run_steps": 100},
                        "estimated_cost": {"cost_to_test": "~30s"},
                    },
                ]
            }
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            # Critic prompt — accept first proposal
            return {"rankings": []}  # Empty rankings = fallback to first n
        else:
            return {}


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "research")
    workspace.init()
    return workspace


def test_planner_generates_when_queue_empty(ws):
    mock = PlannerMockLLM()
    planner = Planner(ws, llm_call_fn=mock, min_queue_size=3, n_proposals=2, n_select=1)

    summary = planner.tick()

    # Should have generated proposals (queue was empty < min_queue_size)
    assert summary["generated"] == 2
    # And promoted some to todo (critic ran on backlog)
    assert ws.count_proposals("backlog") + ws.count_proposals("todo") == 2


def test_planner_orients_on_new_results(ws):
    mock = PlannerMockLLM()
    planner = Planner(ws, llm_call_fn=mock, min_queue_size=10)

    # Create a done proposal with observation
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5},
        outcome_success=True,
        wall_time_s=60.0,
    )
    ws.save_observation(obs)

    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
        status="done",
    )
    p.observation_id = obs.id
    ws.save_proposal(p)

    summary = planner.tick()

    assert summary["oriented"] == 1
    # World model should have been updated
    wm = ws.load_world_model()
    assert wm.version > 0


def test_planner_doesnt_regenerate_when_queue_full(ws):
    mock = PlannerMockLLM()
    planner = Planner(ws, llm_call_fn=mock, min_queue_size=2, n_proposals=5)

    # Pre-fill backlog
    for i in range(3):
        p = Proposal(
            intent=f"existing {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        ws.save_proposal(p)

    summary = planner.tick()

    # Should NOT generate (3 >= min_queue_size of 2)
    assert summary["generated"] == 0
    # But should still critique existing backlog
    assert ws.count_proposals("backlog") + ws.count_proposals("todo") == 3


def test_planner_doesnt_reprocess_observations(ws):
    mock = PlannerMockLLM()
    planner = Planner(ws, llm_call_fn=mock, min_queue_size=10)

    obs = Observation(
        intervention_type="config_change", intervention_spec={"x": "1"},
        outcome_success=True, wall_time_s=60.0,
    )
    ws.save_observation(obs)
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
        status="done",
    )
    p.observation_id = obs.id
    ws.save_proposal(p)

    # First tick processes the observation
    planner.tick()
    calls_after_first = mock.call_count

    # Second tick should not reprocess
    planner.tick()
    # The call count should only increase by generator+critic calls, not orientation
    # (orientation would add 1 more call for the same observation)
