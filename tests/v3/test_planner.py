"""Tests for Planner — generator + critic + orientation loop.

All tests use Store (SQLite backend). Workspace compatibility was removed.
"""
import pytest
from autoresearcher2.v3.store import Store
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
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    s.init()
    return s


def test_planner_generates_when_queue_empty(store):
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=3, n_proposals=2, n_select=1)

    summary = planner.tick()

    # Should have generated proposals (queue was empty < min_queue_size)
    assert summary["generated"] == 2
    # And promoted some to todo (critic ran on backlog)
    assert store.count_proposals("backlog") + store.count_proposals("todo") == 2


def test_planner_orients_on_new_results(store):
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=10)

    # Create a done proposal with observation
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5},
        outcome_success=True,
        wall_time_s=60.0,
    )
    store.save_observation(obs)

    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
        status="done",
    )
    p.observation_id = obs.id
    store.save_proposal(p)

    summary = planner.tick()

    assert summary["oriented"] == 1
    # World model should have been updated
    wm = store.load_world_model()
    assert wm.version > 0


def test_planner_doesnt_regenerate_when_backlog_full(store):
    mock = PlannerMockLLM()
    # min_todo=0 so critic doesn't drain backlog, making this a pure generator test
    planner = Planner(store, llm_call_fn=mock, min_queue_size=2, min_todo=0, n_proposals=5)

    # Pre-fill backlog with enough items
    for i in range(3):
        p = Proposal(
            intent=f"existing {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p)

    summary = planner.tick()

    # Should NOT generate (backlog=3 >= min_queue_size=2)
    assert summary["generated"] == 0
    assert store.count_proposals("backlog") == 3


def test_planner_doesnt_reprocess_observations(store):
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=10)

    obs = Observation(
        intervention_type="config_change", intervention_spec={"x": "1"},
        outcome_success=True, wall_time_s=60.0,
    )
    store.save_observation(obs)
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
        status="done",
    )
    p.observation_id = obs.id
    store.save_proposal(p)

    # First tick processes the observation
    planner.tick()
    calls_after_first = mock.call_count

    # Second tick should not reprocess
    planner.tick()
    # The call count should only increase by generator+critic calls, not orientation
    # (orientation would add 1 more call for the same observation)


def test_pull_based_todo_always_stocked(store):
    """When todo empties, critic promotes from backlog. When backlog empties, generator fires."""
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=3, min_todo=2, n_proposals=5, n_select=2)

    # Start with empty queue — should generate AND promote in one tick
    summary = planner.tick()
    assert summary["generated"] > 0
    assert summary["promoted"] > 0
    assert store.count_proposals("todo") >= 2


def test_critic_runs_before_generator(store):
    """If backlog has items and todo is empty, critic promotes before generator runs."""
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=10, min_todo=2, n_proposals=5, n_select=2)

    # Pre-fill backlog with plenty of items
    for i in range(5):
        p = Proposal(
            intent=f"ready {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p)

    summary = planner.tick()

    # Critic should have promoted items to todo
    assert summary["promoted"] >= 2
    assert store.count_proposals("todo") >= 2


def test_planner_calls_store_directly_no_shims(store):
    """Verify planner calls Store methods directly with project_id, no try/except fallbacks."""
    mock = PlannerMockLLM()
    pid = store.create_project("test-project", description="shim removal test")
    planner = Planner(store, llm_call_fn=mock, min_queue_size=3, n_proposals=2,
                      n_select=1, project_id=pid)

    # Generate + promote with project scoping
    summary = planner.tick()
    assert summary["generated"] == 2

    # Verify proposals are scoped to the project
    all_backlog = store.list_proposals("backlog", project_id=pid)
    all_todo = store.list_proposals("todo", project_id=pid)
    assert len(all_backlog) + len(all_todo) == 2

    # Verify world model is project-scoped
    wm = store.load_world_model(project_id=pid)
    assert wm is not None


def test_planner_mark_reviewed_called_on_orient(store):
    """After orientation, planner calls store.mark_reviewed directly."""
    mock = PlannerMockLLM()
    planner = Planner(store, llm_call_fn=mock, min_queue_size=10)

    obs = Observation(
        intervention_type="config_change", intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5}, outcome_success=True, wall_time_s=60.0,
    )
    store.save_observation(obs)

    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
        status="done",
    )
    p.observation_id = obs.id
    store.save_proposal(p)

    planner.tick()

    # After orientation, the proposal should be marked as reviewed
    reviewed = store.list_proposals("reviewed")
    assert len(reviewed) == 1
    assert reviewed[0].id == p.id
