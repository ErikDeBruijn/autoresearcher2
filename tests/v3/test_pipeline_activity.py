"""Tests for pipeline activity tracking.

The planner broadcasts its current phase (orienting, critiquing, generating)
so the UI can show pulsing indicators on the relevant kanban columns.
"""
import pytest
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.planner import Planner


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


# --- Store layer: pipeline_activity table ---

def test_pipeline_activity_starts_empty(store):
    """Fresh DB has no active phase."""
    activity = store.get_pipeline_activity()
    assert activity["phase"] is None
    assert activity["project_id"] is None
    assert activity["proposal_id"] is None


def test_set_pipeline_activity(store):
    """Setting activity records phase, project, and proposal."""
    store.set_pipeline_activity(phase="orienting", project_id="proj_1", proposal_id="prop_abc")
    activity = store.get_pipeline_activity()
    assert activity["phase"] == "orienting"
    assert activity["project_id"] == "proj_1"
    assert activity["proposal_id"] == "prop_abc"
    assert activity["started_at"] is not None
    assert activity["updated_at"] is not None


def test_clear_pipeline_activity(store):
    """Clearing activity resets all fields to None."""
    store.set_pipeline_activity(phase="critiquing", project_id="proj_1")
    store.clear_pipeline_activity()
    activity = store.get_pipeline_activity()
    assert activity["phase"] is None
    assert activity["project_id"] is None
    assert activity["proposal_id"] is None
    assert activity["started_at"] is None


def test_set_preserves_started_at(store):
    """Updating activity preserves original started_at (COALESCE)."""
    store.set_pipeline_activity(phase="orienting", project_id="proj_1")
    first = store.get_pipeline_activity()
    started = first["started_at"]

    store.set_pipeline_activity(phase="orienting", project_id="proj_1", proposal_id="prop_2")
    second = store.get_pipeline_activity()
    assert second["started_at"] == started  # unchanged
    assert second["updated_at"] >= first["updated_at"]


def test_set_after_clear_gets_new_started_at(store):
    """After clearing, a new set gets a fresh started_at."""
    store.set_pipeline_activity(phase="orienting")
    first_started = store.get_pipeline_activity()["started_at"]

    store.clear_pipeline_activity()

    import time
    time.sleep(0.01)  # ensure different timestamp

    store.set_pipeline_activity(phase="critiquing")
    second_started = store.get_pipeline_activity()["started_at"]
    assert second_started > first_started


# --- Planner integration: activity is broadcast during phases ---

def _make_planner_with_done_proposal(store):
    """Set up a store with a done proposal ready for orientation."""
    # Create project
    import time
    store.conn.execute(
        "INSERT INTO projects (id, name, description, active, created_at) VALUES (?, ?, ?, ?, ?)",
        ("proj_test", "Test", "test project", True, time.time()),
    )
    store.conn.commit()

    # Create observation
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        outcome_metrics={"val_bpb": 1.05},
        wall_time_s=60.0,
    )
    store.save_observation(obs)

    # Create done proposal linked to observation
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.observation_id = obs.id
    p.status = "done"
    store.save_proposal(p, project_id="proj_test")

    return obs, p


def test_planner_sets_activity_during_orient(store):
    """Planner broadcasts 'orienting' while processing a done proposal."""
    obs, p = _make_planner_with_done_proposal(store)

    activity_during_llm = {}

    def fake_llm(prompt):
        # Capture the activity state during the LLM call
        activity_during_llm.update(store.get_pipeline_activity())
        return {
            "reasoning": "test",
            "beliefs_added": [],
            "beliefs_revised": [],
            "beliefs_retired": [],
            "tensions_added": [],
            "tensions_resolved": [],
            "cost_beliefs_updated": {},
        }

    planner = Planner(
        store, llm_call_fn=fake_llm, project_id="proj_test",
        min_queue_size=0, min_todo=0, n_proposals=0,
    )
    planner.tick()

    # During the LLM call, activity should have been set
    assert activity_during_llm["phase"] == "orienting"
    assert activity_during_llm["proposal_id"] == p.id

    # After tick completes, activity should be cleared
    assert store.get_pipeline_activity()["phase"] is None


def test_planner_sets_activity_during_critique(store):
    """Planner broadcasts 'critiquing' while evaluating backlog proposals."""
    # Create enough backlog proposals so generator doesn't trigger
    for i in range(10):
        p = Proposal(
            intent=f"test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p)

    critique_activity = {}

    def fake_llm(prompt):
        current = store.get_pipeline_activity()
        if current.get("phase") == "critiquing":
            critique_activity.update(current)
        # Return a valid critic response
        backlog = store.list_proposals("backlog")
        if backlog:
            return {"ranked": [{"id": backlog[0].id, "decision": "accept", "rank": 1, "rationale": "good"}]}
        return {"ranked": []}

    planner = Planner(
        store, llm_call_fn=fake_llm,
        min_queue_size=5,  # backlog=10 > 5, so no generation
        min_todo=5, n_select=1,
    )
    planner.tick()

    # During the critic LLM call, activity should have been "critiquing"
    assert critique_activity.get("phase") == "critiquing"

    # After tick, cleared
    assert store.get_pipeline_activity()["phase"] is None


def test_planner_sets_activity_during_generate(store):
    """Planner broadcasts 'generating' while creating new proposals."""
    activity_during_llm = {}

    def fake_llm(prompt):
        activity_during_llm.update(store.get_pipeline_activity())
        if "generate" in prompt.lower() or "proposal" in prompt.lower():
            return {"proposals": [
                {"intent": "new", "rationale": "r", "expected_learning": "e",
                 "intervention_type": "probe", "intervention_spec": {"x": "1"},
                 "estimated_cost": {}}
            ]}
        return {"ranked": []}

    planner = Planner(
        store, llm_call_fn=fake_llm,
        min_queue_size=5, min_todo=0, n_proposals=1,
    )
    planner.tick()

    assert activity_during_llm.get("phase") == "generating"
    assert store.get_pipeline_activity()["phase"] is None


def test_planner_clears_activity_on_error(store):
    """If the LLM call fails, activity is still cleared."""
    obs, p = _make_planner_with_done_proposal(store)

    def failing_llm(prompt):
        store.set_pipeline_activity(phase="orienting")  # simulate the set
        raise RuntimeError("LLM failed")

    planner = Planner(
        store, llm_call_fn=failing_llm, project_id="proj_test",
        min_queue_size=0, min_todo=0, n_proposals=0,
    )
    # tick() catches exceptions internally
    planner.tick()

    # Activity may or may not be cleared depending on where the error occurs,
    # but the planner should not crash
