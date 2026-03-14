"""Tests for filesystem → SQLite migration."""
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.migrate import migrate_workspace_to_store
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


@pytest.fixture
def populated_workspace(tmp_path):
    """Workspace with some data in each layer."""
    ws = Workspace(tmp_path / "filesystem")
    ws.init()

    # Add observations
    for i in range(3):
        obs = Observation(
            intervention_type="config_change",
            intervention_spec={"x": str(i)},
            outcome_metrics={"y": float(i)},
            outcome_success=True,
            wall_time_s=60.0,
        )
        ws.save_observation(obs)

    # Add proposals in various stages
    p1 = Proposal(intent="backlog item", rationale="r", expected_learning="e",
                   intervention_type="config_change", intervention_spec={"a": "1"})
    ws.save_proposal(p1)

    p2 = Proposal(intent="todo item", rationale="r", expected_learning="e",
                   intervention_type="probe", intervention_spec={"b": "2"})
    p2.set_critic_decision("accept", rank=1, rationale="good")
    p2.promote("todo")
    ws.save_proposal(p2)

    p3 = Proposal(intent="done item", rationale="r", expected_learning="e",
                   intervention_type="config_change", intervention_spec={"c": "3"},
                   status="done")
    ws.save_proposal(p3)

    # Update world model a few times
    wm = ws.load_world_model()
    wm.add_belief(claim="test belief", confidence=0.7, evidence_for=["obs_1"])
    ws.save_world_model(wm)
    wm.add_belief(claim="another belief", confidence=0.5, evidence_for=["obs_2"])
    ws.save_world_model(wm)

    return ws


def test_migrate_preserves_observations(populated_workspace, tmp_path):
    store = Store(tmp_path / "sqlite" / "research.db")
    summary = migrate_workspace_to_store(populated_workspace, store)

    assert summary["observations"] == 3
    obs_list = store.list_observations()
    assert len(obs_list) == 3
    store.close()


def test_migrate_preserves_proposals(populated_workspace, tmp_path):
    store = Store(tmp_path / "sqlite" / "research.db")
    summary = migrate_workspace_to_store(populated_workspace, store)

    assert summary["proposals"] == 3
    assert store.count_proposals("backlog") == 1
    assert store.count_proposals("todo") == 1
    assert store.count_proposals("done") == 1
    store.close()


def test_migrate_preserves_world_model(populated_workspace, tmp_path):
    store = Store(tmp_path / "sqlite" / "research.db")
    migrate_workspace_to_store(populated_workspace, store)

    wm = store.load_world_model()
    assert len(wm.beliefs) == 2

    history = store.get_world_model_history()
    # seed + history versions + current
    assert len(history) >= 3
    store.close()


def test_migrated_store_works_with_planner(populated_workspace, tmp_path):
    """After migration, planner+worker can operate on the SQLite store."""
    from autoresearcher2.v3.planner import Planner
    from autoresearcher2.v3.worker import Worker

    store = Store(tmp_path / "sqlite" / "research.db")
    migrate_workspace_to_store(populated_workspace, store)

    def mock_llm(prompt):
        if "Generate" in prompt and "proposals" in prompt.lower():
            return {"proposals": [{
                "intent": "post-migration test",
                "rationale": "verify migration works",
                "expected_learning": "system integrity",
                "intervention_type": "probe",
                "intervention_spec": {"check": "true"},
                "estimated_cost": {},
            }]}
        elif "ranking" in prompt.lower() or "evaluate" in prompt.lower():
            return {"rankings": []}
        return {}

    planner = Planner(store, llm_call_fn=mock_llm, min_queue_size=10, n_proposals=1, n_select=1)
    worker = Worker(store, execute_fn=lambda p: {"metrics": {"ok": True}})

    planner.tick()
    result = worker.tick()
    assert result is not None
    assert result["outcome_success"] is True
    store.close()
