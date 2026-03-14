"""Tests for Workspace — filesystem-based research workspace."""
import json
import pytest
from autoresearcher2.v3.workspace import Workspace
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path / "research")
    workspace.init()
    return workspace


def test_init_creates_directories(ws):
    assert ws.backlog_dir.exists()
    assert ws.todo_dir.exists()
    assert ws.running_dir.exists()
    assert ws.done_dir.exists()
    assert ws.results_dir.exists()
    assert ws.history_dir.exists()
    assert ws.world_model_path.exists()


def test_init_creates_empty_world_model(ws):
    wm = ws.load_world_model()
    assert wm.version == 0
    assert wm.beliefs == []


def test_save_and_load_world_model(ws):
    wm = ws.load_world_model()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=[])
    wm.apply_delta({"cost_beliefs_updated": {"probe": {"wall_time_s": 30}}})
    ws.save_world_model(wm)

    wm2 = ws.load_world_model()
    assert wm2.version == 1
    assert len(wm2.beliefs) == 1
    # History should exist
    assert (ws.history_dir / "world_model_v0.json").exists()


def test_save_proposal_to_backlog(ws):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    ws.save_proposal(p)
    assert (ws.backlog_dir / f"{p.id}.json").exists()


def test_list_proposals(ws):
    for i in range(3):
        p = Proposal(
            intent=f"test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        ws.save_proposal(p)

    proposals = ws.list_proposals("backlog")
    assert len(proposals) == 3


def test_move_proposal_through_stages(ws):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    ws.save_proposal(p)
    assert ws.count_proposals("backlog") == 1

    ws.move_proposal(p, "todo")
    assert ws.count_proposals("backlog") == 0
    assert ws.count_proposals("todo") == 1
    assert p.status == "todo"


def test_claim_next_todo(ws):
    # Add two proposals with different ranks
    p1 = Proposal(
        intent="low priority", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p1.set_critic_decision("accept", rank=2, rationale="ok")
    p1.promote("todo")
    ws.save_proposal(p1)

    p2 = Proposal(
        intent="high priority", rationale="test", expected_learning="test",
        intervention_type="probe", intervention_spec={"x": "2"},
    )
    p2.set_critic_decision("accept", rank=1, rationale="best")
    p2.promote("todo")
    ws.save_proposal(p2)

    claimed = ws.claim_next_todo("worker_0")
    assert claimed is not None
    assert claimed.id == p2.id  # Higher priority (rank=1) claimed first
    assert claimed.status == "running"
    assert ws.count_proposals("todo") == 1
    assert ws.count_proposals("running") == 1


def test_claim_empty_todo_returns_none(ws):
    assert ws.claim_next_todo("worker_0") is None


def test_complete_proposal(ws):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    p.promote("running")
    ws.save_proposal(p)

    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5},
        outcome_success=True,
        wall_time_s=60.0,
    )

    ws.complete_proposal(p, obs)
    assert ws.count_proposals("running") == 0
    assert ws.count_proposals("done") == 1
    assert len(ws.list_observations()) == 1
    assert p.observation_id == obs.id


def test_proposals_survive_reload(ws):
    """v3.1 criterion: proposals survive restart."""
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    ws.save_proposal(p)

    # Simulate restart: new Workspace instance, same root
    ws2 = Workspace(ws.root)
    proposals = ws2.list_proposals("backlog")
    assert len(proposals) == 1
    assert proposals[0].id == p.id
    assert proposals[0].intent == "test"


def test_world_model_versioning(ws):
    """v3.1 criterion: after 3 updates, history has versions 0-2."""
    wm = ws.load_world_model()

    for i in range(3):
        wm.add_belief(claim=f"belief {i}", confidence=0.5, evidence_for=[])
        wm.apply_delta({"cost_beliefs_updated": {f"type_{i}": {"wall_time_s": i * 100}}})
        ws.save_world_model(wm)

    assert (ws.history_dir / "world_model_v0.json").exists()
    assert (ws.history_dir / "world_model_v1.json").exists()
    assert (ws.history_dir / "world_model_v2.json").exists()

    current = ws.load_world_model()
    assert current.version == 3


def test_no_prompt_soup(ws):
    """v3.1 criterion: structured fields are queryable JSON."""
    p = Proposal(
        intent="Test lr effect",
        rationale="High uncertainty",
        expected_learning="lr sensitivity",
        intervention_type="config_change",
        intervention_spec={"MATRIX_LR": "0.08"},
        estimated_cost={"cost_to_test": "~5 min"},
    )
    ws.save_proposal(p)

    # Read raw JSON and verify structured fields
    raw = json.loads((ws.backlog_dir / f"{p.id}.json").read_text())
    assert raw["intent"] == "Test lr effect"
    assert raw["intervention_type"] == "config_change"
    assert raw["intervention_spec"]["MATRIX_LR"] == "0.08"
    assert raw["estimated_cost"]["cost_to_test"] == "~5 min"
