"""Tests for multi-project support in Store."""
import json
import pytest
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


# --- Project CRUD ---

def test_create_project(store):
    pid = store.create_project(
        name="NanoGPT",
        description="Optimize NanoGPT training hyperparameters",
        domain_config={"name": "NanoGPT training", "parameters": "DEPTH, MATRIX_LR"},
    )
    assert pid is not None
    project = store.get_project(pid)
    assert project["name"] == "NanoGPT"
    assert project["description"] == "Optimize NanoGPT training hyperparameters"
    assert project["domain_config"]["name"] == "NanoGPT training"
    assert project["active"] is True


def test_list_projects(store):
    store.create_project(name="NanoGPT", description="GPT training")
    store.create_project(name="Atari", description="RL game optimization")
    projects = store.list_projects()
    assert len(projects) == 2
    names = {p["name"] for p in projects}
    assert names == {"NanoGPT", "Atari"}


def test_update_project(store):
    pid = store.create_project(name="NanoGPT", description="old")
    store.update_project(pid, description="new description", active=False)
    project = store.get_project(pid)
    assert project["description"] == "new description"
    assert project["active"] is False


# --- Project association ---

def test_proposal_with_project(store):
    pid = store.create_project(name="NanoGPT", description="GPT")
    p = Proposal(
        intent="Test lr", rationale="test", expected_learning="lr effect",
        intervention_type="config_change", intervention_spec={"MATRIX_LR": "0.08"},
    )
    store.save_proposal(p, project_id=pid)
    proposals = store.list_proposals("backlog")
    assert len(proposals) == 1
    assert proposals[0].project_id == pid


def test_proposal_without_project(store):
    """Backwards compat: proposals without project_id still work."""
    p = Proposal(
        intent="Test lr", rationale="test", expected_learning="lr effect",
        intervention_type="config_change", intervention_spec={"MATRIX_LR": "0.08"},
    )
    store.save_proposal(p)
    proposals = store.list_proposals("backlog")
    assert len(proposals) == 1
    assert proposals[0].project_id is None


def test_list_proposals_by_project(store):
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    for i in range(3):
        p = Proposal(
            intent=f"GPT test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p, project_id=pid1)

    for i in range(2):
        p = Proposal(
            intent=f"Atari test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"game": "Breakout"},
        )
        store.save_proposal(p, project_id=pid2)

    all_proposals = store.list_proposals("backlog")
    assert len(all_proposals) == 5

    gpt_proposals = store.list_proposals("backlog", project_id=pid1)
    assert len(gpt_proposals) == 3
    assert all(p.project_id == pid1 for p in gpt_proposals)

    atari_proposals = store.list_proposals("backlog", project_id=pid2)
    assert len(atari_proposals) == 2


def test_observation_with_project(store):
    pid = store.create_project(name="NanoGPT", description="GPT")
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"MATRIX_LR": "0.08"},
        outcome_success=True,
        outcome_metrics={"val_bpb": 1.05},
        wall_time_s=300.0,
    )
    store.save_observation(obs, project_id=pid)
    loaded = store.load_observation(obs.id)
    assert loaded.project_id == pid


def test_list_observations_by_project(store):
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    for i in range(3):
        obs = Observation(
            intervention_type="config_change",
            intervention_spec={"x": str(i)},
            outcome_success=True, wall_time_s=60.0,
        )
        store.save_observation(obs, project_id=pid1)

    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"game": "Breakout"},
        outcome_success=True, wall_time_s=1800.0,
    )
    store.save_observation(obs, project_id=pid2)

    all_obs = store.list_observations()
    assert len(all_obs) == 4

    gpt_obs = store.list_observations(project_id=pid1)
    assert len(gpt_obs) == 3

    atari_obs = store.list_observations(project_id=pid2)
    assert len(atari_obs) == 1


def test_world_model_per_project(store):
    pid1 = store.create_project(name="NanoGPT", description="GPT")
    pid2 = store.create_project(name="Atari", description="RL")

    # Each project gets its own world model
    wm1 = store.load_world_model(project_id=pid1)
    wm1.add_belief(claim="lr=0.04 is best", confidence=0.7, evidence_for=["obs_1"])
    store.save_world_model(wm1, project_id=pid1,
                           delta={"beliefs_added": [{"claim": "lr=0.04 is best"}]})

    wm2 = store.load_world_model(project_id=pid2)
    wm2.add_belief(claim="PPO beats DQN on Breakout", confidence=0.6, evidence_for=["obs_2"])
    store.save_world_model(wm2, project_id=pid2,
                           delta={"beliefs_added": [{"claim": "PPO beats DQN"}]})

    # Load separately — no cross-contamination
    loaded1 = store.load_world_model(project_id=pid1)
    assert len(loaded1.beliefs) == 1
    assert loaded1.beliefs[0]["claim"] == "lr=0.04 is best"

    loaded2 = store.load_world_model(project_id=pid2)
    assert len(loaded2.beliefs) == 1
    assert loaded2.beliefs[0]["claim"] == "PPO beats DQN on Breakout"


def test_world_model_without_project(store):
    """Backwards compat: world model without project_id still works."""
    wm = store.load_world_model()
    assert wm.version >= 1  # Initial seed


def test_claim_respects_active_projects(store):
    pid_active = store.create_project(name="Active", description="running")
    pid_paused = store.create_project(name="Paused", description="paused")
    store.update_project(pid_paused, active=False)

    p1 = Proposal(
        intent="Active project work", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p1.promote("todo")
    store.save_proposal(p1, project_id=pid_active)

    p2 = Proposal(
        intent="Paused project work", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "2"},
    )
    p2.promote("todo")
    store.save_proposal(p2, project_id=pid_paused)

    # Worker should only claim from active projects
    claimed = store.claim_next_todo("worker_0")
    assert claimed is not None
    assert claimed.id == p1.id
    assert claimed.project_id == pid_active

    # No more claimable (paused project skipped)
    claimed2 = store.claim_next_todo("worker_0")
    assert claimed2 is None


def test_count_proposals_by_project(store):
    pid = store.create_project(name="NanoGPT", description="GPT")
    for i in range(3):
        p = Proposal(
            intent=f"test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p, project_id=pid)

    assert store.count_proposals("backlog", project_id=pid) == 3
    assert store.count_proposals("backlog") == 3  # all


def test_cancel_running_proposal(store):
    """Cancel a running proposal: moves it back to backlog."""
    p = Proposal(
        intent="test cancel", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    store.save_proposal(p)

    # Claim it (moves to running)
    claimed = store.claim_next_todo("worker_0")
    assert claimed is not None
    assert store.count_proposals("running") == 1

    # Cancel it
    assert store.cancel_proposal(claimed.id) is True
    assert store.count_proposals("running") == 0
    assert store.count_proposals("backlog") == 1

    # Can't cancel again
    assert store.cancel_proposal(claimed.id) is False


def test_cancel_non_running_proposal_fails(store):
    """Cancel only works on running proposals."""
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    store.save_proposal(p)  # in backlog
    assert store.cancel_proposal(p.id) is False
