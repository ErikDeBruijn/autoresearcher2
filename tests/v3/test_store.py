"""Tests for Store — SQLite-backed research workspace (v4.0)."""
import json
import pytest
from autoresearcher2.v3.store import Store
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.observation import Observation


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "research.db")
    s.init()
    yield s
    s.close()


# --- Layer 1: Observations ---

def test_save_and_load_observation(store):
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"val_bpb": 1.05},
        outcome_success=True,
        wall_time_s=60.0,
        compute_cost=0.50,
    )
    store.save_observation(obs)
    loaded = store.load_observation(obs.id)
    assert loaded is not None
    assert loaded.id == obs.id
    assert loaded.intervention_spec == {"x": "1"}
    assert loaded.outcome_metrics["val_bpb"] == 1.05
    assert loaded.outcome_success is True


def test_list_observations(store):
    for i in range(3):
        obs = Observation(
            intervention_type="config_change",
            intervention_spec={"x": str(i)},
            outcome_success=True,
            wall_time_s=float(i),
        )
        store.save_observation(obs)
    assert len(store.list_observations()) == 3


def test_observation_append_only(store):
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        wall_time_s=60.0,
    )
    store.save_observation(obs)
    # Trying to insert same ID should fail
    with pytest.raises(Exception):
        store.save_observation(obs)


def test_observation_with_artifacts(store):
    """Observations can persist artifact paths through the store."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"game": "Breakout"},
        outcome_metrics={"mean_reward": 3.6},
        outcome_success=True,
        wall_time_s=300.0,
    )
    obs.artifact_paths = {"video": "/tmp/breakout.mp4"}
    store.save_observation(obs)
    loaded = store.load_observation(obs.id)
    assert loaded.artifact_paths == {"video": "/tmp/breakout.mp4"}


def test_observation_without_artifacts_backwards_compat(store):
    """Observations without artifacts still load fine."""
    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_success=True,
        wall_time_s=60.0,
    )
    store.save_observation(obs)
    loaded = store.load_observation(obs.id)
    assert loaded.artifact_paths == {}


# --- Layer 2: World Model ---

def test_initial_world_model(store):
    wm = store.load_world_model()
    assert wm.beliefs == []
    assert wm.version == 1  # SQLite autoincrement starts at 1


def test_save_and_load_world_model(store):
    wm = store.load_world_model()
    wm.add_belief(claim="lr=0.04 is best", confidence=0.7, evidence_for=["obs_1"])
    store.save_world_model(wm, trigger_obs_id="obs_1",
                           delta={"beliefs_added": [{"claim": "lr=0.04 is best"}]},
                           reasoning="New observation supports this")
    loaded = store.load_world_model()
    assert loaded.version == 2
    assert len(loaded.beliefs) == 1
    assert loaded.beliefs[0]["claim"] == "lr=0.04 is best"


def test_world_model_versioning(store):
    wm = store.load_world_model()
    for i in range(3):
        wm.add_belief(claim=f"belief {i}", confidence=0.5, evidence_for=[])
        store.save_world_model(wm, delta={"beliefs_added": [{"claim": f"belief {i}"}]})

    current = store.load_world_model()
    assert current.version == 4  # 1 (seed) + 3 updates

    history = store.get_world_model_history()
    assert len(history) == 4
    assert history[0]["version"] == 1
    assert history[3]["version"] == 4


def test_world_model_delta_traceability(store):
    wm = store.load_world_model()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=["obs_1"])
    delta = {"beliefs_added": [{"claim": "test"}]}
    store.save_world_model(wm, trigger_obs_id="obs_1", delta=delta,
                           reasoning="Observation showed X")

    history = store.get_world_model_history()
    latest = history[-1]
    assert latest["trigger_obs_id"] == "obs_1"
    assert latest["delta"]["beliefs_added"][0]["claim"] == "test"
    assert latest["reasoning"] == "Observation showed X"


# --- Layer 3: Queue ---

def test_save_proposal_to_backlog(store):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    store.save_proposal(p)
    assert store.count_proposals("backlog") == 1


def test_list_proposals(store):
    for i in range(3):
        p = Proposal(
            intent=f"test {i}", rationale="test", expected_learning="test",
            intervention_type="config_change", intervention_spec={"x": str(i)},
        )
        store.save_proposal(p)
    assert len(store.list_proposals("backlog")) == 3


def test_move_proposal_through_stages(store):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    store.save_proposal(p)
    assert store.count_proposals("backlog") == 1

    store.move_proposal(p, "todo")
    assert store.count_proposals("backlog") == 0
    assert store.count_proposals("todo") == 1


def test_claim_next_todo(store):
    p1 = Proposal(
        intent="low priority", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p1.set_critic_decision("accept", rank=2, rationale="ok")
    p1.promote("todo")
    store.save_proposal(p1)

    p2 = Proposal(
        intent="high priority", rationale="test", expected_learning="test",
        intervention_type="probe", intervention_spec={"x": "2"},
    )
    p2.set_critic_decision("accept", rank=1, rationale="best")
    p2.promote("todo")
    store.save_proposal(p2)

    claimed = store.claim_next_todo("worker_0")
    assert claimed is not None
    assert claimed.id == p2.id  # rank=1 claimed first
    assert store.count_proposals("todo") == 1
    assert store.count_proposals("running") == 1


def test_claim_empty_returns_none(store):
    assert store.claim_next_todo("worker_0") is None


def test_complete_proposal(store):
    p = Proposal(
        intent="test", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    p.promote("running")
    store.save_proposal(p)

    obs = Observation(
        intervention_type="config_change",
        intervention_spec={"x": "1"},
        outcome_metrics={"y": 0.5},
        outcome_success=True,
        wall_time_s=60.0,
    )
    store.complete_proposal(p, obs)
    assert store.count_proposals("running") == 0
    assert store.count_proposals("done") == 1
    assert len(store.list_observations()) == 1


def test_proposals_survive_reconnect(tmp_path):
    """v4.0 criterion: data survives process restart."""
    db_path = tmp_path / "research.db"
    store1 = Store(db_path)
    store1.init()
    p = Proposal(
        intent="survives restart", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    store1.save_proposal(p)
    store1.close()

    store2 = Store(db_path)
    proposals = store2.list_proposals("backlog")
    assert len(proposals) == 1
    assert proposals[0].intent == "survives restart"
    store2.close()


def test_reclaim_stale_running(store):
    """Proposals stuck in 'running' after service restart get moved back to 'todo'."""
    import time as _time

    p = Proposal(
        intent="stuck experiment", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    store.save_proposal(p)

    # Simulate claiming and then a stale started_at timestamp
    store.conn.execute(
        "UPDATE queue SET stage = 'running', worker_id = 'dead_worker', started_at = ? WHERE id = ?",
        (_time.time() - 1000, p.id),  # Started 1000s ago
    )
    store.conn.commit()
    assert store.count_proposals("running") == 1

    reclaimed = store.reclaim_stale_running(timeout_s=600)
    assert reclaimed == 1
    assert store.count_proposals("running") == 0
    assert store.count_proposals("todo") == 1


def test_reclaim_does_not_touch_fresh_running(store):
    """Recently started proposals should not be reclaimed."""
    p = Proposal(
        intent="active experiment", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"x": "1"},
    )
    p.promote("todo")
    store.save_proposal(p)
    store.claim_next_todo("active_worker")
    assert store.count_proposals("running") == 1

    reclaimed = store.reclaim_stale_running(timeout_s=600)
    assert reclaimed == 0
    assert store.count_proposals("running") == 1


def test_structured_data_queryable(store):
    """v4.0 criterion: structured fields are queryable."""
    p = Proposal(
        intent="Test lr effect",
        rationale="High uncertainty",
        expected_learning="lr sensitivity",
        intervention_type="config_change",
        intervention_spec={"MATRIX_LR": "0.08"},
        estimated_cost={"cost_to_test": "~5 min"},
    )
    store.save_proposal(p)

    # Query by intervention type directly
    row = store.conn.execute(
        "SELECT * FROM queue WHERE intervention_type = ?", ("config_change",)
    ).fetchone()
    assert row is not None
    assert row["intent"] == "Test lr effect"
    spec = json.loads(row["intervention_spec"])
    assert spec["MATRIX_LR"] == "0.08"


def test_claim_next_todo_with_project_filter(store):
    """CPU/GPU worker separation: claim only proposals for specific projects."""
    # Create two projects
    store.conn.execute(
        "INSERT INTO projects (id, name, active, created_at) VALUES (?, ?, ?, ?)",
        ("proj_atari", "Atari Breakout", 1, 1.0),
    )
    store.conn.execute(
        "INSERT INTO projects (id, name, active, created_at) VALUES (?, ?, ?, ?)",
        ("proj_gpt", "NanoGPT", 1, 1.0),
    )
    store.conn.commit()

    p_atari = Proposal(
        intent="Test Atari config", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"game": "Breakout"},
    )
    p_atari.project_id = "proj_atari"
    p_atari.set_critic_decision("accept", rank=1, rationale="good")
    p_atari.promote("todo")
    store.save_proposal(p_atari)

    p_gpt = Proposal(
        intent="Test GPT lr", rationale="test", expected_learning="test",
        intervention_type="config_change", intervention_spec={"lr": "0.04"},
    )
    p_gpt.project_id = "proj_gpt"
    p_gpt.set_critic_decision("accept", rank=1, rationale="good")
    p_gpt.promote("todo")
    store.save_proposal(p_gpt)

    # CPU worker claims only Atari
    claimed = store.claim_next_todo("cpu_worker", project_ids=["proj_atari"])
    assert claimed is not None
    assert claimed.id == p_atari.id

    # GPU worker claims only NanoGPT
    claimed = store.claim_next_todo("gpu_worker", project_ids=["proj_gpt"])
    assert claimed is not None
    assert claimed.id == p_gpt.id

    # Both claimed — nothing left for either
    assert store.claim_next_todo("cpu_worker", project_ids=["proj_atari"]) is None
    assert store.claim_next_todo("gpu_worker", project_ids=["proj_gpt"]) is None


def test_claim_next_todo_empty_project_ids_returns_none(store):
    """Empty project_ids list means no projects to claim from."""
    p = Proposal(
        intent="orphan", rationale="test", expected_learning="test",
        intervention_type="probe", intervention_spec={"x": "1"},
    )
    p.set_critic_decision("accept", rank=1, rationale="ok")
    p.promote("todo")
    store.save_proposal(p)

    assert store.claim_next_todo("idle_worker", project_ids=[]) is None
    # Proposal still in todo (not claimed)
    assert store.count_proposals("todo") == 1


def test_claim_next_todo_without_filter_gets_all(store):
    """Without project_ids filter, workers claim any project."""
    store.conn.execute(
        "INSERT INTO projects (id, name, active, created_at) VALUES (?, ?, ?, ?)",
        ("proj_any", "AnyProject", 1, 1.0),
    )
    store.conn.commit()

    p = Proposal(
        intent="generic", rationale="test", expected_learning="test",
        intervention_type="probe", intervention_spec={"x": "1"},
    )
    p.project_id = "proj_any"
    p.set_critic_decision("accept", rank=1, rationale="ok")
    p.promote("todo")
    store.save_proposal(p)

    # No filter — claims anything
    claimed = store.claim_next_todo("any_worker")
    assert claimed is not None
    assert claimed.id == p.id
