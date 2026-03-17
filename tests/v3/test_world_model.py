"""Tests for WorldModel — structured epistemic state."""
import json
import pytest
from autoresearcher2.v3.world_model import WorldModel


def test_empty_world_model():
    wm = WorldModel()
    assert wm.version == 0
    assert wm.beliefs == []
    assert wm.expectations == []
    assert wm.tensions == []
    assert wm.cost_beliefs == {}


def test_add_belief():
    wm = WorldModel()
    bid = wm.add_belief(
        claim="learning_rate is the dominant factor",
        confidence=0.5,
        evidence_for=["obs_001"],
    )
    assert bid.startswith("B")
    assert len(wm.beliefs) == 1
    assert wm.beliefs[0]["claim"] == "learning_rate is the dominant factor"
    assert wm.beliefs[0]["confidence"] == 0.5
    assert wm.beliefs[0]["evidence_for"] == ["obs_001"]
    assert wm.beliefs[0]["evidence_against"] == []


def test_add_multiple_beliefs_unique_ids():
    wm = WorldModel()
    b1 = wm.add_belief(claim="a", confidence=0.5, evidence_for=[])
    b2 = wm.add_belief(claim="b", confidence=0.5, evidence_for=[])
    assert b1 != b2


def test_add_expectation():
    wm = WorldModel()
    wm.add_expectation(
        condition={"DEPTH": "8", "MATRIX_LR": "0.04"},
        prediction="outcome > 1.03",
        confidence=0.7,
        basis=["B1"],
    )
    assert len(wm.expectations) == 1
    assert wm.expectations[0]["confidence"] == 0.7


def test_add_tension():
    wm = WorldModel()
    tid = wm.add_tension(
        belief_ids=["B1", "B3"],
        nature="B1 claims lr dominant but B3 shows network_size matters at large scale",
        salience="high",
    )
    assert tid.startswith("T")
    assert len(wm.tensions) == 1


def test_serialize_roundtrip():
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.7, evidence_for=["obs_001"])
    wm.add_tension(belief_ids=["B1"], nature="test tension", salience="low")
    wm.cost_beliefs = {"config_change": {"wall_time_s": 300}}
    data = wm.to_dict()
    wm2 = WorldModel.from_dict(data)
    assert wm2.beliefs == wm.beliefs
    assert wm2.tensions == wm.tensions
    assert wm2.cost_beliefs == wm.cost_beliefs


def test_save_and_load(tmp_path):
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.7, evidence_for=[])
    path = tmp_path / "world_model.json"
    wm.save(path)
    wm2 = WorldModel.load(path)
    assert wm2.beliefs[0]["claim"] == "test"
    # Verify it's valid JSON
    data = json.loads(path.read_text())
    assert data["version"] == 0


def test_apply_delta_revise_belief():
    wm = WorldModel()
    bid = wm.add_belief(claim="lr matters", confidence=0.5, evidence_for=["obs_001"])
    delta = {
        "beliefs_revised": [
            {"id": bid, "new_confidence": 0.8, "new_evidence_for": ["obs_003"], "reason": "confirmed"}
        ],
    }
    wm.apply_delta(delta)
    assert wm.version == 1
    assert wm.beliefs[0]["confidence"] == 0.8
    assert "obs_003" in wm.beliefs[0]["evidence_for"]


def test_apply_delta_add_belief():
    wm = WorldModel()
    delta = {
        "beliefs_added": [
            {"claim": "depth has diminishing returns", "confidence": 0.3, "evidence_for": ["obs_002"]}
        ],
    }
    wm.apply_delta(delta)
    assert len(wm.beliefs) == 1
    assert wm.beliefs[0]["claim"] == "depth has diminishing returns"


def test_apply_delta_retire_belief():
    wm = WorldModel()
    bid = wm.add_belief(claim="obsolete claim", confidence=0.1, evidence_for=[])
    delta = {
        "beliefs_retired": [{"id": bid, "reason": "contradicted by all evidence"}],
    }
    wm.apply_delta(delta)
    assert len(wm.beliefs) == 0


def test_apply_delta_tensions():
    wm = WorldModel()
    wm.add_belief(claim="a", confidence=0.5, evidence_for=[])
    wm.add_belief(claim="b", confidence=0.5, evidence_for=[])
    delta = {
        "tensions_added": [
            {"beliefs": ["B1", "B2"], "nature": "contradictory", "salience": "high"}
        ],
    }
    wm.apply_delta(delta)
    assert len(wm.tensions) == 1
    # Now resolve it
    tid = wm.tensions[0]["id"]
    delta2 = {
        "tensions_resolved": [{"id": tid, "resolution": "B1 was wrong", "reasoning": "new data"}],
    }
    wm.apply_delta(delta2)
    assert len(wm.tensions) == 0


def test_apply_delta_cost_beliefs():
    wm = WorldModel()
    delta = {
        "cost_beliefs_updated": {"config_change": {"wall_time_s": 300, "compute_cost": 0.5}},
    }
    wm.apply_delta(delta)
    assert wm.cost_beliefs["config_change"]["wall_time_s"] == 300


def test_save_with_history(tmp_path):
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=[])
    path = tmp_path / "world_model.json"
    history = tmp_path / "history"
    wm.save(path)

    wm.apply_delta({"beliefs_revised": [{"id": "B1", "new_confidence": 0.9, "reason": "updated"}]})
    wm.save(path, history_dir=history)

    assert (history / "world_model_v0.json").exists()
    # Current file should be v1
    current = json.loads(path.read_text())
    assert current["version"] == 1
    # History file should be v0
    old = json.loads((history / "world_model_v0.json").read_text())
    assert old["version"] == 0


def test_apply_delta_missing_fields_is_safe():
    """Delta with only some fields should not crash."""
    wm = WorldModel()
    wm.add_belief(claim="test", confidence=0.5, evidence_for=[])
    # Minimal delta — only cost update
    delta = {"cost_beliefs_updated": {"probe": {"wall_time_s": 30}}}
    wm.apply_delta(delta)
    assert wm.version == 1
    assert len(wm.beliefs) == 1  # unchanged


def test_no_probe_fidelity_attribute():
    """probe_fidelity was never used — it should not exist on WorldModel."""
    wm = WorldModel()
    assert not hasattr(wm, "probe_fidelity")


def test_no_top_level_salience_attribute():
    """Top-level salience dict was never populated — it should not exist on WorldModel.

    Note: individual tensions still have a 'salience' field, which IS used.
    """
    wm = WorldModel()
    assert not hasattr(wm, "salience")
