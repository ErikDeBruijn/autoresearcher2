"""Tests for orientation step — LLM-led world model update."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.orientation import orient, build_orientation_prompt


def make_world_model():
    wm = WorldModel()
    wm.add_belief(
        claim="learning_rate=0.04 is optimal",
        confidence=0.6,
        evidence_for=["obs_001"],
    )
    wm.cost_beliefs = {"config_change": {"wall_time_s": 300}}
    return wm


def make_observation(outcome=1.03, wall_time=312.5):
    return Observation(
        intervention_type="config_change",
        intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04", "WEIGHT_DECAY": "0.1"},
        outcome_metrics={"val_bpb": 2.0 - outcome, "outcome": outcome},
        outcome_success=True,
        wall_time_s=wall_time,
    )


def test_orientation_prompt_has_evidence_first_order():
    """Prompt must follow: facts → state → instructions → schema."""
    wm = make_world_model()
    obs = make_observation()
    prompt = build_orientation_prompt(wm, obs)

    # Facts (observation) should come before state (world model)
    obs_pos = prompt.index("NEW OBSERVATION")
    wm_pos = prompt.index("CURRENT WORLD MODEL")
    inst_pos = prompt.index("UPDATE INSTRUCTIONS")
    schema_pos = prompt.index("REQUIRED OUTPUT")

    assert obs_pos < wm_pos < inst_pos < schema_pos


def test_orientation_applies_delta():
    """After orientation, world model should be updated."""
    wm = make_world_model()
    obs = make_observation()

    # Mock LLM that returns a structured delta
    def mock_llm(prompt):
        return {
            "beliefs_revised": [
                {"id": "B1", "new_confidence": 0.75, "new_evidence_for": [obs.id], "reason": "confirmed by new data"}
            ],
            "cost_beliefs_updated": {"config_change": {"wall_time_s": 312.5}},
        }

    delta = orient(wm, obs, mock_llm)

    assert wm.version == 1
    assert wm.beliefs[0]["confidence"] == 0.75
    assert obs.id in wm.beliefs[0]["evidence_for"]
    assert wm.cost_beliefs["config_change"]["wall_time_s"] == 312.5


def test_orientation_adds_tension():
    wm = make_world_model()
    wm.add_belief(claim="DEPTH doesn't matter", confidence=0.5, evidence_for=["obs_002"])
    obs = make_observation()

    def mock_llm(prompt):
        return {
            "tensions_added": [
                {"beliefs": ["B1", "B2"], "nature": "lr and depth may interact", "salience": "high"}
            ],
        }

    orient(wm, obs, mock_llm)
    assert len(wm.tensions) == 1
    assert wm.tensions[0]["salience"] == "high"


def test_orientation_handles_llm_failure():
    """If LLM fails, world model should not be modified."""
    wm = make_world_model()
    obs = make_observation()
    original_version = wm.version

    def failing_llm(prompt):
        raise RuntimeError("SSH connection failed")

    delta = orient(wm, obs, failing_llm)

    assert delta == {}
    assert wm.version == original_version


def test_orientation_handles_delta_wrapper():
    """Some LLMs may wrap delta in a 'delta' key."""
    wm = make_world_model()
    obs = make_observation()

    def mock_llm(prompt):
        return {
            "delta": {
                "beliefs_revised": [
                    {"id": "B1", "new_confidence": 0.8, "reason": "strong confirmation"}
                ],
            }
        }

    orient(wm, obs, mock_llm)
    assert wm.beliefs[0]["confidence"] == 0.8


def test_orientation_cost_learning():
    """Orientation should update cost beliefs from observed wall_time."""
    wm = make_world_model()
    # Initial cost belief is 300s
    assert wm.cost_beliefs["config_change"]["wall_time_s"] == 300

    obs = make_observation(wall_time=600.0)

    def mock_llm(prompt):
        return {
            "cost_beliefs_updated": {"config_change": {"wall_time_s": 450}},
        }

    orient(wm, obs, mock_llm)
    assert wm.cost_beliefs["config_change"]["wall_time_s"] == 450
