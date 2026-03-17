"""Tests for orientation step — LLM-led world model update."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.observation import Observation
from autoresearcher2.v3.orientation import orient, build_orientation_prompt, DELTA_SCHEMA
from autoresearcher2.v3.generator import DomainConfig


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


def test_orientation_prompt_domain_agnostic_with_non_ml_domain():
    """When given a non-ML DomainConfig, the prompt should not contain ML-specific terms."""
    wm = WorldModel()
    wm.add_belief(claim="pH 7.0 is optimal for growth", confidence=0.6, evidence_for=["obs_001"])
    wm.cost_beliefs = {"lab_experiment": {"wall_time_s": 3600}}

    obs = Observation(
        intervention_type="lab_experiment",
        intervention_spec={"pH": "7.5", "temperature_c": "37"},
        outcome_metrics={"colony_count": 150, "outcome": 150},
        outcome_success=True,
        wall_time_s=3500,
    )

    domain = DomainConfig(
        name="microbiology experiment",
        description="We run lab experiments to optimize bacterial colony growth.",
        hardware="incubator + plate reader",
    )

    prompt = build_orientation_prompt(wm, obs, domain=domain)

    # Extract the instruction section (between UPDATE INSTRUCTIONS and REQUIRED OUTPUT)
    # This is where hardcoded ML language would live — data sections may contain
    # field names from the Observation model, which is fine.
    instruction_section = prompt.split("UPDATE INSTRUCTIONS")[1].split("REQUIRED OUTPUT")[0]

    # The instruction text should NOT contain ML-specific terms
    ml_terms = ["GPU", "power monitoring", "Shelly meter", "GPU sensors"]
    for term in ml_terms:
        assert term.lower() not in instruction_section.lower(), (
            f"ML-specific term '{term}' found in orientation instructions for non-ML domain"
        )

    # The prompt SHOULD reference the domain
    assert "microbiology" in prompt.lower() or "lab experiment" in prompt.lower()


def test_orientation_prompt_includes_hardware_hint():
    """When domain has hardware, it should appear in the cost measurement hint."""
    wm = make_world_model()
    obs = make_observation()
    domain = DomainConfig(
        name="ML training",
        description="We train neural networks.",
        hardware="Shelly meter + GPU sensors",
    )
    prompt = build_orientation_prompt(wm, obs, domain=domain)
    assert "Shelly meter + GPU sensors" in prompt


def test_orient_passes_domain_to_prompt():
    """orient() should accept and pass domain config through."""
    wm = make_world_model()
    obs = make_observation()
    domain = DomainConfig(name="chemistry experiment", description="We mix chemicals.")
    captured_prompts = []

    def mock_llm(prompt):
        captured_prompts.append(prompt)
        return {"beliefs_revised": [{"id": "B1", "new_confidence": 0.65, "reason": "test"}]}

    orient(wm, obs, mock_llm, domain=domain)
    assert "chemistry experiment" in captured_prompts[0]


def test_orientation_prompt_default_domain_is_generic():
    """Without a DomainConfig, the prompt should still be domain-agnostic."""
    wm = make_world_model()
    obs = make_observation()

    prompt = build_orientation_prompt(wm, obs)

    # Default prompt should not hardcode ML-specific measurement terms in instructions
    # (They may appear in the data itself, which is fine)
    instruction_section = prompt.split("UPDATE INSTRUCTIONS")[1].split("REQUIRED OUTPUT")[0]
    assert "Shelly meter" not in instruction_section
    assert "GPU sensors" not in instruction_section


def test_delta_schema_has_no_expectations():
    """expectations were never used in the pipeline — removed from schema entirely."""
    assert "expectations_added" not in DELTA_SCHEMA
    assert "expectations_revised" not in DELTA_SCHEMA


def test_delta_schema_has_no_salience_updated():
    """salience_updated was never handled in apply_delta — remove from schema."""
    assert "salience_updated" not in DELTA_SCHEMA
