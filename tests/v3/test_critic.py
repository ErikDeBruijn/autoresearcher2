"""Tests for critic — ordinal ranking of proposals."""
import pytest
from autoresearcher2.v3.world_model import WorldModel
from autoresearcher2.v3.proposal import Proposal
from autoresearcher2.v3.critic import critique_proposals, build_critic_prompt


def make_world_model():
    wm = WorldModel()
    wm.add_belief(claim="lr is dominant", confidence=0.8, evidence_for=["obs_001"])
    wm.add_belief(claim="WD effect is unclear", confidence=0.3, evidence_for=["obs_002"])
    wm.cost_beliefs = {"config_change": {"wall_time_s": 300}}
    return wm


def make_proposals():
    return [
        Proposal(
            id="prop_expensive",
            intent="Full grid search of depth×lr",
            rationale="Comprehensive but expensive",
            expected_learning="Complete picture",
            intervention_type="config_change",
            intervention_spec={"DEPTH": "10", "MATRIX_LR": "0.08"},
            estimated_cost={"cost_to_test": "~30 min GPU"},
        ),
        Proposal(
            id="prop_cheap_probe",
            intent="Quick test of WD effect at optimal lr",
            rationale="B2 has low confidence, cheap to test",
            expected_learning="Whether WD matters at lr=0.04",
            intervention_type="probe",
            intervention_spec={"WEIGHT_DECAY": "0.1", "run_steps": 100},
            estimated_cost={"cost_to_test": "~30s"},
        ),
        Proposal(
            id="prop_redundant",
            intent="Confirm lr=0.04 is good",
            rationale="We already know this but let's be sure",
            expected_learning="Nothing new",
            intervention_type="replication",
            intervention_spec={"DEPTH": "8", "MATRIX_LR": "0.04"},
            estimated_cost={"cost_to_test": "~5 min GPU"},
        ),
    ]


def test_critic_prompt_includes_proposals():
    wm = make_world_model()
    proposals = make_proposals()
    prompt = build_critic_prompt(wm, proposals, n_select=2)
    assert "prop_expensive" in prompt
    assert "prop_cheap_probe" in prompt
    assert "ORDINALLY" in prompt or "ordinally" in prompt.lower()


def test_critic_ranks_and_selects():
    wm = make_world_model()
    proposals = make_proposals()

    def mock_llm(prompt):
        return {
            "rankings": [
                {"proposal_id": "prop_cheap_probe", "decision": "accept", "rank": 1,
                 "rationale": "Cheap probe testing weakest belief"},
                {"proposal_id": "prop_expensive", "decision": "accept", "rank": 2,
                 "rationale": "Informative but expensive"},
                {"proposal_id": "prop_redundant", "decision": "reject", "rank": None,
                 "rationale": "Confirms what we already know with high confidence"},
            ]
        }

    accepted = critique_proposals(wm, proposals, n_select=2, llm_call_fn=mock_llm)

    assert len(accepted) == 2
    # Cheap probe should be ranked first
    assert accepted[0].id == "prop_cheap_probe"
    assert accepted[0].critic["decision"] == "accept"
    assert accepted[0].critic["rank"] == 1
    # Expensive one second
    assert accepted[1].id == "prop_expensive"
    # Redundant one should be rejected
    redundant = next(p for p in proposals if p.id == "prop_redundant")
    assert redundant.critic["decision"] == "reject"


def test_critic_handles_llm_failure():
    """On LLM failure, fall back to first n proposals."""
    wm = make_world_model()
    proposals = make_proposals()

    def failing_llm(prompt):
        raise RuntimeError("SSH failed")

    accepted = critique_proposals(wm, proposals, n_select=2, llm_call_fn=failing_llm)
    assert len(accepted) == 2


def test_critic_handles_empty_proposals():
    wm = make_world_model()
    accepted = critique_proposals(wm, [], n_select=2, llm_call_fn=lambda p: {})
    assert accepted == []


def test_critic_respects_n_select():
    """Even if LLM accepts more, we only return n_select."""
    wm = make_world_model()
    proposals = make_proposals()

    def mock_llm(prompt):
        return {
            "rankings": [
                {"proposal_id": "prop_cheap_probe", "decision": "accept", "rank": 1, "rationale": "best"},
                {"proposal_id": "prop_expensive", "decision": "accept", "rank": 2, "rationale": "good"},
                {"proposal_id": "prop_redundant", "decision": "accept", "rank": 3, "rationale": "ok"},
            ]
        }

    accepted = critique_proposals(wm, proposals, n_select=1, llm_call_fn=mock_llm)
    assert len(accepted) == 1
    assert accepted[0].id == "prop_cheap_probe"
